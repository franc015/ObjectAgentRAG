import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from agentic_search.dates import DateResolver, SYSTEM, parse_answer
from agentic_search.ingest import Extracted, ingest_corpus
from agentic_search.llm import LLM, LLMError, Turn

from fixtures import build_corpus, fake_fetch

TEXT = "Le marché de l'assurance évolue vite. " * 20


def answer(d, conf="haute", indice="publié le"):
    return Turn(text=json.dumps({"date": d, "confiance": conf, "indice": indice}),
                usage={"input": 400, "output": 30})


class Fake(LLM):
    provider, model = "test", "léger"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def turn(self, system, messages, tools):
        self.calls.append((system, messages[0]["content"], tools))
        if self.error:
            raise self.error
        return self.reply() if callable(self.reply) else self.reply


def ex(date_=None, origin="", title="Rapport"):
    return Extracted(title=title, date=date_, date_origin=origin, sections=[("", TEXT)])


class TestParseAnswer(unittest.TestCase):
    def test_valides(self):
        today = date(2026, 11, 6)
        self.assertEqual(parse_answer('{"date": "2024-03-12", "confiance": "haute"}', today), ("2024-03-12", "haute"))
        self.assertEqual(parse_answer('Voici : {"date": "2024-03", "confiance": "moyenne"} merci', today), ("2024-03-01", "moyenne"))
        self.assertEqual(parse_answer('```json\n{"date": "2019", "confiance": "haute"}\n```', today), ("2019-01-01", "haute"))

    def test_rejets(self):
        today = date(2026, 11, 6)
        for bad in ['{"date": null, "confiance": "haute"}', '{"date": "2027-01-01", "confiance": "haute"}',
                    '{"date": "1500-01-01", "confiance": "haute"}', '{"date": "2024-13-45", "confiance": "haute"}',
                    '{"date": "hier", "confiance": "haute"}', '{"date": "2024-03-12", "confiance": "certaine"}',
                    '{"date": 20240312, "confiance": "haute"}', "pas de json", "", "{cassé"]:
            self.assertEqual(parse_answer(bad, today)[0], None, bad)


class TestResolver(unittest.TestCase):
    def test_remplace_la_date_de_fichier(self):
        llm = Fake(answer("2024-03-12"))
        r = DateResolver(llm)
        e = ex("2026-11-06", "metadonnee_fichier")
        r(e, "rapport.pdf", "pdf")
        self.assertEqual((e.date, e.date_origin), ("2024-03-12", "modele"))
        system, prompt, tools = llm.calls[0]
        self.assertIn("2026-11-06", prompt)                   # la date du fichier est donnée comme indice
        self.assertIn("<document>", prompt)
        self.assertIn("DONNÉE", system)
        self.assertEqual(tools, [])                           # aucun outil pour ce modèle
        self.assertEqual(r.usage, {"input": 400, "output": 30})

    def test_date_manquante_completee(self):
        e = ex()
        DateResolver(Fake(answer("2022-11-03", "moyenne")))(e, "note.md", "text")
        self.assertEqual((e.date, e.date_origin), ("2022-11-03", "modele"))

    def test_dates_fiables_non_touchees(self):
        llm = Fake(answer("1999-01-01"))
        r = DateResolver(llm)
        web, url, epub = ex("2025-01-20", "meta_web"), ex("2023-06-14", "url"), ex("2019-05-01", "metadonnee_fichier")
        r(web, "p.html", "web"); r(url, "https://x/2023/06/14/a", "web"); r(epub, "l.epub", "epub")
        self.assertEqual([web.date, url.date, epub.date], ["2025-01-20", "2023-06-14", "2019-05-01"])
        self.assertEqual(llm.calls, [])

    def test_confiance_basse_ou_null_conserve_l_existant(self):
        for reply in (answer("2001-01-01", "basse"), answer(None, "haute"), Turn(text="je ne sais pas")):
            e = ex("2026-11-06", "metadonnee_fichier")
            DateResolver(Fake(reply))(e, "a.pdf", "pdf")
            self.assertEqual((e.date, e.date_origin), ("2026-11-06", "metadonnee_fichier"))

    def test_texte_trop_court_pas_d_appel(self):
        llm = Fake(answer("2024-01-01"))
        e = Extracted(title="t", sections=[("", "quelques mots")])
        DateResolver(llm)(e, "a.pdf", "pdf")
        self.assertEqual(llm.calls, [])

    def test_extrait_debut_et_fin(self):
        llm = Fake(answer("2024-01-01"))
        long = Extracted(title="t", sections=[("", "DEBUT " + "mot " * 3000 + "FIN")])
        DateResolver(llm, head_chars=100, tail_chars=50)(long, "a.pdf", "pdf")
        prompt = llm.calls[0][1]
        self.assertIn("DEBUT", prompt)
        self.assertIn("FIN", prompt)
        self.assertLess(len(prompt), 700)

    def test_cache(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "dates_cache.json"
            r1 = DateResolver(Fake(answer("2024-03-12")), path)
            r1(ex("2026-11-06", "metadonnee_fichier"), "a.pdf", "pdf")
            r1.save()
            llm2 = Fake(error=AssertionError("ne doit pas être appelé"))
            r2 = DateResolver(llm2, path)
            e = ex("2026-11-06", "metadonnee_fichier")
            r2(e, "a.pdf", "pdf")
            self.assertEqual((e.date, r2.cached, len(llm2.calls)), ("2024-03-12", 1, 0))

    def test_coupe_circuit_si_le_modele_est_indisponible(self):
        llm = Fake(error=LLMError("HTTP 529"))
        r = DateResolver(llm, max_failures=3)
        exs = [ex("2026-11-06", "metadonnee_fichier", title=f"d{i}") for i in range(6)]
        for i, e in enumerate(exs):
            r(e, f"{i}.pdf", "pdf")
        self.assertEqual(len(llm.calls), 3)                   # plus d'appel après 3 échecs consécutifs
        self.assertTrue(all(e.date == "2026-11-06" for e in exs))   # les dates existantes sont conservées
        self.assertTrue(any("indisponible" in w for e in exs for w in e.warnings))


class TestIngestAvecModele(unittest.TestCase):
    def test_integration(self):
        with tempfile.TemporaryDirectory() as t:
            build_corpus(Path(t) / "c")
            llm = Fake(answer("2023-09-14"))
            r = DateResolver(llm)
            docs, _, _ = ingest_corpus(Path(t) / "c", words=120, overlap=20, fetch=fake_fetch, date_resolver=r)
            by = {d["source"]: d for d in docs}
            for name in ("article_habitation.docx", "rapport_inondations.pdf", "note.md"):
                self.assertEqual((by[name]["date"], by[name]["date_origine"]), ("2023-09-14", "modele"), name)
            self.assertEqual(by["page_climat.html"]["date"], "2025-01-20")      # date de la page conservée
            self.assertEqual(by["livre_transformation.epub"]["date"], "2019-05-01")
            self.assertEqual(len(llm.calls), 3)                                  # seulement les cas douteux


if __name__ == "__main__":
    unittest.main()
