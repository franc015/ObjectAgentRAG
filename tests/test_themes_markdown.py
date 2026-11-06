"""Thèmes (modèle léger), markdown conservé, métadonnées JSON, filtre par thème, réindexation."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from agentic_search.backend import read_neutral, write_neutral
from agentic_search.clickhouse_backend import ClickHouseBackend, check_documents
from agentic_search.config import load_config
from agentic_search.ingest import (Extracted, build_passages, ingest_corpus, markdown_to_sections,
                                   to_markdown)
from agentic_search.llm import LLM, LLMError, Turn
from agentic_search.themes import DEFAULT_VOCABULARY, ThemeTagger, canonical, parse_answer
from agentic_search.tools import SCHEMAS, Toolbox, build_schemas

from fixtures import build_corpus, fake_fetch
from test_clickhouse_plumbing import FakeClient, Result, check_placeholders

VOC = ["cybersécurité", "data", "CI/CD", "IA agentique", "java"]
TEXT = "Le déploiement continu automatise les livraisons et les tests. " * 12


class Fake(LLM):
    provider, model = "test", "léger"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def turn(self, system, messages, tools):
        self.calls += 1
        self.system, self.prompt = system, messages[0]["content"]
        if self.error:
            raise self.error
        return Turn(text=self.reply, usage={"input": 300, "output": 20})


def ex(title="Pipeline", text=TEXT):
    return Extracted(title=title, sections=[("", text)])


class TestParseAndCanonical(unittest.TestCase):
    def test_reponses_valides(self):
        self.assertEqual(parse_answer('{"themes": ["java", "CI/CD"]}', VOC), ["java", "CI/CD"])
        self.assertEqual(parse_answer('```json\n{"themes": ["ci cd", "CYBERSECURITE", "ia agentique"]}\n```', VOC, 2),
                         ["CI/CD", "cybersécurité"])                     # casse, accents, limite à 2
        self.assertEqual(parse_answer('{"themes": []}', VOC), [])
        self.assertEqual(parse_answer('{"themes": ["java", "Java", "java"]}', VOC), ["java"])

    def test_hors_vocabulaire_ecarte_et_inexploitable(self):
        self.assertEqual(parse_answer('{"themes": ["blockchain", "java"]}', VOC), ["java"])
        for bad in ["", "pas de json", '{"themes": "java"}', '{"autre": 1}', "{cassé"]:
            self.assertIsNone(parse_answer(bad, VOC), bad)
        self.assertEqual(parse_answer('{"themes": [1, null, "data"]}', VOC), ["data"])

    def test_canonical(self):
        self.assertEqual(canonical(["Ci/cd", "JAVA", "zzz"], VOC), (["CI/CD", "java"], ["zzz"]))

    def test_vocabulaire_par_defaut(self):
        for t in ("cybersécurité", "data", "delivery", "CI/CD", "IA agentique", "architecture", "java", "web services"):
            self.assertIn(t, DEFAULT_VOCABULARY)


class TestTagger(unittest.TestCase):
    def test_attribue_les_themes(self):
        llm = Fake('{"themes": ["CI/CD", "java"]}')
        t = ThemeTagger(llm, VOC)
        e = ex()
        t(e, "a.pdf", "pdf")
        self.assertEqual(e.themes, ["CI/CD", "java"])
        self.assertIn('"CI/CD"', llm.prompt)                                # vocabulaire fourni au modèle
        self.assertIn("<document>", llm.prompt)
        self.assertIn("DONNÉE", llm.system)

    def test_cache_et_invalidation_par_le_vocabulaire(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            llm = Fake('{"themes": ["java"]}')
            t = ThemeTagger(llm, VOC, path)
            t(ex(), "a.pdf", "pdf")
            t.save()
            t2 = ThemeTagger(llm, VOC, path)
            e = ex()
            t2(e, "a.pdf", "pdf")
            self.assertEqual((llm.calls, e.themes, t2.cached), (1, ["java"], 1))
            ThemeTagger(llm, VOC + ["cloud"], path)(ex(), "a.pdf", "pdf")
            self.assertEqual(llm.calls, 2)                                  # vocabulaire modifié : nouvel appel

    def test_textes_courts_reponses_invalides_et_disjoncteur(self):
        llm = Fake('{"themes": ["java"]}')
        t = ThemeTagger(llm, VOC)
        short = Extracted(title="x", sections=[("", "trop court")])
        t(short, "a", "pdf")
        self.assertEqual((llm.calls, short.themes), (0, []))
        bad = Fake("n'importe quoi")
        t = ThemeTagger(bad, VOC)
        e = ex()
        t(e, "a", "pdf")
        t(ex(), "a", "pdf")
        self.assertEqual((e.themes, bad.calls, t.failures), ([], 2, 2))      # une réponse invalide n'est pas mise en cache
        down = Fake(error=LLMError("HTTP 529"))
        t = ThemeTagger(down, VOC, max_failures=2)
        for k in range(5):
            e = ex(title=f"t{k}")
            t(e, f"s{k}", "pdf")
        self.assertEqual(down.calls, 2)
        self.assertTrue(any("interrompu" in w for w in e.warnings + ex().warnings) or t.failures == 2)

    def test_aucun_theme_n_etiquette_pas(self):
        t = ThemeTagger(Fake('{"themes": []}'), VOC)
        e = ex()
        t(e, "a", "pdf")
        self.assertEqual((e.themes, t.tagged), ([], 0))


class TestMarkdown(unittest.TestCase):
    def test_aller_retour(self):
        secs = [("", "Chapeau.\nSuite du chapeau."), ("Contexte", "Para 1.\n\nPara 2."), ("Conclusion", "Fin.")]
        md = to_markdown("Mon  titre", secs)
        self.assertTrue(md.startswith("# Mon titre\n"))
        self.assertIn("## Contexte\n\nPara 1.", md)
        back = markdown_to_sections(md)
        self.assertEqual([s for s, _ in back], ["", "Contexte", "Conclusion"])
        self.assertEqual([t.replace("\n\n", "\n") for _, t in back], [t.replace("\n\n", "\n") for _, t in secs])
        self.assertEqual(markdown_to_sections(to_markdown("", [])), [])

    def test_ingestion_avec_markdown_et_metadonnees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_corpus(Path(tmp) / "corpus")
            base, bp, _ = ingest_corpus(root, words=120, overlap=20, fetch=fake_fetch)
            docs, passages, _ = ingest_corpus(root, words=120, overlap=20, fetch=fake_fetch, keep_markdown=True)
            for d in base:                                                  # version de base : format inchangé
                self.assertFalse({"markdown", "meta", "themes", "content_sha256"} & set(d), d["id"])
            self.assertEqual(bp, passages)                                  # mêmes passages avec ou sans markdown
            for d in docs:
                self.assertTrue(d["markdown"].startswith(f"# {d['title']}"))
                self.assertEqual(len(d["content_sha256"]), 64)
                rebuilt = build_passages(d["id"], markdown_to_sections(d["markdown"]), 120, 20)
                mine = [p for p in passages if p["doc_id"] == d["id"]]
                self.assertEqual(rebuilt, mine, d["source"])                # réindexation = ingestion
            fichiers = [d for d in docs if d["type"] != "web"]
            self.assertTrue(all(d["meta"]["fichier"] == d["source"] and len(d["meta"]["sha256_fichier"]) == 64
                                and d["meta"]["taille_octets"] > 0 for d in fichiers))
            web = [d for d in docs if d["source"].startswith("http")][0]
            self.assertEqual(web["meta"]["url"], "https://exemple.test/2023/06/14/reassurance")

    def test_format_neutre(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_corpus(Path(tmp) / "corpus")
            docs, passages, _ = ingest_corpus(root, words=120, overlap=20, fetch=fake_fetch, keep_markdown=True)
            out = Path(tmp) / "data"
            write_neutral(out, docs, passages)
            self.assertNotIn("markdown", json.loads((out / "documents.json").read_text(encoding="utf-8"))[0])
            self.assertEqual(len(list((out / "markdown").glob("*.md"))), len(docs))      # un fichier par document
            self.assertEqual((out / "markdown" / f"{docs[0]['id']}.md").read_text(encoding="utf-8"), docs[0]["markdown"])
            back, _ = read_neutral(out)
            self.assertEqual(back, docs)
            base, bp, _ = ingest_corpus(root, words=120, overlap=20, fetch=fake_fetch)
            write_neutral(out, base, bp)                                    # une ingestion sans markdown efface l'ancien
            self.assertEqual(list((out / "markdown").glob("*.md")), [])

    def test_themes_dans_les_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_corpus(Path(tmp) / "corpus")
            tagger = ThemeTagger(Fake('{"themes": ["data"]}'), VOC)
            docs, _, _ = ingest_corpus(root, words=120, overlap=20, fetch=fake_fetch, theme_tagger=tagger)
            self.assertTrue(docs and all(d["themes"] == ["data"] for d in docs))


class TestChargementCommun(unittest.TestCase):
    def test_meme_data_pour_les_deux_backends(self):
        from agentic_search.__main__ import cmd_load
        from agentic_search.memory_backend import MemoryBackend
        with tempfile.TemporaryDirectory() as tmp:
            root = build_corpus(Path(tmp) / "corpus")
            docs, passages, _ = ingest_corpus(root, words=120, overlap=20, fetch=fake_fetch, keep_markdown=True)
            write_neutral(Path(tmp) / "data", docs, passages)
            mem = MemoryBackend(Path(tmp) / "data")
            self.assertEqual(mem.passages, passages)                       # le backend mémoire relit data/
            self.assertTrue(all(mem.docs[d["id"]]["markdown"] == d["markdown"] for d in docs))
            cfg = themed_cfg()
            cfg["data_dir"] = str(Path(tmp) / "data")
            fake = FakeClient()
            import agentic_search.__main__ as cli
            original = cli.open_backend
            cli.open_backend = lambda c: ClickHouseBackend(c, client=fake, database="test_db")
            try:
                cmd_load(None, cfg)
            finally:
                cli.open_backend = original
            ins = {t: r for t, _, c, r in fake.inserts}
            self.assertEqual(len(ins["passages"]), len(passages))          # mêmes passages dans ClickHouse
            self.assertEqual(len(ins["documents_contenu"]), len(docs))


def themed_cfg(markdown=True):
    cfg = load_config()
    cfg["themes"]["mode"] = "llm"
    cfg["ingest"]["keep_markdown"] = markdown
    return cfg


class FakeJsonClient(FakeClient):
    def __init__(self, fail_json=False, meta_type=None, **kw):
        super().__init__(**kw)
        self.fail_json, self.meta_type = fail_json, meta_type

    def command(self, sql):
        if self.fail_json and "meta JSON" in sql:
            raise RuntimeError("Unknown type JSON")
        super().command(sql)

    def query(self, sql, parameters=None):
        if "FROM system.columns" in sql and self.meta_type:
            self.queries.append((sql, parameters or {}))
            return Result([(self.meta_type,)])
        return super().query(sql, parameters)


DOC = {"id": "aaaaaaaa", "title": "Pipeline CI/CD", "type": "web", "source": "https://x.test/a", "author": "",
       "date": "2025-02-03", "date_origine": "meta_web", "n_passages": 2, "n_words": 30,
       "themes": ["CI/CD", "java"], "meta": {"url": "https://x.test/a", "site": "X"},
       "content_sha256": "f" * 64,
       "markdown": "# Pipeline CI/CD\n\nPremier paragraphe sur le déploiement continu.\n\n"
                   "## Tests\n\nLes tests automatisés tournent à chaque commit.\n"}


class TestClickHouseThemesMarkdown(unittest.TestCase):
    def make(self, cfg=None, **kw):
        self.fake = FakeJsonClient(**kw)
        return ClickHouseBackend(cfg or themed_cfg(), client=self.fake, database="test_db")

    def test_schema_et_capacites(self):
        b = self.make()
        sql = "\n".join(self.fake.commands)
        self.assertIn("themes Array(LowCardinality(String)), meta JSON, content_sha256 String", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS `test_db`.`documents_contenu`", sql)
        self.assertIn("markdown String CODEC(ZSTD(3))", sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS themes", sql)               # migration d'une base existante
        self.assertEqual(b.capabilities, {"themes", "markdown"})
        self.assertEqual(self.make(load_config()).capabilities, frozenset())  # configuration de base : rien d'ajouté
        self.assertNotIn("UPDATE", sql.replace("UPDATE_", ""))             # pas de mise à jour légère

    def test_repli_json_vers_string(self):
        b = self.make(fail_json=True)
        self.assertFalse(b.meta_json)
        b.load([], [DOC])
        cols, rows = [(c, r) for t, _, c, r in self.fake.inserts if t == "documents"][0]
        self.assertEqual(json.loads(rows[0][cols.index("meta")]), DOC["meta"])   # texte JSON en repli

    def test_le_type_reel_de_la_colonne_fait_foi(self):
        self.assertFalse(self.make(meta_type="String").meta_json)
        self.assertTrue(self.make(meta_type="JSON(max_dynamic_paths=1024)").meta_json)

    def test_load_stocke_themes_meta_et_markdown(self):
        b = self.make()
        p = build_passages("aaaaaaaa", markdown_to_sections(DOC["markdown"]))
        b.load(p, [DOC])
        by = {t: (c, r) for t, _, c, r in self.fake.inserts}
        cols, rows = by["documents"]
        self.assertEqual(rows[0][cols.index("themes")], ["CI/CD", "java"])
        self.assertEqual(rows[0][cols.index("meta")], DOC["meta"])           # dictionnaire pour une colonne JSON
        self.assertEqual(rows[0][cols.index("content_sha256")], "f" * 64)
        self.assertEqual(rows[0][cols.index("date_pub")], date(2025, 2, 3))
        ccols, crows = by["documents_contenu"]
        self.assertEqual(crows[0][ccols.index("markdown")], DOC["markdown"])

    def test_filtre_par_theme(self):
        b = self.make()
        self.fake.handlers = [("count(), avg(len)", [(3, 6.0)])]
        b.search("déploiement", themes=["CI/CD", "java"])
        sql, params = [q for q in self.fake.queries if "AS p" in q[0]][0]
        self.assertIn("hasAny(themes, {ths:Array(String)})", sql)
        self.assertEqual(params["ths"], ["CI/CD", "java"])
        check_placeholders(self, sql, params)
        b.list_documents(themes=["data"])
        sql, params = self.fake.queries[-1]
        self.assertIn("hasAny(themes, {ths:Array(String)})", sql)
        check_placeholders(self, sql, params)
        b.list_documents()
        self.assertNotIn("hasAny", self.fake.queries[-1][0])

    def test_set_themes_reinsere_une_version(self):
        b = self.make()
        self.fake.handlers = [("FROM `test_db`.`documents` FINAL WHERE id =", [(
            "aaaaaaaa", "Pipeline", "web", "https://x.test/a", "", date(2025, 2, 3), "meta_web", 2, 30, "n",
            ["java"], {"site": "X"}, "abc")])]
        self.assertTrue(b.set_themes("aaaaaaaa", ["data"]))
        cols, rows = [(c, r) for t, _, c, r in self.fake.inserts if t == "documents"][-1]
        self.assertEqual((rows[0][cols.index("themes")], rows[0][cols.index("meta")]), (["data"], {"site": "X"}))
        self.assertEqual(rows[0][cols.index("date_pub")], date(2025, 2, 3))
        self.assertFalse([c for c in self.fake.commands if c.startswith("UPDATE") or " UPDATE " in c])

    def test_get_document_par_tranches_utf8(self):
        b = self.make()
        self.fake.handlers = [("SELECT count() FROM `test_db`.`passages`", [(2,)]),
                              ("substringUTF8", [(120, "tranche")]),
                              ("FROM `test_db`.`documents` FINAL WHERE id IN",
                               [("aaaaaaaa", "Pipeline", "web", "https://x.test/a", date(2025, 2, 3), 2, "meta_web", ["java"])])]
        got = b.get_document("aaaaaaaa", 10, 7)
        self.assertEqual((got["texte"], got["caracteres_total"], got["suite"], got["themes"]), ("tranche", 120, True, ["java"]))
        sql, params = [q for q in self.fake.queries if "substringUTF8" in q[0]][0]
        self.assertIn("lengthUTF8(markdown)", sql)
        self.assertEqual((params["o"], params["l"]), (10, 7))
        check_placeholders(self, sql, params)
        self.fake.handlers = []
        self.assertIsNone(b.get_document("zzzzzzzz"))

    def _export_handlers(self, with_md=True):
        row = ("aaaaaaaa", "Pipeline CI/CD", "web", "https://x.test/a", "", date(2025, 2, 3), "meta_web", 2, 30,
               ["CI/CD", "java"], {"url": "https://x.test/a", "site": "X"}, "f" * 64)
        return [("FROM `test_db`.`documents_contenu` FINAL", [("aaaaaaaa", DOC["markdown"])] if with_md else []),
                ("themes, meta, content_sha256 FROM `test_db`.`documents` FINAL ORDER BY id", [row])]

    def test_reindex_reconstruit_sans_toucher_aux_documents(self):
        b = self.make()
        self.fake.handlers = self._export_handlers()
        res = b.reindex(120, 20)
        expected = build_passages("aaaaaaaa", markdown_to_sections(DOC["markdown"]), 120, 20)
        self.assertEqual(res, {"documents": 1, "passages": len(expected)})
        truncated = [c for c in self.fake.commands if c.startswith("TRUNCATE")]
        self.assertEqual(truncated, ["TRUNCATE TABLE `test_db`.`passages`", "TRUNCATE TABLE `test_db`.`postings`"])
        self.assertNotIn("documents_contenu", [t for t, *_ in self.fake.inserts])
        ptable = [r for t, _, c, r in self.fake.inserts if t == "passages"][0]
        self.assertEqual(len(ptable), len(expected))

    def test_reindex_refuse_sans_markdown(self):
        b = self.make()
        self.fake.handlers = self._export_handlers(with_md=False)
        with self.assertRaises(ValueError):
            b.reindex()
        self.assertFalse([c for c in self.fake.commands if c.startswith("TRUNCATE")])

    def test_stats_themes(self):
        b = self.make()
        self.fake.handlers = [("arrayJoin(themes)", [("java", 3), ("data", 1)]),
                              ("SELECT count() FROM `test_db`.`passages`", [(9,)]),
                              ("SELECT count() FROM `test_db`.`documents_figures` FINAL", [(6,)]),
                              ("countIf(empty(themes))", [(None, None, 2, 4)]),
                              ("FROM `test_db`.`documents_contenu` FINAL", [(5,)]),
                              ("SELECT count() FROM `test_db`.`documents` FINAL", [(7,)])]
        st = b.stats()
        self.assertEqual((st["par_theme"], st["sans_theme"], st["avec_markdown"], st["figures"]),
                         ({"java": 3, "data": 1}, 4, 5, 6))

    def test_selftest_documents(self):
        b = self.make()
        self.fake.handlers = [("SELECT count() FROM `test_db`.`passages`", [(2,)]),
                              ("substringUTF8", [(len(DOC["markdown"]), DOC["markdown"][5:45])]),
                              ("FROM `test_db`.`documents` FINAL WHERE hasAny", [
            ("aaaaaaaa", "Pipeline CI/CD", "web", "https://x.test/a", date(2025, 2, 3), "meta_web", 2, ["CI/CD", "java"])])
        ] + self._export_handlers(with_md=True)
        passages = build_passages("aaaaaaaa", markdown_to_sections(DOC["markdown"]), 250, 40)
        lines = check_documents(b, [DOC], passages, load_config())
        self.assertTrue(lines[0].startswith("OK"), lines)
        self.assertTrue(any("réindexation" in l and l.startswith("OK") for l in lines), lines)
        self.assertTrue(any("filtre par thème 'CI/CD'" in l and l.startswith("OK") for l in lines), lines)
        altered = dict(DOC, themes=["data"])
        self.assertTrue(check_documents(b, [altered], passages, load_config())[0].startswith("ÉCART"))


class Backend:
    def __init__(self, caps=()):
        self.capabilities = frozenset(caps)
        self.calls = []

    def search(self, query, **kw):
        self.calls.append(("search", kw))
        return [{"id": "aaaaaaaa-0000", "doc_id": "aaaaaaaa", "title": "T", "type": "web", "date": "2025-02-03",
                 "section": "", "source": "s", "score": 1.0, "extrait": "…", "themes": ["java"]}]

    def list_documents(self, **kw):
        self.calls.append(("list", kw))
        return []

    def get_document(self, doc_id, offset=0, length=6000):
        self.calls.append(("doc", (doc_id, offset, length)))
        return {"id": doc_id, "texte": "x", "suite": False} if doc_id == "aaaaaaaa" else None


class TestOutils(unittest.TestCase):
    def names(self, schemas):
        return [s["name"] for s in schemas]

    def test_schemas_selon_les_capacites(self):
        self.assertEqual(build_schemas(frozenset()), SCHEMAS)               # base : outils inchangés
        full = build_schemas({"themes", "markdown"}, VOC)
        self.assertEqual(self.names(full), ["search", "get_passage", "list_documents", "get_document"])
        prop = full[0]["parameters"]["properties"]["themes"]
        self.assertEqual(prop["items"]["enum"], VOC)
        self.assertIn("themes", full[2]["parameters"]["properties"])
        self.assertNotIn("themes", SCHEMAS[0]["parameters"]["properties"])  # le schéma d'origine n'est pas modifié

    def test_themes_transmis_seulement_si_gere(self):
        b = Backend()
        Toolbox(b).call("search", {"query": "x", "themes": ["java"]})
        self.assertNotIn("themes", b.calls[0][1])
        b = Backend({"themes"})
        out = Toolbox(b).call("search", {"query": "x", "themes": ["java", " "]})
        self.assertEqual(b.calls[0][1]["themes"], ["java"])
        self.assertIn('"themes": ["java"]', out.text)
        Toolbox(b).call("list_documents", {"themes": "data"})
        self.assertEqual(b.calls[1][1]["themes"], ["data"])

    def test_get_document(self):
        b = Backend({"markdown"})
        tb = Toolbox(b)
        out = tb.call("get_document", {"doc_id": "aaaaaaaa", "offset": "3", "length": 99999})
        self.assertEqual(b.calls[0][1], ("aaaaaaaa", 3, 8000))              # tranche plafonnée
        self.assertIn('"suite": false', out.text)
        self.assertIn("inconnu", tb.call("get_document", {"doc_id": "zzzzzzzz"}).text)
        self.assertIn("outil inconnu", Toolbox(Backend()).call("get_document", {"doc_id": "aaaaaaaa"}).text)


if __name__ == "__main__":
    unittest.main()
