"""Ingestion incrémentale : seuls les fichiers nouveaux ou modifiés sont relus."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agentic_search.ingest as ing
from agentic_search.backend import write_neutral
from agentic_search.ingest import ingest_corpus, load_previous, params_id, write_manifest

from fixtures import filler

PID = params_id(v=1, test=True)


class Env:
    """Dossier de corpus + dossier data, avec compteurs d'appels aux lecteurs et à fetch."""

    def __init__(self, tmp: str):
        self.root, self.data = Path(tmp) / "corpus", Path(tmp) / "data"
        self.root.mkdir()
        self.reads: list[str] = []
        self.fetched: list[str] = []

    def write(self, name: str, text: str):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def fetch(self, url, figures=None):
        self.fetched.append(url)
        return ing.extract_html(f"<html><head><title>Page {url[-3:]}</title></head><body><article><p>"
                                f"{filler(60, len(url))}</p></article></body></html>", url)

    def run(self, full=False, pid=PID, **kw):
        previous, reason = (None, "complet") if full else load_previous(self.data, pid)
        report: dict = {}
        real = ing.read_text
        env = self

        def spy(path, figures=None):
            env.reads.append(path.name)
            return real(path, figures=figures) if figures else real(path)

        self.reads.clear()
        self.fetched.clear()
        with mock.patch.dict(ing.READERS, {".md": spy, ".txt": spy}):
            docs, passages, warnings = ingest_corpus(
                self.root, words=60, overlap=10, fetch=self.fetch, keep_markdown=True, previous=previous,
                report=report, pid=pid, **kw)
        write_neutral(self.data, docs, passages)
        write_manifest(self.data, report["manifest"])
        return docs, passages, warnings, report, previous


class TestIncremental(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.e = Env(self.tmp.name)
        self.e.write("a.md", "# Article A\n\n" + filler(150, 1))
        self.e.write("b.txt", "Article B\n\n" + filler(150, 2))
        self.e.write("urls.txt", "# liens\nhttps://exemple.test/p/aaa\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_premiere_puis_deuxieme_ingestion_sans_relecture(self):
        docs1, pass1, _, rep1, prev1 = self.e.run()
        self.assertIsNone(prev1)
        self.assertEqual(sorted(self.e.reads), ["a.md", "b.txt"])
        self.assertEqual(len(rep1["nouveaux"]), 3)
        docs2, pass2, _, rep2, prev2 = self.e.run()
        self.assertIsNotNone(prev2)
        self.assertEqual(self.e.reads, [])                      # aucun fichier relu
        self.assertEqual(self.e.fetched, [])                    # aucun lien retéléchargé
        self.assertEqual((rep2["inchanges"], rep2["nouveaux"], rep2["modifies"], rep2["supprimes"]), (3, [], [], []))
        self.assertEqual((docs2, pass2), (docs1, pass1))        # résultat identique, markdown compris

    def test_ajout_modification_suppression(self):
        self.e.run()
        self.e.write("c.md", "# Article C\n\n" + filler(150, 3))
        self.e.write("a.md", "# Article A modifié\n\n" + filler(200, 9))
        (self.e.root / "b.txt").unlink()
        self.e.write("urls.txt", "https://exemple.test/p/aaa\nhttps://exemple.test/p/bbb\n")
        docs, passages, _, rep, _ = self.e.run()
        self.assertEqual(sorted(self.e.reads), ["a.md", "c.md"])         # b.txt supprimé, a.md modifié, c.md nouveau
        self.assertEqual(self.e.fetched, ["https://exemple.test/p/bbb"])  # seul le nouveau lien est téléchargé
        self.assertEqual((rep["nouveaux"], rep["modifies"], rep["supprimes"], rep["inchanges"]),
                         (["c.md", "https://exemple.test/p/bbb"], ["a.md"], ["b.txt"], 1))
        self.assertEqual(sorted(d["source"] for d in docs), ["a.md", "c.md", "https://exemple.test/p/aaa",
                                                              "https://exemple.test/p/bbb"])
        self.assertEqual({p["doc_id"] for p in passages}, {d["id"] for d in docs})   # plus de passages du supprimé
        a = [d for d in docs if d["source"] == "a.md"][0]
        self.assertIn("Article A modifié", a["markdown"])

    def test_meme_resultat_qu_un_retraitement_complet(self):
        self.e.run()
        self.e.write("c.md", "# Article C\n\n" + filler(150, 3))
        (self.e.root / "a.md").write_text("# A bis\n\n" + filler(120, 5), encoding="utf-8")
        inc_docs, inc_pass, *_ = self.e.run()
        full_docs, full_pass, *_ = self.e.run(full=True)
        self.assertEqual((inc_docs, inc_pass), (full_docs, full_pass))

    def test_les_corrections_manuelles_des_inchanges_sont_conservees(self):
        self.e.run()
        path = self.e.data / "documents.json"
        docs = json.loads(path.read_text(encoding="utf-8"))
        for d in docs:
            if d["source"] == "b.txt":
                d["date"], d["date_origine"] = "2020-02-02", "manuel"
        path.write_text(json.dumps(docs), encoding="utf-8")
        again, *_ = self.e.run()
        b = [d for d in again if d["source"] == "b.txt"][0]
        self.assertEqual((b["date"], b["date_origine"]), ("2020-02-02", "manuel"))

    def test_reglages_modifies_ou_full_retraitent_tout(self):
        self.e.run()
        _, _, _, rep, prev = self.e.run(pid=params_id(v=1, test="autre"))
        self.assertIsNone(prev)                                 # empreinte des réglages différente
        self.assertEqual(sorted(self.e.reads), ["a.md", "b.txt"])
        self.assertEqual(len(rep["nouveaux"]), 3)
        self.e.run(full=True)
        self.assertEqual(sorted(self.e.reads), ["a.md", "b.txt"])
        self.assertEqual(self.e.fetched, ["https://exemple.test/p/aaa"])

    def test_markdown_disparu_force_le_retraitement(self):
        self.e.run()
        next((self.e.data / "markdown").glob("*.md")).unlink()
        self.e.run()
        self.assertEqual(len(self.e.reads), 1)                  # uniquement le document dont le markdown manque

    def test_donnees_illisibles(self):
        self.e.run()
        (self.e.data / "manifest.json").write_text("{cassé", encoding="utf-8")
        prev, reason = load_previous(self.e.data, PID)
        self.assertIsNone(prev)
        self.assertIn("illisible", reason)
        self.assertEqual(load_previous(Path(self.tmp.name) / "vide", PID)[1], "première ingestion")

    def test_figures_d_un_fichier_supprime_ou_modifie_sont_retirees(self):
        figs = self.e.data / "figures"
        self.e.run(keep_figures=True, figures_dir=figs)
        victim = ing.doc_id_for("b.txt")
        (figs / victim).mkdir(parents=True)
        (figs / victim / "x.png").write_bytes(b"x")
        (self.e.root / "b.txt").unlink()
        self.e.run(keep_figures=True, figures_dir=figs)
        self.assertFalse((figs / victim).exists())
        (figs / "../keep.txt").write_text("à ne pas toucher", encoding="utf-8")
        self.assertTrue((self.e.data / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
