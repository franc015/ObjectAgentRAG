"""Tests de la couche ClickHouse avec un faux client.

Ils vérifient la plomberie (schéma, insertions, paramètres liés, mise en forme des résultats),
PAS la sémantique SQL : celle-ci se valide sur un vrai serveur avec `agentic-search ch-selftest`.
"""

import re
import unittest
import uuid
from datetime import date, datetime, timezone

from agentic_search.clickhouse_backend import ClickHouseBackend, ClickHouseTracer
from agentic_search.config import load_config
from agentic_search.text import index_text, tokenize
from agentic_search.tracing import EventTrace, RunTrace

PLACEHOLDER = re.compile(r"\{(\w+):([A-Za-z0-9_()]+)\}")


class Result:
    def __init__(self, rows):
        self.result_rows = rows


class FakeClient:
    def __init__(self, fail_text_index=False):
        self.commands, self.queries, self.inserts = [], [], []
        self.fail_text_index = fail_text_index
        self.handlers = []      # (sous-chaîne SQL, lignes)

    def command(self, sql):
        if self.fail_text_index and "TYPE text(" in sql:
            raise RuntimeError("Unknown index type 'text'")
        self.commands.append(sql)

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters or {}))
        for needle, rows in self.handlers:
            if needle in sql:
                return Result(rows)
        return Result([])

    def insert(self, table, data, column_names=None, database=None):
        assert all(len(r) == len(column_names) for r in data), f"colonnes incohérentes pour {table}"
        self.inserts.append((table, database, column_names, data))

    def close(self):
        pass


DOCS = [
    {"id": "aaaaaaaa", "title": "Inondations en Europe", "type": "pdf", "source": "rapport.pdf", "author": "X",
     "date": "2024-05-01", "date_origine": "meta", "n_passages": 2, "n_words": 20},
    {"id": "bbbbbbbb", "title": "Sans date", "type": "text", "source": "note.md", "author": "",
     "date": None, "date_origine": "", "n_passages": 1, "n_words": 8},
]
PASSAGES = [
    {"id": "aaaaaaaa-0000", "doc_id": "aaaaaaaa", "position": 0, "section": "Intro", "text": "Les inondations ont touché la région."},
    {"id": "aaaaaaaa-0001", "doc_id": "aaaaaaaa", "position": 1, "section": "Intro", "text": "Les assureurs paient les dégâts des eaux."},
    {"id": "bbbbbbbb-0000", "doc_id": "bbbbbbbb", "position": 0, "section": "", "text": "Note interne sur les inondations."},
]


def check_placeholders(test, sql, params):
    names = {m.group(1) for m in PLACEHOLDER.finditer(sql)}
    test.assertEqual(names, set(params), f"paramètres incohérents dans :\n{sql}")
    test.assertNotRegex(sql, r"\$(documents|passages|postings|agent_runs|agent_events)")   # tables résolues
    test.assertNotIn("{{", sql)
    test.assertNotIn("}}", sql)


class TestClickHousePlumbing(unittest.TestCase):
    def make(self, **kw):
        self.fake = FakeClient(**kw)
        return ClickHouseBackend(load_config(), client=self.fake, database="test_db")

    def test_schema(self):
        b = self.make()
        sql = "\n".join(self.fake.commands)
        self.assertIn("CREATE DATABASE IF NOT EXISTS `test_db`", sql)
        for t in ("documents", "passages", "postings", "agent_runs", "agent_events"):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS `test_db`.`{t}`", sql)
        self.assertIn("ReplacingMergeTree(version) ORDER BY id", sql)      # dédoublonnage sur l'identifiant seul
        self.assertIn("TYPE text(tokenizer = splitByNonAlpha)", sql)
        self.assertIn("TTL toDateTime(ts) + INTERVAL 12 MONTH", sql)
        self.assertTrue(b.text_index)

    def test_repli_sans_index_de_texte(self):
        b = self.make(fail_text_index=True)
        self.assertFalse(b.text_index)
        passages_ddl = [c for c in self.fake.commands if "`passages`" in c][0]
        self.assertNotIn("INDEX", passages_ddl)

    def test_nom_de_base_invalide(self):
        with self.assertRaises(ValueError):
            ClickHouseBackend(load_config(), client=FakeClient(), database="x; DROP")

    def test_load(self):
        b = self.make()
        b.load(PASSAGES, DOCS)
        self.assertEqual([c for c in self.fake.commands if c.startswith("TRUNCATE")],
                         [f"TRUNCATE TABLE `test_db`.`{t}`" for t in ("documents", "documents_contenu", "documents_figures", "passages", "postings")])
        by_table = {t: (cols, rows) for t, db, cols, rows in self.fake.inserts}
        self.assertEqual({db for _, db, _, _ in self.fake.inserts}, {"test_db"})
        dcols, drows = by_table["documents"]
        self.assertEqual(drows[0][dcols.index("date_pub")], date(2024, 5, 1))
        self.assertIsNone(drows[1][dcols.index("date_pub")])
        pcols, prows = by_table["passages"]
        self.assertEqual(len(prows), 3)
        tcols, trows = by_table["postings"]
        pid = "aaaaaaaa-0001"
        expected = tokenize(index_text(DOCS[0]["title"], "Intro", PASSAGES[1]["text"]))
        mine = [r for r in trows if r[tcols.index("passage_id")] == pid]
        self.assertEqual(sum(r[tcols.index("tf")] for r in mine), len(expected))
        self.assertTrue(all(r[tcols.index("dl")] == len(expected) for r in mine))
        # ordre des colonnes = ordre des valeurs
        row = prows[1]
        self.assertEqual((row[pcols.index("id")], row[pcols.index("len")]), (pid, len(expected)))
        self.assertNotIn("é", row[pcols.index("text_norm")])

    def test_recherche_avec_filtres(self):
        b = self.make()
        self.fake.handlers = [
            ("count(), avg(len)", [(3, 6.0)]),
            ("FROM `test_db`.`postings` AS p", [("aaaaaaaa-0001", "aaaaaaaa", 1.234)]),
            ("position, section, text FROM `test_db`.`passages` WHERE id IN",
             [("aaaaaaaa-0001", "aaaaaaaa", 1, "Intro", PASSAGES[1]["text"])]),
            ("FROM `test_db`.`documents` FINAL WHERE id IN",
             [("aaaaaaaa", "Inondations en Europe", "pdf", "rapport.pdf", date(2024, 5, 1), 2, "meta", ["java"])]),
        ]
        hits = b.search('assureurs "dégâts des eaux"', limit=3, doc_type="pdf", source_contains="Rapport",
                        date_min="2024", date_max="2024-06")
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual((h["id"], h["date"], h["score"], h["title"]), ("aaaaaaaa-0001", "2024-05-01", 1.234, "Inondations en Europe"))
        self.assertIn("assureurs", h["extrait"])
        for sql, params in self.fake.queries:
            check_placeholders(self, sql, params)
        sql, params = [q for q in self.fake.queries if "AS p" in q[0]][0]
        self.assertIn("LIMIT {per_doc:UInt32} BY did", sql)
        self.assertIn("p.doc_id IN (SELECT id FROM `test_db`.`documents` FINAL WHERE", sql)
        self.assertIn("hasAllTokens(text_norm, {pt0:Array(String)})", sql)
        self.assertEqual(params["ph0"], "degats des eaux")
        self.assertEqual(params["pt0"], ["degats", "des", "eaux"])
        self.assertEqual(params["dmin"], date(2024, 1, 1))
        self.assertEqual(params["dmax"], date(2024, 6, 30))
        self.assertEqual((params["dtype"], params["src"]), ("pdf", "rapport"))
        self.assertEqual(params["terms"], sorted(set(params["terms"])))
        self.assertIn("assureur", params["terms"])
        self.assertEqual((params["n"], params["avgdl"]), (3, 6.0))

    def test_sans_index_pas_de_hasAllTokens(self):
        b = self.make(fail_text_index=True)
        self.fake.handlers = [("count(), avg(len)", [(3, 6.0)])]
        b.search('"dégâts des eaux"')
        sql, params = [q for q in self.fake.queries if "AS p" in q[0]][0]
        self.assertNotIn("hasAllTokens", sql)
        check_placeholders(self, sql, params)

    def test_corpus_vide_et_requete_vide(self):
        b = self.make()
        self.fake.handlers = [("count(), avg(len)", [(0, None)])]
        self.assertEqual(b.search("inondations"), [])
        self.assertEqual(b.search("x"), [])
        with self.assertRaises(ValueError):
            b.search("x", date_min="hier")

    def test_liste_par_date(self):
        b = self.make()
        self.fake.handlers = [("FROM `test_db`.`documents` FINAL", [
            ("aaaaaaaa", "Inondations", "pdf", "rapport.pdf", date(2024, 5, 1), "meta", 2, ["java"]),
            ("bbbbbbbb", "Sans date", "text", "note.md", None, "", 1, [])])]
        docs = b.list_documents(date_min="2024", order="date_asc")
        self.assertEqual([d["date"] for d in docs], ["2024-05-01", None])
        sql, params = self.fake.queries[-1]
        self.assertIn("ORDER BY date_pub ASC NULLS LAST", sql)
        check_placeholders(self, sql, params)
        b.list_documents()
        self.assertIn("ORDER BY date_pub DESC NULLS LAST", self.fake.queries[-1][0])

    def test_set_date_reinsere_une_version_plus_recente(self):
        b = self.make()
        self.fake.handlers = [("FROM `test_db`.`documents` FINAL WHERE id =",
                               [("bbbbbbbb", "Sans date", "text", "note.md", "", None, "", 1, 8, "sans date note.md",
                                 ["java"], {"url": "https://x"}, "abc")])]
        self.assertTrue(b.set_date("bbbbbbbb", "2022-11"))
        table, db, cols, rows = self.fake.inserts[-1]
        self.assertEqual(table, "documents")
        self.assertEqual(rows[0][cols.index("date_pub")], date(2022, 11, 1))
        self.assertEqual(rows[0][cols.index("date_origine")], "manuel")
        self.assertEqual(rows[0][cols.index("themes")], ["java"])             # thèmes et métadonnées conservés
        self.assertEqual(rows[0][cols.index("meta")], {"url": "https://x"})
        self.assertEqual(rows[0][cols.index("content_sha256")], "abc")
        self.assertFalse(self.make().set_date("zzzzzzzz", "2022"))

    def test_tracer(self):
        b = self.make()
        run = RunTrace(question="Q ?", provider="anthropic", model="claude-sonnet-5-5", turns=2, tokens_in=10,
                       tokens_out=5, duration_ms=1200, status="ok", answer="R [aaaaaaaa-0001]",
                       cited=["aaaaaaaa-0001"],
                       events=[EventTrace(step=0, kind="llm", tokens_in=10, tokens_out=5),
                               EventTrace(step=1, kind="tool", tool="search", arguments="{}",
                                          passage_ids=["aaaaaaaa-0001"], n_results=1)])
        ClickHouseTracer(b).record(run)
        runs = [i for i in self.fake.inserts if i[0] == "agent_runs"][0]
        events = [i for i in self.fake.inserts if i[0] == "agent_events"][0]
        self.assertIsInstance(runs[3][0][0], uuid.UUID)
        self.assertEqual(runs[3][0][runs[2].index("passages_cites")], ["aaaaaaaa-0001"])
        self.assertEqual(len(events[3]), 2)
        self.assertEqual(events[3][1][events[2].index("outil")], "search")
        self.assertIsNotNone(runs[3][0][1].tzinfo)       # horodatage en UTC


if __name__ == "__main__":
    unittest.main()
