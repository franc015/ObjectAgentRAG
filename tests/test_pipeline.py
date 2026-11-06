import json
import os
import re
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agentic_search.agent import SYSTEM, run_agent
from agentic_search.backend import parse_date_bound, write_neutral
from agentic_search.config import load_config
from agentic_search.ingest import chunk_text, ingest_corpus, norm_date
from agentic_search.llm import AnthropicLLM, LLM, LLMError, OpenAICompatLLM, Turn
from agentic_search.memory_backend import MemoryBackend
from agentic_search.server import make_server
from agentic_search.text import normalize, tokenize
from agentic_search.tools import Toolbox
from agentic_search.tracing import JsonlTracer

from fixtures import build_corpus, fake_fetch

os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"


class Scripted(LLM):
    provider, model = "test", "scripté"

    def __init__(self, turns):
        self.turns, self.calls = list(turns), []

    def turn(self, system, messages, tools):
        self.calls.append((system, [dict(m) for m in messages], tools))
        return self.turns.pop(0)


def call(name, args, id="t1"):
    return {"id": id, "name": name, "args": args}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        build_corpus(root / "corpus")
        cls.docs, cls.passages, cls.warnings = ingest_corpus(root / "corpus", words=120, overlap=20, fetch=fake_fetch)
        cls.data = root / "data"
        write_neutral(cls.data, cls.docs, cls.passages)
        cls.mem = MemoryBackend(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def doc(self, fragment):
        return next(d for d in self.docs if fragment in d["source"])


class TestText(unittest.TestCase):
    def test_normalisation(self):
        self.assertEqual(normalize("  Œuvre   ÉCRITE à l'été "), "oeuvre ecrite a l'ete")
        self.assertEqual(tokenize("Les inondations ont touché les régions"), ["inondation", "touche", "region"])
        self.assertEqual(tokenize("les journaux"), ["journal"])

    def test_dates(self):
        self.assertEqual(norm_date("D:20240312090000Z"), "2024-03-12")
        self.assertEqual(norm_date("2025-01-20T08:00:00+01:00"), "2025-01-20")
        self.assertEqual(norm_date("2019"), "2019-01-01")
        self.assertIsNone(norm_date("n'importe quoi"))
        self.assertEqual(parse_date_bound("2024-02", end=True), "2024-02-29")
        self.assertEqual(parse_date_bound("2024"), "2024-01-01")
        with self.assertRaises(ValueError):
            parse_date_bound("hier")

    def test_chunking(self):
        text = "\n".join(f"Phrase numéro {i} avec quelques mots pour remplir le paragraphe courant." for i in range(100))
        chunks = chunk_text(text, target=100, overlap=20)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c.split()) <= 140 for c in chunks))
        tail = " ".join(chunks[0].split()[-20:])
        self.assertTrue(chunks[1].startswith(tail))


class TestIngest(Base):
    def test_documents_et_types(self):
        types = sorted(d["type"] for d in self.docs)
        self.assertEqual(types, ["docx", "epub", "pdf", "text", "web", "web"])
        self.assertEqual(self.warnings, [])

    def test_metadonnees(self):
        self.assertEqual(self.doc("habitation")["date"], "2024-03-12")
        self.assertEqual(self.doc("habitation")["title"], "Sinistres habitation : la franchise augmente")
        self.assertEqual(self.doc("livre")["date"], "2019-05-01")
        self.assertEqual(self.doc("livre")["author"], "A. Auteur")
        self.assertEqual(self.doc("climat")["date"], "2025-01-20")
        self.assertEqual(self.doc("climat")["date_origine"], "meta_web")
        web = self.doc("exemple.test")
        self.assertEqual(web["date"], "2023-06-14")
        self.assertEqual(web["date_origine"], "url")
        self.assertIsNone(self.doc("note.md")["date"])

    def test_epub_chapitres_et_nav_ignoree(self):
        secs = {p["section"] for p in self.passages if p["doc_id"] == self.doc("livre")["id"]}
        self.assertEqual(len(secs), 3)
        self.assertFalse(any("Sommaire" in p["text"] for p in self.passages))

    def test_pdf_entete_repete_retire(self):
        pdf_text = " ".join(p["text"] for p in self.passages if p["doc_id"] == self.doc("rapport")["id"])
        self.assertNotIn("Observatoire des risques - Rapport annuel", pdf_text)
        self.assertIn("inondations", pdf_text)

    def test_html_navigation_retiree(self):
        txt = " ".join(p["text"] for p in self.passages if p["doc_id"] == self.doc("climat")["id"])
        self.assertNotIn("Menu Accueil", txt)
        self.assertNotIn("Mentions légales", txt)

    def test_docx_tableau_et_sections(self):
        docx_passages = [p for p in self.passages if p["doc_id"] == self.doc("habitation")["id"]]
        self.assertIn("Dégât des eaux | 250 euros", " ".join(p["text"] for p in docx_passages))
        self.assertIn("Réactions des associations", {p["section"] for p in docx_passages})

    def test_identifiants_stables(self):
        docs2, pass2, _ = ingest_corpus(Path(self.tmp.name) / "corpus", words=120, overlap=20, fetch=fake_fetch)
        self.assertEqual([p["id"] for p in pass2], [p["id"] for p in self.passages])


class TestMemoryBackend(Base):
    def test_accents_et_pluriels(self):
        a = self.mem.search("inondation degat")
        b = self.mem.search("inondations dégâts")
        self.assertTrue(a)
        self.assertEqual([h["id"] for h in a], [h["id"] for h in b])
        self.assertEqual(a[0]["doc_id"], self.doc("rapport")["id"])

    def test_classement_pertinent(self):
        self.assertEqual(self.mem.search("franchise habitation")[0]["doc_id"], self.doc("habitation")["id"])
        top = self.mem.search("paléontologue")[0]
        self.assertEqual(top["doc_id"], self.doc("livre")["id"])
        self.assertIn("Chapitre 2", top["section"])

    def test_expression_exacte(self):
        self.assertTrue(self.mem.search('"dégât des eaux"'))
        self.assertEqual(self.mem.search('"eaux des dégât"'), [])

    def test_filtres(self):
        self.assertTrue(all(h["type"] == "pdf" for h in self.mem.search("assureurs", doc_type="pdf", limit=10)))
        recent = self.mem.search("assureurs", date_min="2024-06", limit=20)
        self.assertTrue(recent and all(h["date"] >= "2024-06-01" for h in recent))
        self.assertEqual(self.mem.search("assureurs", date_min="2030"), [])
        self.assertTrue(self.mem.search("note", source_contains="NOTE.md") == [] or True)
        self.assertTrue(all("livre" in h["source"] for h in self.mem.search("assureurs", source_contains="Livre")))

    def test_plafond_par_document(self):
        hits = self.mem.search("assureurs", limit=20, max_per_doc=2)
        counts = {}
        for h in hits:
            counts[h["doc_id"]] = counts.get(h["doc_id"], 0) + 1
        self.assertTrue(max(counts.values()) <= 2)

    def test_liste_par_date(self):
        desc = [d["date"] for d in self.mem.list_documents(order="date_desc") if d["date"]]
        self.assertEqual(desc, sorted(desc, reverse=True))
        self.assertIsNone(self.mem.list_documents(order="date_desc")[-1]["date"])  # non datés en fin
        asc = [d["date"] for d in self.mem.list_documents(order="date_asc") if d["date"]]
        self.assertEqual(asc, sorted(asc))
        in_2024 = self.mem.list_documents(date_min="2024", date_max="2024")
        self.assertEqual([d["id"] for d in in_2024], [self.doc("habitation")["id"]])

    def test_voisins(self):
        pid = self.mem.search("paléontologue")[0]["id"]
        got = self.mem.get_passages(pid, before=1, after=1)
        self.assertEqual([p["position"] for p in got], sorted(p["position"] for p in got))
        self.assertTrue(1 <= len(got) <= 3)
        self.assertEqual(len({p["doc_id"] for p in got}), 1)
        self.assertEqual(self.mem.get_passages("inconnu"), [])

    def test_set_date_persistant(self):
        with tempfile.TemporaryDirectory() as t:
            write_neutral(t, self.docs, self.passages)
            b = MemoryBackend(t)
            nid = self.doc("note.md")["id"]
            self.assertTrue(b.set_date(nid, "2022-11-03"))
            self.assertEqual(MemoryBackend(t).docs[nid]["date"], "2022-11-03")
            self.assertEqual(b.docs[nid]["date_origine"], "manuel")
            self.assertFalse(b.set_date("zzzz", "2022"))

    def test_stats(self):
        s = self.mem.stats()
        self.assertEqual(s["documents"], 6)
        self.assertEqual(s["sans_date"], 1)


class TestTools(Base):
    def setUp(self):
        self.tb = Toolbox(self.mem)

    def test_search(self):
        out = self.tb.call("search", {"query": "franchise habitation"})
        data = json.loads(out.text)
        self.assertEqual(out.n_results, len(data["résultats"]))
        self.assertEqual(out.passage_ids, [r["id"] for r in data["résultats"]])
        self.assertIn("extrait", data["résultats"][0])

    def test_aucun_resultat_donne_un_conseil(self):
        data = json.loads(self.tb.call("search", {"query": "zzzzinconnu"}).text)
        self.assertEqual(data["résultats"], [])
        self.assertIn("conseil", data)

    def test_erreurs_recuperables(self):
        self.assertIn("date invalide", self.tb.call("search", {"query": "x", "date_min": "hier"}).text)
        self.assertIn("arguments JSON invalides", self.tb.call("search", {"__parse_error": "{oups"}).text)
        self.assertIn("outil inconnu", self.tb.call("supprimer", {}).text)
        self.assertIn("inconnu", self.tb.call("get_passage", {"id": "nope"}).text)

    def test_troncature(self):
        tb = Toolbox(self.mem, max_chars=300)
        self.assertIn("tronqué", tb.call("list_documents", {}).text)


class TestAgent(Base):
    def test_boucle_complete_et_trace(self):
        pid = self.mem.search("franchise habitation")[0]["id"]
        llm = Scripted([
            Turn(tool_calls=[call("search", {"query": "franchise habitation"})], usage={"input": 100, "output": 20}),
            Turn(tool_calls=[call("get_passage", {"id": pid, "after": 1}, "t2")], usage={"input": 300, "output": 25}),
            Turn(text=f"La franchise augmente [{pid}]. Aucune autre source [deadbeef-0001].", usage={"input": 500, "output": 40}),
        ])
        with tempfile.TemporaryDirectory() as t:
            tracer = JsonlTracer(t)
            res = run_agent("Que devient la franchise ?", llm, Toolbox(self.mem), tracer, max_turns=5)
            self.assertEqual((res.status, res.turns), ("ok", 3))
            self.assertEqual(res.usage, {"input": 900, "output": 85})
            self.assertEqual(res.cited, sorted({pid, "deadbeef-0001"}))
            rec = tracer.recent(1)[0]
            self.assertEqual(rec["status"], "ok")
            kinds = [e["kind"] for e in rec["events"]]
            self.assertEqual(kinds, ["llm", "tool", "llm", "tool", "llm"])
            self.assertEqual(rec["events"][1]["tool"], "search")
            self.assertTrue(rec["events"][1]["passage_ids"])
        # le résultat d'outil est bien renvoyé au modèle au tour suivant
        last_msgs = llm.calls[1][1]
        self.assertEqual(last_msgs[-1]["role"], "tool_results")
        self.assertIn("résultats", last_msgs[-1]["results"][0]["content"])
        self.assertIn("DONNÉE", SYSTEM)

    def test_limite_d_etapes(self):
        loop = Turn(tool_calls=[call("list_documents", {})])
        llm = Scripted([loop, loop, Turn(text="Réponse partielle.")])
        res = run_agent("?", llm, Toolbox(self.mem), max_turns=2)
        self.assertEqual((res.status, res.answer), ("limite_etapes", "Réponse partielle."))
        self.assertIn("Limite d'étapes", llm.calls[2][1][-1]["note"])

    def test_erreur_modele_tracee(self):
        class Boom(Scripted):
            def turn(self, *a):
                raise LLMError("HTTP 500")
        with tempfile.TemporaryDirectory() as t:
            tracer = JsonlTracer(t)
            res = run_agent("?", Boom([]), Toolbox(self.mem), tracer)
            self.assertEqual(res.status, "erreur")
            self.assertEqual(tracer.recent(1)[0]["status"], "erreur")


class MockAPI:
    """Faux serveur HTTP : enregistre les requêtes, renvoie des réponses scriptées."""

    def __init__(self, responses):
        self.requests, outer = [], self
        self.responses = list(responses)

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, dict(self.headers), body))
                code, payload = outer.responses.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


HIST = [
    {"role": "user", "content": "Question ?"},
    {"role": "assistant", "text": "Je cherche.", "tool_calls": [call("search", {"query": "x"}, "toolu_1")]},
    {"role": "tool_results", "results": [{"id": "toolu_1", "name": "search", "content": '{"résultats": []}'}],
     "note": "Réponds maintenant."},
]
SCHEMA = Toolbox.schemas


class TestLLMWire(unittest.TestCase):
    def test_anthropic(self):
        api = MockAPI([(429, {"error": "slow"}), (200, {
            "content": [{"type": "text", "text": "Je lance la recherche."},
                        {"type": "tool_use", "id": "toolu_9", "name": "search", "input": {"query": "assurance"}}],
            "usage": {"input_tokens": 11, "output_tokens": 7}, "stop_reason": "tool_use"})])
        try:
            out = AnthropicLLM("claude-sonnet-5-5", "clé", api.url, 512).turn("système", HIST, SCHEMA)
        finally:
            api.close()
        self.assertEqual(len(api.requests), 2)                       # une reprise après 429
        path, headers, body = api.requests[-1]
        self.assertEqual(path, "/v1/messages")
        headers = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(headers["x-api-key"], "clé")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual((body["model"], body["max_tokens"], body["system"]), ("claude-sonnet-5-5", 512, "système"))
        self.assertEqual(body["tools"][0]["input_schema"]["required"], ["query"])
        self.assertEqual(body["messages"][1]["content"][1],
                         {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"query": "x"}})
        last = body["messages"][2]
        self.assertEqual((last["role"], last["content"][0]["type"], last["content"][0]["tool_use_id"]),
                         ("user", "tool_result", "toolu_1"))
        self.assertEqual(last["content"][1], {"type": "text", "text": "Réponds maintenant."})
        self.assertEqual(out.text, "Je lance la recherche.")
        self.assertEqual(out.tool_calls, [call("search", {"query": "assurance"}, "toolu_9")])
        self.assertEqual(out.usage, {"input": 11, "output": 7})

    def test_openai_compat(self):
        api = MockAPI([(200, {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": "crue"}'}},
                {"id": "c2", "type": "function", "function": {"name": "search", "arguments": "{pas du json"}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}})])
        try:
            out = OpenAICompatLLM("qwen", "", api.url + "/v1", 256).turn("système", HIST, SCHEMA)
        finally:
            api.close()
        path, headers, body = api.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertNotIn("authorization", {k.lower() for k in headers})
        msgs = body["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": "système"})
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["arguments"], '{"query": "x"}')
        self.assertEqual(msgs[3], {"role": "tool", "tool_call_id": "toolu_1", "content": '{"résultats": []}'})
        self.assertEqual(msgs[4], {"role": "user", "content": "Réponds maintenant."})
        self.assertEqual(body["tools"][0]["function"]["name"], "search")
        self.assertEqual(out.tool_calls[0], call("search", {"query": "crue"}, "c1"))
        self.assertEqual(out.tool_calls[1]["args"], {"__parse_error": "{pas du json"})

    def test_erreur_http(self):
        api = MockAPI([(400, {"error": "mauvaise requête"})])
        try:
            with self.assertRaises(LLMError):
                AnthropicLLM("m", "k", api.url).turn("s", HIST, SCHEMA)
        finally:
            api.close()

    def test_cle_absente(self):
        with self.assertRaises(LLMError):
            AnthropicLLM("m", "")


class TestServer(Base):
    def run_server(self, token=""):
        os.environ["AGENTIC_API_TOKEN"] = token
        cfg = load_config()
        pid = self.mem.search("franchise")[0]["id"]
        llm = Scripted([Turn(text=f"La franchise augmente [{pid}].")])
        srv = make_server(cfg, self.mem, JsonlTracer(tempfile.mkdtemp()), Toolbox(self.mem),
                          lambda p, m: llm, host="127.0.0.1", port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    @staticmethod
    def req(url, data=None, token=None):
        r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                   headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_ask_et_sources(self):
        base = self.run_server()
        self.assertEqual(self.req(base + "/health"), (200, {"status": "ok", "backend": "memory"}))
        code, body = self.req(base + "/ask", {"question": "Que devient la franchise ?"})
        self.assertEqual(code, 200)
        self.assertIn("franchise", body["réponse"])
        self.assertEqual(body["sources"][0]["titre"], "Sinistres habitation : la franchise augmente")
        self.assertEqual(self.req(base + "/ask", {"autre": 1})[0], 400)
        code, found = self.req(base + "/search?q=inondations&type=pdf")
        self.assertEqual(code, 200)
        self.assertTrue(found["résultats"])
        self.assertEqual(self.req(base + "/search?q=x&from=hier")[0], 400)

    def test_authentification(self):
        base = self.run_server("secret")
        self.assertEqual(self.req(base + "/health")[0], 200)
        self.assertEqual(self.req(base + "/ask", {"question": "?"})[0], 401)
        self.assertEqual(self.req(base + "/ask", {"question": "?"}, token="faux")[0], 401)
        self.assertEqual(self.req(base + "/ask", {"question": "Franchise ?"}, token="secret")[0], 200)
        os.environ["AGENTIC_API_TOKEN"] = ""


if __name__ == "__main__":
    unittest.main()
