"""Service HTTP minimal (bibliothèque standard) : /health, /search, /ask."""

import hmac
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .agent import run_agent
from .llm import LLMError

log = logging.getLogger("agentic_search.server")
MAX_BODY = 1_000_000


def make_server(cfg: dict, backend, tracer, toolbox, llm_factory, host=None, port=None) -> ThreadingHTTPServer:
    token = os.environ.get(cfg["server"].get("token_env", ""), "")
    max_turns = cfg["agent"]["max_turns"]

    class Handler(BaseHTTPRequestHandler):
        server_version = "agentic-search"

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not token:
                return True
            given = self.headers.get("Authorization", "")
            return hmac.compare_digest(given, f"Bearer {token}")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self._send(200, {"status": "ok", "backend": backend.name})
            if not self._authorized():
                return self._send(401, {"erreur": "authentification requise"})
            if url.path == "/stats":
                return self._send(200, backend.stats())
            if url.path == "/search":
                q = parse_qs(url.query)
                g = lambda k: (q.get(k) or [None])[0]
                try:
                    hits = backend.search(g("q") or "", limit=int(g("limit") or 5), doc_type=g("type"),
                                          date_min=g("from"), date_max=g("to"))
                except ValueError as e:
                    return self._send(400, {"erreur": str(e)})
                return self._send(200, {"résultats": hits})
            self._send(404, {"erreur": "inconnu"})

        def do_POST(self):
            if urlparse(self.path).path != "/ask":
                return self._send(404, {"erreur": "inconnu"})
            if not self._authorized():
                return self._send(401, {"erreur": "authentification requise"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_BODY:
                    return self._send(413, {"erreur": "corps absent ou trop volumineux"})
                body = json.loads(self.rfile.read(n))
                question = str(body["question"]).strip()
            except (ValueError, KeyError, json.JSONDecodeError):
                return self._send(400, {"erreur": 'attendu : {"question": "..."}'})
            if not question or len(question) > 4000:
                return self._send(400, {"erreur": "question vide ou trop longue (4000 caractères max)"})
            try:
                llm = llm_factory(body.get("provider"), body.get("model"))
                res = run_agent(question, llm, toolbox, tracer, max_turns=max_turns)
            except LLMError as e:
                return self._send(502, {"erreur": str(e)})
            sources = []
            for pid in res.cited:
                got = backend.get_passages(pid)
                if got:
                    sources.append({"id": pid, "titre": got[0]["title"], "date": got[0]["date"],
                                    "source": got[0]["source"]})
            self._send(200, {"réponse": res.answer, "statut": res.status, "tours": res.turns,
                             "jetons": res.usage, "run_id": res.run_id, "sources": sources})

    return ThreadingHTTPServer((host or cfg["server"]["host"], port if port is not None else cfg["server"]["port"]),
                               Handler)
