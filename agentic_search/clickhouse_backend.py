"""Backend ClickHouse : documents, passages, postings (BM25 en SQL) et traces.

Le classement BM25 est calculé en SQL à partir d'une table de postings
(terme, passage, fréquence, longueur). Les requêtes n'ont pas été exécutées contre un vrai
serveur lors de l'écriture : lancez `agentic-search ch-selftest`, qui les compare au backend
mémoire sur votre corpus avant tout usage.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter
from datetime import date

from .backend import SearchBackend, parse_date_bound
from .text import index_text, make_snippet, normalize, query_terms, raw_tokens, tokenize
from .tracing import RunTrace, Tracer

log = logging.getLogger(__name__)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BATCH = 5000


def connect(cfg: dict, database: str | None = None):
    try:
        import clickhouse_connect
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("installez le client : pip install 'agentic-search[clickhouse]'") from e
    c = cfg["clickhouse"]
    return clickhouse_connect.get_client(
        host=c["host"], port=int(c["port"]), username=c["username"],
        password=os.environ.get(c.get("password_env", ""), ""),
        database=database or "default", secure=bool(c.get("secure", False)))


class ClickHouseBackend(SearchBackend):
    name = "clickhouse"

    def __init__(self, cfg: dict, client=None, database: str | None = None, k1=1.5, b=0.75):
        self.cfg = cfg
        self.db = database or cfg["clickhouse"]["database"]
        if not _IDENT.match(self.db):
            raise ValueError(f"nom de base invalide : {self.db!r}")
        self.k1, self.b = k1, b
        self.client = client or connect(cfg)
        self._lock = threading.Lock()
        self.text_index = True
        self._stats_cache: tuple[int, float] | None = None
        self.ensure_schema()

    # ------------------------------------------------------------------ SQL
    def _sql(self, text: str) -> str:
        for t in ("documents", "passages", "postings", "agent_runs", "agent_events"):
            text = text.replace(f"${t}", f"`{self.db}`.`{t}`")
        return text

    def _q(self, sql, params=None):
        with self._lock:
            return self.client.query(self._sql(sql), parameters=params or {})

    def _cmd(self, sql):
        with self._lock:
            return self.client.command(self._sql(sql))

    def _insert(self, table, rows, cols):
        with self._lock:
            for k in range(0, len(rows), BATCH):
                self.client.insert(table, rows[k:k + BATCH], column_names=cols, database=self.db)

    def ensure_schema(self):
        retention = int(self.cfg["clickhouse"].get("retention_months", 12))
        with self._lock:
            self.client.command(f"CREATE DATABASE IF NOT EXISTS `{self.db}`")
        self._cmd("""CREATE TABLE IF NOT EXISTS $documents (
            id String, title String, type LowCardinality(String), source String, author String,
            date_pub Nullable(Date), date_origine LowCardinality(String),
            n_passages UInt32, n_words UInt32, norm String, version UInt64
        ) ENGINE = ReplacingMergeTree(version) ORDER BY id""")
        base = """CREATE TABLE IF NOT EXISTS $passages (
            id String, doc_id String, position UInt32, section String, text String,
            text_norm String, len UInt32 {index}
        ) ENGINE = MergeTree ORDER BY (doc_id, position)"""
        try:
            self._cmd(base.format(index=", INDEX idx_text text_norm TYPE text(tokenizer = splitByNonAlpha)"))
        except Exception as e:     # serveur antérieur à la 26.2 : pas d'index de texte
            log.warning("index de texte indisponible (%s) : création sans index", str(e)[:120])
            self.text_index = False
            self._cmd(base.format(index=""))
        self._cmd("""CREATE TABLE IF NOT EXISTS $postings (
            term String, passage_id String, doc_id String, tf UInt16, dl UInt32
        ) ENGINE = MergeTree ORDER BY (term, passage_id)""")
        self._cmd(f"""CREATE TABLE IF NOT EXISTS $agent_runs (
            run_id UUID, ts DateTime64(3, 'UTC'), question String, provider LowCardinality(String),
            modele LowCardinality(String), tours UInt8, jetons_in UInt32, jetons_out UInt32,
            duree_ms UInt32, statut LowCardinality(String), reponse String,
            passages_cites Array(String)
        ) ENGINE = MergeTree ORDER BY (ts, run_id)
        TTL toDateTime(ts) + INTERVAL {retention} MONTH""")
        self._cmd(f"""CREATE TABLE IF NOT EXISTS $agent_events (
            run_id UUID, ts DateTime64(3, 'UTC'), etape UInt8, genre LowCardinality(String),
            outil LowCardinality(String), arguments String, passages_ids Array(String),
            n_resultats UInt16, latence_ms UInt32, jetons_in UInt32, jetons_out UInt32
        ) ENGINE = MergeTree ORDER BY (run_id, etape, ts)
        TTL toDateTime(ts) + INTERVAL {retention} MONTH""")

    # ------------------------------------------------------------------ chargement
    def load(self, passages, documents):
        docs = {d["id"]: d for d in documents}
        for t in ("documents", "passages", "postings"):
            self._cmd(f"TRUNCATE TABLE $" + t)
        version = int(time.time())
        rows = []
        for d in documents:
            rows.append([d["id"], d["title"], d["type"], d["source"], d.get("author", ""),
                         date.fromisoformat(d["date"]) if d.get("date") else None,
                         d.get("date_origine", ""), d["n_passages"], d.get("n_words", 0),
                         normalize(f'{d["title"]} {d["source"]}'), version])
        self._insert("documents", rows, ["id", "title", "type", "source", "author", "date_pub",
                                         "date_origine", "n_passages", "n_words", "norm", "version"])
        prow, trow = [], []
        for p in passages:
            full = index_text(docs[p["doc_id"]]["title"], p.get("section", ""), p["text"])
            toks = tokenize(full)
            prow.append([p["id"], p["doc_id"], p["position"], p.get("section", ""), p["text"],
                         normalize(full), len(toks)])
            for term, tf in Counter(toks).items():
                trow.append([term, p["id"], p["doc_id"], min(tf, 65535), len(toks)])
        self._insert("passages", prow, ["id", "doc_id", "position", "section", "text", "text_norm", "len"])
        self._insert("postings", trow, ["term", "passage_id", "doc_id", "tf", "dl"])
        self._stats_cache = None

    # ------------------------------------------------------------------ recherche
    def _corpus_stats(self) -> tuple[int, float]:
        if self._stats_cache is None:
            r = self._q("SELECT count(), avg(len) FROM $passages").result_rows[0]
            self._stats_cache = (int(r[0]), float(r[1] or 1.0))
        return self._stats_cache

    def _doc_conditions(self, doc_type, source_contains, dmin, dmax, params) -> list[str]:
        conds = []
        if doc_type:
            conds.append("type = {dtype:String}")
            params["dtype"] = doc_type
        if source_contains:
            conds.append("position(norm, {src:String}) > 0")
            params["src"] = normalize(source_contains)
        if dmin:
            conds.append("date_pub >= {dmin:Date}")
            params["dmin"] = date.fromisoformat(dmin)
        if dmax:
            conds.append("date_pub <= {dmax:Date}")
            params["dmax"] = date.fromisoformat(dmax)
        return conds

    def search(self, query, limit=5, doc_type=None, source_contains=None,
               date_min=None, date_max=None, max_per_doc=3):
        dmin, dmax = parse_date_bound(date_min), parse_date_bound(date_max, end=True)
        terms, phrases = query_terms(query)
        uniq = sorted(set(terms))
        if not uniq:
            return []
        n, avgdl = self._corpus_stats()
        if n == 0:
            return []
        params = {"terms": uniq, "n": n, "avgdl": avgdl, "k1": self.k1, "b": self.b,
                  "per_doc": int(max_per_doc), "limit": int(limit)}
        filters = ""
        dconds = self._doc_conditions(doc_type, source_contains, dmin, dmax, params)
        if dconds:
            filters += " AND p.doc_id IN (SELECT id FROM $documents FINAL WHERE " + " AND ".join(dconds) + ")"
        if phrases:
            pconds = []
            for i, ph in enumerate(phrases):
                pconds.append(f"position(text_norm, {{ph{i}:String}}) > 0")
                params[f"ph{i}"] = ph
                toks = raw_tokens(ph)
                if self.text_index and toks:
                    pconds.append(f"hasAllTokens(text_norm, {{pt{i}:Array(String)}})")
                    params[f"pt{i}"] = toks
            filters += " AND p.passage_id IN (SELECT id FROM $passages WHERE " + " AND ".join(pconds) + ")"
        sql = f"""
        WITH df AS (SELECT term, count() AS c FROM $postings WHERE term IN {{terms:Array(String)}} GROUP BY term)
        SELECT p.passage_id AS pid, p.doc_id AS did,
               sum(log(1 + ({{n:UInt64}} - df.c + 0.5) / (df.c + 0.5))
                   * p.tf * ({{k1:Float64}} + 1)
                   / (p.tf + {{k1:Float64}} * (1 - {{b:Float64}} + {{b:Float64}} * p.dl / {{avgdl:Float64}}))) AS score
        FROM $postings AS p
        INNER JOIN df ON p.term = df.term
        WHERE p.term IN {{terms:Array(String)}} {filters}
        GROUP BY pid, did
        ORDER BY score DESC, pid ASC
        LIMIT {{per_doc:UInt32}} BY did
        LIMIT {{limit:UInt32}}"""
        ranked = [(r[0], r[1], float(r[2])) for r in self._q(sql, params).result_rows]
        if not ranked:
            return []
        pass_rows = {r[0]: r for r in self._q(
            "SELECT id, doc_id, position, section, text FROM $passages WHERE id IN {ids:Array(String)}",
            {"ids": [r[0] for r in ranked]}).result_rows}
        docs = self._docs_by_id({r[1] for r in ranked})
        out = []
        for pid, did, score in ranked:
            _, _, position, section, text = pass_rows[pid]
            d = docs[did]
            out.append({"id": pid, "doc_id": did, "title": d["title"], "type": d["type"],
                        "source": d["source"], "date": d["date"], "section": section,
                        "position": position, "n_passages": d["n_passages"],
                        "score": round(score, 3), "extrait": make_snippet(text, terms)})
        return out

    def _docs_by_id(self, ids) -> dict:
        rows = self._q("SELECT id, title, type, source, date_pub, n_passages, date_origine "
                       "FROM $documents FINAL WHERE id IN {ids:Array(String)}", {"ids": sorted(ids)}).result_rows
        return {r[0]: {"title": r[1], "type": r[2], "source": r[3],
                       "date": r[4].isoformat() if r[4] else None, "n_passages": r[5],
                       "date_origine": r[6]} for r in rows}

    def get_passages(self, passage_id, before=0, after=0):
        row = self._q("SELECT doc_id, position FROM $passages WHERE id = {id:String}",
                      {"id": passage_id}).result_rows
        if not row:
            return []
        did, pos = row[0]
        rows = self._q("SELECT id, position, section, text FROM $passages WHERE doc_id = {d:String} "
                       "AND position >= {a:Int64} AND position <= {b:Int64} ORDER BY position",
                       {"d": did, "a": int(pos) - before, "b": int(pos) + after}).result_rows
        d = self._docs_by_id({did})[did]
        return [{"id": r[0], "doc_id": did, "title": d["title"], "type": d["type"], "source": d["source"],
                 "date": d["date"], "section": r[2], "position": r[1], "text": r[3]} for r in rows]

    def list_documents(self, contains=None, doc_type=None, date_min=None, date_max=None,
                       order="date_desc", limit=30):
        dmin, dmax = parse_date_bound(date_min), parse_date_bound(date_max, end=True)
        params: dict = {"limit": int(limit)}
        conds = self._doc_conditions(doc_type, contains, dmin, dmax, params)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        sort = {"date_asc": "date_pub ASC NULLS LAST, id", "titre": "title, id"}.get(
            order, "date_pub DESC NULLS LAST, id")
        rows = self._q(f"SELECT id, title, type, source, date_pub, date_origine, n_passages "
                       f"FROM $documents FINAL {where} ORDER BY {sort} LIMIT {{limit:UInt32}}", params).result_rows
        return [{"id": r[0], "title": r[1], "type": r[2], "source": r[3],
                 "date": r[4].isoformat() if r[4] else None, "date_origine": r[5],
                 "n_passages": r[6]} for r in rows]

    def stats(self):
        n_docs = self._q("SELECT count() FROM $documents FINAL").result_rows[0][0]
        n_pass = self._q("SELECT count() FROM $passages").result_rows[0][0]
        types = dict(self._q("SELECT type, count() FROM $documents FINAL GROUP BY type").result_rows)
        lo, hi, undated = self._q("SELECT min(date_pub), max(date_pub), countIf(date_pub IS NULL) "
                                  "FROM $documents FINAL").result_rows[0]
        return {"backend": self.name, "documents": n_docs, "passages": n_pass, "par_type": types,
                "sans_date": undated, "date_min": lo.isoformat() if lo else None,
                "date_max": hi.isoformat() if hi else None, "index_de_texte": self.text_index}

    def set_date(self, doc_id, date_str, origin="manuel"):
        rows = self._q("SELECT id, title, type, source, author, n_passages, n_words, norm "
                       "FROM $documents FINAL WHERE id = {id:String}", {"id": doc_id}).result_rows
        if not rows:
            return False
        r = rows[0]
        iso = parse_date_bound(date_str)
        self._insert("documents", [[r[0], r[1], r[2], r[3], r[4], date.fromisoformat(iso), origin,
                                    r[5], r[6], r[7], int(time.time() * 1000)]],
                     ["id", "title", "type", "source", "author", "date_pub", "date_origine",
                      "n_passages", "n_words", "norm", "version"])
        return True

    def drop_database(self):
        with self._lock:
            self.client.command(f"DROP DATABASE IF EXISTS `{self.db}`")

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class ClickHouseTracer(Tracer):
    """Écrit les exécutions dans agent_runs / agent_events (un lot par exécution)."""

    def __init__(self, backend: ClickHouseBackend):
        self.b = backend

    def record(self, run: RunTrace) -> None:
        self.b._insert("agent_runs", [[
            uuid.UUID(run.run_id), run.ts, run.question, run.provider, run.model, run.turns,
            run.tokens_in, run.tokens_out, run.duration_ms, run.status, run.answer, run.cited]],
            ["run_id", "ts", "question", "provider", "modele", "tours", "jetons_in", "jetons_out",
             "duree_ms", "statut", "reponse", "passages_cites"])
        if run.events:
            self.b._insert("agent_events", [[
                uuid.UUID(run.run_id), e.ts, e.step, e.kind, e.tool, e.arguments, e.passage_ids,
                e.n_results, e.latency_ms, e.tokens_in, e.tokens_out] for e in run.events],
                ["run_id", "ts", "etape", "genre", "outil", "arguments", "passages_ids",
                 "n_resultats", "latence_ms", "jetons_in", "jetons_out"])

    def recent(self, n: int = 10) -> list[dict]:
        rows = self.b._q("SELECT run_id, ts, question, modele, tours, jetons_in, jetons_out, duree_ms, "
                         "statut FROM $agent_runs ORDER BY ts DESC LIMIT {n:UInt32}", {"n": n}).result_rows
        keys = ["run_id", "ts", "question", "model", "turns", "tokens_in", "tokens_out", "duration_ms", "status"]
        return [dict(zip(keys, (str(x) if k in ("run_id", "ts") else x for k, x in zip(keys, r)))) for r in rows]


def selftest(cfg: dict, reference: SearchBackend, queries: list[str] | None = None) -> list[str]:
    """Charge le corpus de `reference` dans une base temporaire et compare les résultats BM25."""
    docs = list(reference.docs.values())          # type: ignore[attr-defined]
    passages = reference.passages                 # type: ignore[attr-defined]
    db = f"agentic_selftest_{uuid.uuid4().hex[:8]}"
    ch = ClickHouseBackend(cfg, database=db)
    report = []
    try:
        ch.load(passages, docs)
        if queries is None:
            freq = Counter(t for t, pl in reference.postings.items() for _ in range(len(pl)))  # type: ignore[attr-defined]
            common = [t for t, _ in freq.most_common(40)]
            queries = [" ".join(common[i:i + 2]) for i in range(0, 12, 2)] + [common[-1]]
            sample = passages[0]["text"].split()[:3]
            queries.append('"' + " ".join(sample) + '"')
        ok = True
        for q in queries:
            a = [(h["id"], h["score"]) for h in reference.search(q, limit=5)]
            b = [(h["id"], h["score"]) for h in ch.search(q, limit=5)]
            same = len(a) == len(b) and all(x[0] == y[0] and abs(x[1] - y[1]) < 0.01 for x, y in zip(a, b))
            ok &= same
            report.append(f"{'OK ' if same else 'ÉCART'} {q!r}: mémoire={a[:3]} clickhouse={b[:3]}")
        report.append("RÉSULTAT : " + ("identique" if ok else "écarts détectés"))
    finally:
        ch.drop_database()
    return report
