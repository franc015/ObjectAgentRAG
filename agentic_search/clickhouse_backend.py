"""Backend ClickHouse : documents, passages, postings (BM25 en SQL) et traces.

Chaque document peut aussi conserver son markdown complet (table `documents_contenu`, compressé en ZSTD),
des figures (table `documents_figures` ; les fichiers image restent sur disque), des métadonnées variables
(colonne `meta` de type JSON, repli en String avant la 25.3), des thèmes
(`themes`, tableau filtrable) et une empreinte du contenu. Les corrections passent par de nouvelles
versions de ligne (ReplacingMergeTree) : on n'utilise pas les mises à jour légères (UPDATE), encore en bêta.

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
from .ingest import build_passages, markdown_to_sections
from .text import index_text, make_snippet, normalize, query_terms, raw_tokens, tokenize
from .tracing import RunTrace, Tracer

log = logging.getLogger(__name__)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BATCH = 5000
FIG_COLS = ["doc_id", "idx", "sha256", "mime", "width", "height", "octets", "alt", "legende", "page",
            "chemin", "url", "version"]
DOC_COLS = ["id", "title", "type", "source", "author", "date_pub", "date_origine", "n_passages", "n_words",
            "norm", "themes", "meta", "content_sha256", "version"]


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
        self.meta_json = True
        self.keep_figures = bool(cfg.get("ingest", {}).get("keep_figures"))
        self._stats_cache: tuple[int, float] | None = None
        caps = set()
        if cfg.get("themes", {}).get("mode", "none") != "none":
            caps.add("themes")
        if cfg.get("ingest", {}).get("keep_markdown"):
            caps.add("markdown")
        self.capabilities = frozenset(caps)
        self.ensure_schema()

    # ------------------------------------------------------------------ SQL
    def _sql(self, text: str) -> str:
        for t in ("documents_contenu", "documents_figures", "documents", "passages", "postings", "agent_runs", "agent_events"):   # le plus long d'abord
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
        doc_ddl = """CREATE TABLE IF NOT EXISTS $documents (
            id String, title String, type LowCardinality(String), source String, author String,
            date_pub Nullable(Date), date_origine LowCardinality(String),
            n_passages UInt32, n_words UInt32, norm String,
            themes Array(LowCardinality(String)), meta {meta}, content_sha256 String, version UInt64
        ) ENGINE = ReplacingMergeTree(version) ORDER BY id"""
        try:
            self._cmd(doc_ddl.format(meta="JSON"))
        except Exception as e:     # serveur antérieur à la 25.3 : le type JSON n'est pas recommandé
            log.warning("type JSON indisponible (%s) : métadonnées stockées en String", str(e)[:120])
            self.meta_json = False
            self._cmd(doc_ddl.format(meta="String"))
        for col, typ in (("themes", "Array(LowCardinality(String))"), ("meta", "JSON" if self.meta_json else "String"),
                         ("content_sha256", "String")):        # bases créées avant ces colonnes
            self._cmd(f"ALTER TABLE $documents ADD COLUMN IF NOT EXISTS {col} {typ}")
        actual = self._q("SELECT type FROM system.columns WHERE database = {db:String} AND table = 'documents' "
                         "AND name = 'meta'", {"db": self.db}).result_rows
        if actual:                                  # le type réel de la colonne fait foi
            self.meta_json = str(actual[0][0]).startswith("JSON")
        self._cmd("""CREATE TABLE IF NOT EXISTS $documents_contenu (
            id String, markdown String CODEC(ZSTD(3)), version UInt64
        ) ENGINE = ReplacingMergeTree(version) ORDER BY id""")
        self._cmd("""CREATE TABLE IF NOT EXISTS $documents_figures (
            doc_id String, idx UInt16, sha256 String, mime LowCardinality(String), width UInt32, height UInt32,
            octets UInt32, alt String, legende String, page UInt16, chemin String, url String, version UInt64
        ) ENGINE = ReplacingMergeTree(version) ORDER BY (doc_id, idx)""")
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

    # ------------------------------------------------------------------ documents
    def _meta_in(self, meta) -> object:
        meta = meta or {}
        return meta if self.meta_json else json.dumps(meta, ensure_ascii=False)

    @staticmethod
    def _meta_out(value) -> dict:
        if isinstance(value, dict):
            return value
        try:
            out = json.loads(value) if isinstance(value, (str, bytes)) and value else {}
        except json.JSONDecodeError:
            return {}
        return out if isinstance(out, dict) else {}

    def _doc_values(self, vals: dict, version: int) -> list:
        return [vals["id"], vals["title"], vals["type"], vals["source"], vals.get("author", ""),
                vals.get("date_pub"), vals.get("date_origine", ""), vals["n_passages"], vals.get("n_words", 0),
                normalize(f'{vals["title"]} {vals["source"]}'), list(vals.get("themes") or []),
                self._meta_in(vals.get("meta")), vals.get("content_sha256", ""), version]

    def load(self, passages, documents, reset_documents=True, reset_contents=True):
        """Remplace le contenu. `reset_*=False` : n'efface ni les documents ni le markdown
        (utilisé par `reindex`, qui ne reconstruit que les passages et les postings)."""
        docs = {d["id"]: d for d in documents}
        tables = (["documents"] if reset_documents else []) + (["documents_contenu"] if reset_contents else []) \
            + (["documents_figures"] if reset_contents else []) + ["passages", "postings"]
        for t in tables:
            self._cmd(f"TRUNCATE TABLE $" + t)
        version = int(time.time())
        rows = []
        for d in documents:
            rows.append(self._doc_values({**d, "date_pub": date.fromisoformat(d["date"]) if d.get("date") else None},
                                         version))
        self._insert("documents", rows, DOC_COLS)
        if reset_contents:
            figs = [[d["id"], k, f["sha256"], f["mime"], f["width"], f["height"], f["octets"], f["alt"],
                     f["legende"], f["page"], f["chemin"], f["url"], version]
                    for d in documents for k, f in enumerate(d.get("figures") or [])]
            if figs:
                self._insert("documents_figures", figs, FIG_COLS)
            contents = [[d["id"], d["markdown"], version] for d in documents if d.get("markdown")]
            if contents:
                self._insert("documents_contenu", contents, ["id", "markdown", "version"])
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

    def export_documents(self) -> list[dict]:
        """Tous les documents, avec thèmes, métadonnées et markdown (format de `load`)."""
        rows = self._q("SELECT id, title, type, source, author, date_pub, date_origine, n_passages, n_words, "
                       "themes, meta, content_sha256 FROM $documents FINAL ORDER BY id").result_rows
        md = dict(self._q("SELECT id, markdown FROM $documents_contenu FINAL").result_rows)
        figs: dict[str, list] = {}
        for r in self._q("SELECT doc_id, sha256, mime, width, height, octets, alt, legende, page, chemin, url "
                         "FROM $documents_figures FINAL ORDER BY doc_id, idx").result_rows:
            figs.setdefault(r[0], []).append({"sha256": r[1], "mime": r[2], "width": r[3], "height": r[4],
                                              "octets": r[5], "alt": r[6], "legende": r[7], "page": r[8],
                                              "chemin": r[9], "url": r[10]})
        out = []
        for r in rows:
            d = {"id": r[0], "title": r[1], "type": r[2], "source": r[3], "author": r[4],
                 "date": r[5].isoformat() if r[5] else None, "date_origine": r[6], "n_passages": r[7],
                 "n_words": r[8], "themes": list(r[9]), "meta": self._meta_out(r[10]), "content_sha256": r[11]}
            if r[0] in md:
                d["markdown"] = md[r[0]]
            if r[0] in figs:
                d["figures"] = figs[r[0]]
            out.append(d)
        return out

    def reindex(self, words: int = 250, overlap: int = 40) -> dict:
        """Reconstruit passages et postings à partir du markdown conservé (sans relire les fichiers)."""
        docs = self.export_documents()
        missing = [d["id"] for d in docs if not d.get("markdown")]
        if not docs or missing:
            raise ValueError(f"{len(missing) or 'aucun'} document(s) sans markdown enregistré : "
                             "réindexation impossible (relancez `ingest` avec keep_markdown = true)")
        passages = []
        for d in docs:
            ps = build_passages(d["id"], markdown_to_sections(d["markdown"]), words, overlap, figures=self.keep_figures)
            d["n_passages"], d["n_words"] = len(ps), sum(len(p["text"].split()) for p in ps)
            passages.extend(ps)
        self.load(passages, docs, reset_documents=False, reset_contents=False)
        return {"documents": len(docs), "passages": len(passages)}

    def get_figures(self, doc_id: str) -> list[dict]:
        rows = self._q("SELECT sha256, mime, width, height, octets, alt, legende, page, chemin, url "
                       "FROM $documents_figures FINAL WHERE doc_id = {id:String} ORDER BY idx",
                       {"id": doc_id}).result_rows
        keys = ("sha256", "mime", "width", "height", "octets", "alt", "legende", "page", "chemin", "url")
        return [dict(zip(keys, r)) for r in rows]

    def get_document(self, doc_id: str, offset: int = 0, length: int = 6000) -> dict | None:
        rows = self._q("SELECT lengthUTF8(markdown), substringUTF8(markdown, {o:UInt64} + 1, {l:UInt32}) "
                       "FROM $documents_contenu FINAL WHERE id = {id:String}",
                       {"id": doc_id, "o": int(offset), "l": int(length)}).result_rows
        if not rows:
            return None
        total, chunk = int(rows[0][0]), rows[0][1]
        d = self._docs_by_id({doc_id}).get(doc_id, {})
        return {"id": doc_id, "titre": d.get("title", ""), "date": d.get("date"), "source": d.get("source", ""),
                "themes": d.get("themes", []), "caracteres_total": total, "debut": int(offset),
                "texte": chunk, "suite": int(offset) + len(chunk) < total}

    # ------------------------------------------------------------------ recherche
    def _corpus_stats(self) -> tuple[int, float]:
        if self._stats_cache is None:
            r = self._q("SELECT count(), avg(len) FROM $passages").result_rows[0]
            self._stats_cache = (int(r[0]), float(r[1] or 1.0))
        return self._stats_cache

    def _doc_conditions(self, doc_type, source_contains, dmin, dmax, params, themes=None) -> list[str]:
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
        if themes:
            conds.append("hasAny(themes, {ths:Array(String)})")
            params["ths"] = [str(t) for t in themes]
        return conds

    def search(self, query, limit=5, doc_type=None, source_contains=None,
               date_min=None, date_max=None, max_per_doc=3, themes=None):
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
        dconds = self._doc_conditions(doc_type, source_contains, dmin, dmax, params, themes)
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
                        "score": round(score, 3), "extrait": make_snippet(text, terms),
                        "themes": d["themes"]})
        return out

    def _docs_by_id(self, ids) -> dict:
        rows = self._q("SELECT id, title, type, source, date_pub, n_passages, date_origine, themes "
                       "FROM $documents FINAL WHERE id IN {ids:Array(String)}", {"ids": sorted(ids)}).result_rows
        return {r[0]: {"title": r[1], "type": r[2], "source": r[3],
                       "date": r[4].isoformat() if r[4] else None, "n_passages": r[5],
                       "date_origine": r[6], "themes": list(r[7])} for r in rows}

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
                       order="date_desc", limit=30, themes=None):
        dmin, dmax = parse_date_bound(date_min), parse_date_bound(date_max, end=True)
        params: dict = {"limit": int(limit)}
        conds = self._doc_conditions(doc_type, contains, dmin, dmax, params, themes)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        sort = {"date_asc": "date_pub ASC NULLS LAST, id", "titre": "title, id"}.get(
            order, "date_pub DESC NULLS LAST, id")
        rows = self._q(f"SELECT id, title, type, source, date_pub, date_origine, n_passages, themes "
                       f"FROM $documents FINAL {where} ORDER BY {sort} LIMIT {{limit:UInt32}}", params).result_rows
        return [{"id": r[0], "title": r[1], "type": r[2], "source": r[3],
                 "date": r[4].isoformat() if r[4] else None, "date_origine": r[5],
                 "n_passages": r[6], "themes": list(r[7])} for r in rows]

    def stats(self):
        n_docs = self._q("SELECT count() FROM $documents FINAL").result_rows[0][0]
        n_pass = self._q("SELECT count() FROM $passages").result_rows[0][0]
        types = dict(self._q("SELECT type, count() FROM $documents FINAL GROUP BY type").result_rows)
        lo, hi, undated, untagged = self._q(
            "SELECT min(date_pub), max(date_pub), countIf(date_pub IS NULL), countIf(empty(themes)) "
            "FROM $documents FINAL").result_rows[0]
        par_theme = dict(self._q("SELECT arrayJoin(themes) AS t, count() AS c FROM $documents FINAL "
                                 "GROUP BY t ORDER BY c DESC, t").result_rows)
        n_md = self._q("SELECT count() FROM $documents_contenu FINAL").result_rows[0][0]
        n_fig = self._q("SELECT count() FROM $documents_figures FINAL").result_rows[0][0]
        return {"backend": self.name, "documents": n_docs, "passages": n_pass, "par_type": types,
                "sans_date": undated, "date_min": lo.isoformat() if lo else None,
                "date_max": hi.isoformat() if hi else None, "index_de_texte": self.text_index,
                "avec_markdown": n_md, "figures": n_fig, "par_theme": par_theme, "sans_theme": untagged,
                "metadonnees_json": self.meta_json}

    def _rewrite(self, doc_id: str, **changes) -> bool:
        """Insère une nouvelle version de la ligne du document (pas de UPDATE)."""
        rows = self._q("SELECT id, title, type, source, author, date_pub, date_origine, n_passages, n_words, "
                       "norm, themes, meta, content_sha256 FROM $documents FINAL WHERE id = {id:String}",
                       {"id": doc_id}).result_rows
        if not rows:
            return False
        vals = dict(zip(DOC_COLS[:-1], rows[0]))
        vals["meta"] = self._meta_out(vals["meta"])
        vals.update(changes)
        self._insert("documents", [self._doc_values(vals, int(time.time() * 1000))], DOC_COLS)
        return True

    def set_date(self, doc_id, date_str, origin="manuel"):
        iso = parse_date_bound(date_str)
        return self._rewrite(doc_id, date_pub=date.fromisoformat(iso), date_origine=origin)

    def set_themes(self, doc_id: str, themes: list[str]) -> bool:
        return self._rewrite(doc_id, themes=list(themes))

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


def check_documents(ch: ClickHouseBackend, docs: list[dict], passages: list[dict], cfg: dict) -> list[str]:
    """Vérifie sur un vrai serveur ce que la comptabilité SQL ne montre pas : aller-retour des thèmes,
    des métadonnées JSON et du markdown, filtre par thème, lecture par tranches, réindexation."""
    out = []
    exp = {d["id"]: d for d in ch.export_documents()}
    bad = []
    for d in docs:
        e = exp.get(d["id"])
        if e is None:
            bad.append(f"{d['id']} absent")
            continue
        for key, default in (("themes", []), ("meta", {}), ("content_sha256", ""), ("markdown", None),
                             ("figures", [])):
            if (e.get(key) or default) != (d.get(key) or default):
                bad.append(f"{d['id']}.{key}")
    out.append(f"{'ÉCART' if bad else 'OK '} documents (thèmes, métadonnées {'JSON' if ch.meta_json else 'String'}, "
               f"markdown, figures) : {bad[:5] if bad else f'{len(docs)} identiques'}")
    for t in sorted({t for d in docs for t in d.get("themes", [])})[:3]:
        n_ref = sum(1 for d in docs if t in d.get("themes", []))
        n_ch = len(ch.list_documents(themes=[t], limit=100000))
        out.append(f"{'OK ' if n_ref == n_ch else 'ÉCART'} filtre par thème {t!r} : {n_ch}/{n_ref}")
    with_md = [d for d in docs if d.get("markdown")]
    if with_md:
        d = with_md[0]
        got = ch.get_document(d["id"], 5, 40)
        okd = bool(got) and got["texte"] == d["markdown"][5:45] and got["caracteres_total"] == len(d["markdown"])
        out.append(f"{'OK ' if okd else 'ÉCART'} lecture par tranches (get_document, caractères UTF-8)")
        if len(with_md) == len(docs):
            res = ch.reindex(cfg["ingest"]["passage_words"], cfg["ingest"]["overlap_words"])
            n = ch._q("SELECT count() FROM $passages").result_rows[0][0]
            out.append(f"{'OK ' if n == len(passages) else 'ÉCART'} réindexation depuis le markdown : "
                       f"{res['passages']} passages (ingestion : {len(passages)}, mêmes paramètres de découpage requis)")
    return out


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
        lines = check_documents(ch, docs, passages, cfg)
        report.extend(lines)
        ok &= all(not l.startswith("ÉCART") for l in lines)
        report.append("RÉSULTAT : " + ("identique" if ok else "écarts détectés"))
    finally:
        ch.drop_database()
    return report
