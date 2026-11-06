"""Backend de base : index BM25 en mémoire (aucune dépendance, aucun serveur)."""

import json
import math
import threading
from collections import Counter, defaultdict
from pathlib import Path

from .backend import SearchBackend, parse_date_bound, read_neutral
from .text import index_text, make_snippet, normalize, query_terms, tokenize


class MemoryBackend(SearchBackend):
    name = "memory"

    def __init__(self, data_dir: str | Path | None = "data", k1: float = 1.5, b: float = 0.75):
        self.data_dir = Path(data_dir) if data_dir else None
        self.k1, self.b = k1, b
        self._lock = threading.Lock()
        self._reset([], [])
        if self.data_dir and (self.data_dir / "passages.jsonl").exists():
            docs, passages = read_neutral(self.data_dir)
            self.load(passages, docs)

    def _reset(self, passages, documents):
        self.passages = passages
        self.docs = {d["id"]: d for d in documents}
        self.pos = {p["id"]: i for i, p in enumerate(passages)}
        self._norm: list[str] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        self.avg = 1.0

    # ------------------------------------------------------------------
    def load(self, passages, documents):
        self._reset(list(passages), list(documents))
        for i, p in enumerate(self.passages):
            full = index_text(self.docs[p["doc_id"]]["title"], p.get("section", ""), p["text"])
            self._norm.append(normalize(full))
            toks = tokenize(full)
            self.lengths.append(len(toks))
            for t, c in Counter(toks).items():
                self.postings[t].append((i, c))
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0

    def _passed(self, doc, doc_type, sc, dmin, dmax) -> bool:
        if doc_type and doc["type"] != doc_type:
            return False
        if sc and sc not in normalize(f'{doc["title"]} {doc["source"]}'):
            return False
        if dmin or dmax:
            d = doc.get("date")
            if not d or (dmin and d < dmin) or (dmax and d > dmax):
                return False
        return True

    def _hit(self, i, score, terms) -> dict:
        p = self.passages[i]
        d = self.docs[p["doc_id"]]
        return {
            "id": p["id"], "doc_id": p["doc_id"], "title": d["title"], "type": d["type"],
            "source": d["source"], "date": d.get("date"), "section": p.get("section", ""),
            "position": p["position"], "n_passages": d["n_passages"],
            "score": round(score, 3), "extrait": make_snippet(p["text"], terms),
        }

    def search(self, query, limit=5, doc_type=None, source_contains=None,
               date_min=None, date_max=None, max_per_doc=3):
        dmin, dmax = parse_date_bound(date_min), parse_date_bound(date_max, end=True)
        terms, phrases = query_terms(query)
        if not terms:
            return []
        sc = normalize(source_contains) if source_contains else None
        n_pass = len(self.passages)
        scores: dict[int, float] = defaultdict(float)
        for t in set(terms):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (n_pass - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        out, per_doc = [], Counter()
        for i, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            p = self.passages[i]
            if not self._passed(self.docs[p["doc_id"]], doc_type, sc, dmin, dmax):
                continue
            if any(ph not in self._norm[i] for ph in phrases):
                continue
            if per_doc[p["doc_id"]] >= max_per_doc:
                continue
            per_doc[p["doc_id"]] += 1
            out.append(self._hit(i, s, terms))
            if len(out) >= limit:
                break
        return out

    def get_passages(self, passage_id, before=0, after=0):
        if passage_id not in self.pos:
            return []
        i = self.pos[passage_id]
        doc_id = self.passages[i]["doc_id"]
        out = []
        for j in range(i - before, i + after + 1):
            if 0 <= j < len(self.passages) and self.passages[j]["doc_id"] == doc_id:
                p, d = self.passages[j], self.docs[doc_id]
                out.append({"id": p["id"], "doc_id": doc_id, "title": d["title"], "type": d["type"],
                            "source": d["source"], "date": d.get("date"),
                            "section": p.get("section", ""), "position": p["position"],
                            "text": p["text"]})
        return out

    def list_documents(self, contains=None, doc_type=None, date_min=None, date_max=None,
                       order="date_desc", limit=30):
        dmin, dmax = parse_date_bound(date_min), parse_date_bound(date_max, end=True)
        sc = normalize(contains) if contains else None
        docs = [d for d in self.docs.values() if self._passed(d, doc_type, sc, dmin, dmax)]
        if order == "titre":
            docs.sort(key=lambda d: (normalize(d["title"]), d["id"]))
        else:
            dated = sorted((d for d in docs if d.get("date")), key=lambda d: (d["date"], d["id"]),
                           reverse=(order != "date_asc"))
            docs = dated + sorted((d for d in docs if not d.get("date")), key=lambda d: d["id"])
        return [{"id": d["id"], "title": d["title"], "type": d["type"], "source": d["source"],
                 "date": d.get("date"), "date_origine": d.get("date_origine", ""),
                 "n_passages": d["n_passages"]} for d in docs[:limit]]

    def stats(self):
        types = Counter(d["type"] for d in self.docs.values())
        dates = sorted(d["date"] for d in self.docs.values() if d.get("date"))
        return {"backend": self.name, "documents": len(self.docs), "passages": len(self.passages),
                "par_type": dict(types), "sans_date": sum(1 for d in self.docs.values() if not d.get("date")),
                "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None}

    def set_date(self, doc_id, date, origin="manuel"):
        d = self.docs.get(doc_id)
        if d is None:
            return False
        iso = parse_date_bound(date)
        with self._lock:
            d["date"], d["date_origine"] = iso, origin
            if self.data_dir:
                with open(self.data_dir / "documents.json", "w", encoding="utf-8") as f:
                    json.dump(list(self.docs.values()), f, ensure_ascii=False, indent=1)
        return True
