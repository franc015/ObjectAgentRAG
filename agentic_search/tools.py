"""Outils exposés à l'agent : search, get_passage, list_documents."""

import copy
import json
from dataclasses import dataclass, field

from .backend import SearchBackend

SCHEMAS = [
    {
        "name": "search",
        "description": (
            "Recherche LEXICALE (mots-clés, classement BM25) dans le corpus : articles de presse, pages web, "
            "documents et livres, découpés en passages. Il n'y a PAS de recherche sémantique : utilise les "
            "mots probablement présents dans le texte (essaie synonymes et variantes). Les accents et les "
            "pluriels sont ignorés. Mets une expression exacte entre guillemets pour l'exiger. "
            "Retourne des extraits avec l'identifiant de passage à citer."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "mots-clés, ex. : assurance habitation sinistre \"franchise\""},
                "limit": {"type": "integer", "description": "nombre de résultats (défaut 5, max 10)"},
                "doc_type": {"type": "string", "enum": ["pdf", "docx", "epub", "web", "text"],
                             "description": "restreindre à un type de source"},
                "source_contains": {"type": "string", "description": "texte contenu dans le titre ou la source"},
                "date_min": {"type": "string", "description": "AAAA, AAAA-MM ou AAAA-MM-JJ (inclus)"},
                "date_max": {"type": "string", "description": "AAAA, AAAA-MM ou AAAA-MM-JJ (inclus)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_passage",
        "description": "Lit le texte complet d'un passage, avec éventuellement les passages voisins du même document.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "identifiant de passage, ex. 3fa9c1d2-0004"},
                "before": {"type": "integer", "description": "passages précédents à inclure (0 à 3)"},
                "after": {"type": "integer", "description": "passages suivants à inclure (0 à 3)"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "list_documents",
        "description": ("Liste les documents du corpus (titre, type, date), triés par date. "
                        "Utile pour les questions chronologiques ou pour voir ce que contient le corpus."),
        "parameters": {
            "type": "object",
            "properties": {
                "contains": {"type": "string", "description": "texte contenu dans le titre ou la source"},
                "doc_type": {"type": "string", "enum": ["pdf", "docx", "epub", "web", "text"]},
                "date_min": {"type": "string"}, "date_max": {"type": "string"},
                "order": {"type": "string", "enum": ["date_desc", "date_asc", "titre"]},
                "limit": {"type": "integer", "description": "défaut 30, max 100"},
            },
        },
    },
]


THEMES_PROP = {"type": "array", "items": {"type": "string"},
               "description": "restreindre aux documents portant AU MOINS UN de ces thèmes"}
GET_DOCUMENT = {
    "name": "get_document",
    "description": ("Lit le texte (markdown) d'un document entier, par tranches de caractères. "
                    "Pour un passage précis, préfère get_passage ; ici, on lit un article du début à la fin."),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "identifiant de document (8 caractères), ex. 3fa9c1d2"},
            "offset": {"type": "integer", "description": "premier caractère à lire (défaut 0)"},
            "length": {"type": "integer", "description": "nombre de caractères (défaut 6000, max 8000)"},
        },
        "required": ["doc_id"],
    },
}


def build_schemas(capabilities, vocabulary=None) -> list:
    """Schémas d'outils adaptés aux fonctions du backend (thèmes, lecture de document)."""
    schemas = copy.deepcopy(SCHEMAS)
    if "themes" in capabilities:
        prop = copy.deepcopy(THEMES_PROP)
        if vocabulary:
            prop["items"] = {"type": "string", "enum": list(vocabulary)}
        for sch in schemas:
            if sch["name"] in ("search", "list_documents"):
                sch["parameters"]["properties"]["themes"] = copy.deepcopy(prop)
    if "markdown" in capabilities:
        schemas.append(copy.deepcopy(GET_DOCUMENT))
    return schemas


@dataclass
class ToolOutput:
    text: str
    passage_ids: list = field(default_factory=list)
    n_results: int = 0


def _int(v, default, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


class Toolbox:
    def __init__(self, backend: SearchBackend, max_results=5, max_per_doc=3, max_chars=12000, vocabulary=None):
        self.backend = backend
        self.max_results, self.max_per_doc, self.max_chars = max_results, max_per_doc, max_chars
        self.caps = frozenset(getattr(backend, "capabilities", ()))
        self.schemas = build_schemas(self.caps, vocabulary)

    def _themes(self, args) -> dict:
        """Argument `themes` de l'agent, transmis seulement si le backend sait filtrer par thème."""
        if "themes" not in self.caps:
            return {}
        t = args.get("themes")
        if isinstance(t, str):
            t = [t]
        t = [str(x) for x in t if str(x).strip()] if isinstance(t, list) else []
        return {"themes": t} if t else {}

    def _dump(self, obj) -> str:
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) > self.max_chars:
            s = s[: self.max_chars] + '… [résultat tronqué : affinez la recherche ou lisez un passage précis]'
        return s

    def call(self, name: str, args: dict) -> ToolOutput:
        try:
            if "__parse_error" in args:
                return ToolOutput(self._dump({"erreur": "arguments JSON invalides, réessaie",
                                              "reçu": str(args["__parse_error"])[:200]}))
            if name == "search":
                hits = self.backend.search(
                    str(args.get("query", "")), limit=_int(args.get("limit"), self.max_results, 1, 10),
                    doc_type=args.get("doc_type") or None, source_contains=args.get("source_contains") or None,
                    date_min=args.get("date_min") or None, date_max=args.get("date_max") or None,
                    max_per_doc=self.max_per_doc, **self._themes(args))
                if not hits:
                    return ToolOutput(self._dump({"résultats": [], "conseil":
                                                  "aucun résultat : essaie d'autres mots, des synonymes, "
                                                  "moins de termes ou retire les filtres"}))
                res = [{"id": h["id"], "titre": h["title"], "type": h["type"], "date": h["date"],
                        "section": h["section"], "source": h["source"], "score": h["score"],
                        "extrait": h["extrait"], **({"themes": h["themes"]} if h.get("themes") else {})}
                       for h in hits]
                return ToolOutput(self._dump({"résultats": res}), [h["id"] for h in hits], len(hits))
            if name == "get_passage":
                ps = self.backend.get_passages(str(args.get("id", "")),
                                               _int(args.get("before"), 0, 0, 3), _int(args.get("after"), 0, 0, 3))
                if not ps:
                    return ToolOutput(self._dump({"erreur": "identifiant de passage inconnu"}))
                res = [{"id": p["id"], "titre": p["title"], "date": p["date"], "section": p["section"],
                        "source": p["source"], "texte": p["text"]} for p in ps]
                return ToolOutput(self._dump({"passages": res}), [p["id"] for p in ps], len(ps))
            if name == "list_documents":
                ds = self.backend.list_documents(
                    contains=args.get("contains") or None, doc_type=args.get("doc_type") or None,
                    date_min=args.get("date_min") or None, date_max=args.get("date_max") or None,
                    order=args.get("order") or "date_desc", limit=_int(args.get("limit"), 30, 1, 100),
                    **self._themes(args))
                return ToolOutput(self._dump({"documents": ds}), [], len(ds))
            if name == "get_document" and "markdown" in self.caps:
                size = min(8000, max(500, self.max_chars - 1500))
                doc = self.backend.get_document(str(args.get("doc_id", "")), _int(args.get("offset"), 0, 0, 10**9),
                                                _int(args.get("length"), 6000, 500, size))
                if doc is None:
                    return ToolOutput(self._dump({"erreur": "document inconnu ou sans contenu enregistré"}))
                return ToolOutput(self._dump(doc), [], 1)
            return ToolOutput(self._dump({"erreur": f"outil inconnu : {name}"}))
        except ValueError as e:                    # dates invalides, etc. : l'agent peut corriger
            return ToolOutput(self._dump({"erreur": str(e)}))
        except Exception as e:
            return ToolOutput(self._dump({"erreur": f"{type(e).__name__}: {e}"}))
