"""Normalisation et tokenisation du français, utilisées à l'indexation ET à la requête.

Les deux backends (mémoire et ClickHouse) partagent ce module : c'est ce qui garantit
des résultats comparables.
"""

import re
import unicodedata

_LIGATURES = {"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae"}


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces compactés."""
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


_STOP_RAW = """
le la les l un une des du de d et ou en au aux a à ce ces cet cette qui que quoi dont où
ne pas plus par pour sur sous dans avec sans est sont été être était ont avait il elle ils
elles on nous vous je tu se sa son ses leur leurs mais donc or ni car y m t s c qu n j
lui eux ci là si comme très aussi
"""
STOPWORDS = {normalize(w) for w in _STOP_RAW.split()}


def stem_token(t: str) -> str:
    """Radicalisation très légère : pluriels (s, x, aux -> al)."""
    if len(t) > 4 and t.endswith("aux"):
        return t[:-3] + "al"
    if len(t) > 3 and t.endswith(("s", "x")):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", normalize(text))
    return [stem_token(t) for t in toks if t not in STOPWORDS and len(t) > 1]


def raw_tokens(text: str) -> list[str]:
    """Jetons normalisés sans radicalisation ni mots vides (pour l'index de texte ClickHouse)."""
    return re.findall(r"[a-z0-9]+", normalize(text))


def index_text(title: str, section: str, text: str) -> str:
    """Texte réellement indexé pour un passage (titre et section inclus)."""
    return f"{title} {section or ''} {text}"


def query_terms(query: str) -> tuple[list[str], list[str]]:
    """Retourne (termes, expressions exactes entre guillemets, normalisées)."""
    phrases = [normalize(m) for m in re.findall(r'"([^"]+)"', query)]
    rest = re.sub(r'"[^"]*"', " ", query)
    terms = tokenize(rest) + [t for ph in phrases for t in tokenize(ph)]
    return terms, phrases


def make_snippet(text: str, terms: list[str], words: int = 60) -> str:
    """Fenêtre de texte autour du premier terme trouvé."""
    toks = text.split()
    wanted = set(terms)
    centre = 0
    for k, w in enumerate(toks):
        parts = tokenize(w)
        if parts and parts[0] in wanted:
            centre = k
            break
    start = max(0, centre - words // 3)
    end = min(len(toks), start + words)
    return ("… " if start > 0 else "") + " ".join(toks[start:end]) + (" …" if end < len(toks) else "")
