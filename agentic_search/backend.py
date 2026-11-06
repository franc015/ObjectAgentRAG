"""Interface commune des moteurs de recherche (mémoire, ClickHouse)."""

import calendar
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path


def parse_date_bound(value, end: bool = False) -> str | None:
    """Accepte 'AAAA', 'AAAA-MM' ou 'AAAA-MM-JJ' ; renvoie une date ISO complète.

    `end=True` donne le dernier jour de la période, sinon le premier.
    """
    if value in (None, ""):
        return None
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", s)
    if not m:
        raise ValueError(f"date invalide : {value!r} (formats : AAAA, AAAA-MM, AAAA-MM-JJ)")
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    if mo is None:
        mo_i, d_i = (12, 31) if end else (1, 1)
    else:
        mo_i = int(mo)
        if not 1 <= mo_i <= 12:
            raise ValueError(f"mois invalide : {value!r}")
        d_i = int(d) if d else (calendar.monthrange(y, mo_i)[1] if end else 1)
    return f"{y:04d}-{mo_i:02d}-{d_i:02d}"


class SearchBackend(ABC):
    """Contrat que les outils de l'agent attendent, quel que soit le stockage."""

    name = "abstract"

    @abstractmethod
    def search(self, query: str, limit: int = 5, doc_type: str | None = None,
               source_contains: str | None = None, date_min: str | None = None,
               date_max: str | None = None, max_per_doc: int = 3) -> list[dict]:
        """Retourne des résultats classés : id, doc_id, title, type, source, date,
        section, position, score, extrait."""

    @abstractmethod
    def get_passages(self, passage_id: str, before: int = 0, after: int = 0) -> list[dict]:
        """Le passage demandé et ses voisins (même document), avec le texte complet."""

    @abstractmethod
    def list_documents(self, contains: str | None = None, doc_type: str | None = None,
                       date_min: str | None = None, date_max: str | None = None,
                       order: str = "date_desc", limit: int = 30) -> list[dict]:
        """Documents filtrés et triés (date_desc, date_asc, titre)."""

    @abstractmethod
    def stats(self) -> dict:
        ...

    @abstractmethod
    def load(self, passages: list[dict], documents: list[dict]) -> None:
        """Remplace tout le contenu du moteur."""

    @abstractmethod
    def set_date(self, doc_id: str, date: str, origin: str = "manuel") -> bool:
        ...

    def close(self) -> None:
        pass


# --- format neutre produit par l'ingestion ---------------------------------

def write_neutral(data_dir: str | Path, documents: list[dict], passages: list[dict]) -> None:
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=1)
    with open(d / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def read_neutral(data_dir: str | Path) -> tuple[list[dict], list[dict]]:
    d = Path(data_dir)
    with open(d / "documents.json", encoding="utf-8") as f:
        documents = json.load(f)
    with open(d / "passages.jsonl", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f if line.strip()]
    return documents, passages
