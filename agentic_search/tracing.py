"""Traçage des exécutions de l'agent : aucun, fichiers JSONL, ou ClickHouse."""

import json
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class EventTrace:
    step: int
    kind: str                      # "llm" ou "tool"
    tool: str = ""
    arguments: str = ""
    passage_ids: list = field(default_factory=list)
    n_results: int = 0
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RunTrace:
    question: str
    provider: str
    model: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    status: str = "en_cours"
    answer: str = ""
    cited: list = field(default_factory=list)
    events: list = field(default_factory=list)


class Tracer(ABC):
    """Une exécution est enregistrée en un seul lot, à la fin."""

    @abstractmethod
    def record(self, run: RunTrace) -> None: ...

    def recent(self, n: int = 10) -> list[dict]:
        return []


class NullTracer(Tracer):
    def record(self, run: RunTrace) -> None:
        pass


class JsonlTracer(Tracer):
    def __init__(self, data_dir: str | Path = "data"):
        self.dir = Path(data_dir) / "traces"
        self._lock = threading.Lock()

    def record(self, run: RunTrace) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"runs-{run.ts:%Y%m}.jsonl"
        line = json.dumps(asdict(run), ensure_ascii=False, default=str)
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def recent(self, n: int = 10) -> list[dict]:
        out: list[dict] = []
        for p in sorted(self.dir.glob("runs-*.jsonl"), reverse=True):
            with open(p, encoding="utf-8") as f:
                out = [json.loads(x) for x in f if x.strip()] + out
            if len(out) >= n:
                break
        return out[-n:][::-1]
