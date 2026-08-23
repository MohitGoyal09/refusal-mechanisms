"""Append-only JSONL store for completed runs.

Upstream's SQLite cache in api.py only writes when *no* call in the batch raised. With
`return_exceptions=True` a single flaky call out of fifty discards the other forty-nine,
and the next run re-bills all of them. This store sits above that: one record per
(condition, model, reasoning, sample index), written as soon as it exists, so a
half-finished sweep resumes instead of restarting.

Records are never rewritten. Re-running a condition tops it up to the requested n.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "results" / "runs.jsonl"


@dataclass(frozen=True)
class RunKey:
    condition: str
    model: str
    reasoning: bool

    def as_dict(self) -> dict:
        return {"condition": self.condition, "model": self.model, "reasoning": self.reasoning}

    @staticmethod
    def of(record: dict) -> "RunKey":
        return RunKey(record["condition"], record["model"], bool(record["reasoning"]))


def load(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append(records: list[dict], path: Path = DEFAULT_PATH) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def count_for(key: RunKey, path: Path = DEFAULT_PATH) -> int:
    """How many samples already exist for this cell."""
    return sum(1 for r in load(path) if RunKey.of(r) == key)


def samples_for(key: RunKey, path: Path = DEFAULT_PATH) -> list[dict]:
    return [r for r in load(path) if RunKey.of(r) == key]


def missing(key: RunKey, n: int, path: Path = DEFAULT_PATH) -> int:
    """How many more samples this cell needs to reach n. Never negative."""
    return max(0, n - count_for(key, path))
