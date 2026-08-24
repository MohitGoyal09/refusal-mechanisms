"""Human labels, and the judge's agreement with them.

No refusal rate in the write-up is trustworthy until the judge has been scored against
labels a person actually assigned. Upstream says as much: his own automated recount
disagreed with his hand labels by 26 points on the reasoning-on cells.

Labels live in their own file so re-grading never touches them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from safety_refusals import store
from safety_refusals.grading import DEFAULT_GRADES_PATH, gradeable_text, load_grades
from safety_refusals.judge import Agreement, agreement

DEFAULT_LABELS_PATH = store.DEFAULT_PATH.parent / "labels.jsonl"


def _key(record: dict) -> tuple:
    return (record["condition"], record["model"], bool(record["reasoning"]), record["index"])


def load_labels(path: Path = DEFAULT_LABELS_PATH) -> dict[tuple, int]:
    """Later lines win, so a corrected label supersedes the first attempt."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[_key(record)] = record["grade"]
    return out


def save_label(record: dict, grade: int, path: Path = DEFAULT_LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({
            "condition": record["condition"],
            "model": record["model"],
            "reasoning": record["reasoning"],
            "index": record["index"],
            "grade": grade,
        }) + "\n")


def spot_check_set(
    n: int = 10,
    *,
    seed: int = 0,
    runs_path: Path = store.DEFAULT_PATH,
    grades_path: Path = DEFAULT_GRADES_PATH,
    codex_path: Path | None = None,
    exclude: set[tuple] | None = None,
) -> list[dict]:
    """The samples worth a human eye, rather than a random draw.

    Two graders agreeing on the refusal boundary removes most of the risk but not a
    shared blind spot. The cases most likely to hide one are the ones the graders
    disputed, and the ones sitting on the boundary itself (grades 1 and 2). A random
    sample spends most of its effort on obvious cases.
    """
    codex_path = codex_path or (store.DEFAULT_PATH.parent / "grades_codex.jsonl")
    exclude = exclude or set()
    grades = load_grades(grades_path)
    codex = {}
    if codex_path.exists():
        codex = {
            _key(r): r for r in
            (json.loads(l) for l in codex_path.read_text().splitlines() if l.strip())
        }

    by_key = {_key(r): r for r in store.load(runs_path) if gradeable_text(r)}
    disputed, boundary = [], []
    for k, record in by_key.items():
        if (k[0], k[3]) in exclude or k not in grades:
            continue
        mine = grades[k]["grade"]
        theirs = codex.get(k, {}).get("grade")
        if theirs is not None and theirs != mine:
            disputed.append(record)
        elif mine in (1, 2):
            boundary.append(record)

    rng = random.Random(seed)
    rng.shuffle(boundary)
    return (disputed + boundary)[:n]


def sample_for_labelling(
    n: int,
    *,
    seed: int,
    runs_path: Path = store.DEFAULT_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
) -> list[dict]:
    """A stratified sample across cells, so no single cell dominates the agreement score.

    Deliberately blind to the judge's verdict: seeing the machine grade first would
    anchor the human label and inflate agreement.
    """
    labelled = load_labels(labels_path)
    pool = [
        r for r in store.load(runs_path)
        if gradeable_text(r) and not r.get("truncated") and _key(r) not in labelled
    ]
    by_cell: dict[tuple, list[dict]] = {}
    for record in pool:
        by_cell.setdefault(_key(record)[:3], []).append(record)

    rng = random.Random(seed)
    for records in by_cell.values():
        rng.shuffle(records)

    picked: list[dict] = []
    cells = sorted(by_cell)
    while len(picked) < n and any(by_cell[c] for c in cells):
        for cell in cells:
            if by_cell[cell] and len(picked) < n:
                picked.append(by_cell[cell].pop())
    return picked


def score(
    labels_path: Path = DEFAULT_LABELS_PATH,
    grades_path: Path = DEFAULT_GRADES_PATH,
) -> tuple[Agreement, list[tuple]]:
    """Agreement over samples that have both a human label and a judge verdict."""
    labels = load_labels(labels_path)
    grades = load_grades(grades_path)
    shared = sorted(set(labels) & set(grades))
    if not shared:
        raise ValueError(
            "No sample has both a human label and a judge verdict yet. Label some first."
        )
    human = [labels[k] for k in shared]
    machine = [grades[k]["grade"] for k in shared]
    disagreements = [
        (k, labels[k], grades[k]["grade"]) for k in shared if labels[k] != grades[k]["grade"]
    ]
    return agreement(human, machine), disagreements
