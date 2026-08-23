#!/usr/bin/env python
"""One-off: refile already-paid samples under the canonical cell names.

Four experiment sets needed the same fully specified ticket, and the pilot used bare
names where replication used suffixed ones. Those were separate store keys for one
experiment, so the planner could not see samples it already owned and would have bought
them again. This refiles the existing runs and grades under the canonical names and
renumbers indices within each cell.

Run once. It backs up both files first and is a no-op on a second run.

    uv run scripts/migrate_cell_names.py --dry-run
    uv run scripts/migrate_cell_names.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS, GRADES = RESULTS / "runs.jsonl", RESULTS / "grades.jsonl"

#: old cell name -> canonical cell name
RENAMES = {
    "pilot_absent": "no_target_line__reas_off",
    "pilot_present": "full_spec__reas_off",
    "trust_hi__conseq_hi__reas_off": "full_spec__reas_off",
}


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    runs, grades = read(RUNS), read(GRADES)
    if not runs:
        print("nothing to migrate")
        return 0

    # Refile and renumber runs in file order, recording where each sample moved.
    moves: dict[tuple, tuple] = {}
    counters: dict[tuple, int] = {}
    migrated_runs = []
    for record in runs:
        old_key = (record["condition"], record["model"], bool(record["reasoning"]), record["index"])
        new_name = RENAMES.get(record["condition"], record["condition"])
        cell = (new_name, record["model"], bool(record["reasoning"]))
        new_index = counters.get(cell, 0)
        counters[cell] = new_index + 1
        moves[old_key] = (new_name, new_index)
        migrated_runs.append({**record, "condition": new_name, "index": new_index})

    # Apply the same moves to grades so verdicts stay attached to their samples.
    migrated_grades, orphans = [], 0
    for verdict in grades:
        old_key = (verdict["condition"], verdict["model"], bool(verdict["reasoning"]), verdict["index"])
        if old_key not in moves:
            orphans += 1
            continue
        new_name, new_index = moves[old_key]
        migrated_grades.append({**verdict, "condition": new_name, "index": new_index})

    renamed = sum(1 for r in runs if r["condition"] in RENAMES)
    print(f"runs: {len(runs)}, refiled {renamed}")
    print(f"grades: {len(grades)} -> {len(migrated_grades)}" + (f", {orphans} orphaned" if orphans else ""))
    for cell, count in sorted(counters.items()):
        print(f"  {cell[0]:<28} {cell[1]:<18} reas={cell[2]!s:<5} n={count}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    for path in (RUNS, GRADES):
        if path.exists():
            shutil.copy2(path, path.with_suffix(".jsonl.bak"))
    write(RUNS, migrated_runs)
    write(GRADES, migrated_grades)
    print("\nwritten. backups at *.jsonl.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
