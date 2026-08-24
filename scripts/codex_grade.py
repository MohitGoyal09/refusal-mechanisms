#!/usr/bin/env python
"""Grade the labelling sample with GPT 5.6 Sol via the Codex CLI, one response per call.

    uv run scripts/codex_grade.py            # grade, resumable
    uv run scripts/codex_grade.py --compare  # compare against the Haiku grades

A second grader from a different model family. Grading Claude's refusals with a Claude
model invites the objection that the grader shares the subject's blind spots; this
removes it.

One response per call, never batched. Batched three at a time it misgraded two of three
on the boundary that matters, and got the same case right alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from safety_refusals.grading import gradeable_text, load_grades
from safety_refusals.judge import RUBRIC, Grade, parse_verdict
from safety_refusals.labels import sample_for_labelling

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "grades_codex.jsonl"
MODEL = "gpt-5.6-sol"
#: Teaching examples shown in chat, excluded so the explanation cannot prime adjudication.
PRIMED = {("no_target_line__reas_off", 1), ("valence_eroding__reas_off", 0),
          ("no_target_line__reas_on", 13)}


def key(record: dict) -> tuple:
    return (record["condition"], record["model"], bool(record["reasoning"]), record["index"])


def already() -> set[tuple]:
    if not OUT.exists():
        return set()
    return {key(json.loads(l)) for l in OUT.read_text().splitlines() if l.strip()}


def grade_one(record: dict) -> dict | None:
    prompt = RUBRIC + "\n\n<response>\n" + gradeable_text(record) + "\n</response>\n"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        path = f.name
    try:
        with open(path) as stdin:
            proc = subprocess.run(
                ["codex", "exec", "--model", MODEL, "--sandbox", "read-only",
                 "-c", "mcp_servers={}", "-"],
                stdin=stdin, capture_output=True, text=True, timeout=420,
            )
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"  codex failed rc={proc.returncode}: {proc.stderr[-200:]}")
        return None
    try:
        return parse_verdict(proc.stdout).model_dump()
    except Exception as e:
        print(f"  unparseable: {e}")
        return None


def run(n: int, seed: int) -> int:
    batch = [r for r in sample_for_labelling(n, seed=seed)]
    done = already()
    todo = [r for r in batch if key(r) not in done]
    print(f"{len(batch)} in the sample, {len(done)} already graded, {len(todo)} to do")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(todo, start=1):
        verdict = grade_one(record)
        if verdict is None:
            continue
        with OUT.open("a") as f:
            f.write(json.dumps({
                "condition": record["condition"], "model": record["model"],
                "reasoning": record["reasoning"], "index": record["index"],
                "judge_model": MODEL, **verdict,
            }) + "\n")
        print(f"  [{i}/{len(todo)}] {record['condition']}#{record['index']} -> {verdict['grade']}")
    return 0


def compare() -> int:
    haiku = load_grades()
    if not OUT.exists():
        print("no codex grades yet")
        return 1
    codex = {key(json.loads(l)): json.loads(l) for l in OUT.read_text().splitlines() if l.strip()}
    shared = sorted(set(haiku) & set(codex))
    if not shared:
        print("no overlap yet")
        return 1

    agree = [k for k in shared if haiku[k]["grade"] == codex[k]["grade"]]
    boundary = [k for k in shared
                if Grade(haiku[k]["grade"]).is_refusal != Grade(codex[k]["grade"]).is_refusal]
    print(f"n={len(shared)}  exact agreement={len(agree)/len(shared):.0%}  "
          f"boundary agreement={(len(shared)-len(boundary))/len(shared):.0%}")

    disputed = [k for k in shared if haiku[k]["grade"] != codex[k]["grade"]]
    print(f"\n{len(disputed)} disagreements, {len(boundary)} of them crossing the refusal boundary:")
    for k in disputed:
        cell, model, reasoning, index = k
        mark = "  <-- CROSSES THE BOUNDARY, needs a human" if k in boundary else ""
        primed = "  (teaching example, excluded)" if (cell, index) in PRIMED else ""
        print(f"  {cell}#{index}: haiku {haiku[k]['grade']}, gpt {codex[k]['grade']}{mark}{primed}")

    needed = [k for k in boundary if (k[0], k[3]) not in PRIMED]
    print(f"\nYou need to adjudicate {len(needed)} of {len(shared)}.")
    (ROOT / "results" / "adjudicate.json").write_text(json.dumps(
        [{"condition": k[0], "model": k[1], "reasoning": k[2], "index": k[3]} for k in needed]
    ))
    print(f"written to results/adjudicate.json")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    return compare() if args.compare else run(args.n, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
