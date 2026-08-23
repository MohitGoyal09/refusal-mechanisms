"""Runs the graded rubric over stored samples and reports per-cell rates.

Grading is a separate pass from generation on purpose: the responses are already on
disk, so the rubric can be revised and re-run without re-paying for any completion.
Verdicts land in their own file keyed by (condition, model, reasoning, index), so a
re-grade tops up rather than duplicating.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from safety_refusals import store
from safety_refusals.budget import Ledger, cost_usd, money
from safety_refusals.judge import judge_messages, parse_verdict, refusal_rate

DEFAULT_GRADES_PATH = store.DEFAULT_PATH.parent / "grades.jsonl"
JUDGE_MODEL = "haiku-4.5"
#: A judge call is the response plus the rubric in, one short JSON object out.
JUDGE_INPUT_TOKENS = 2500
JUDGE_OUTPUT_TOKENS = 200


GRADER_VERSION = "v2-tool-calls"


def gradeable_text(record: dict) -> str | None:
    """What the judge sees: the reply, plus anything delivered through a tool call.

    In an agent harness the model often writes its answer to a file instead of typing it
    out, which leaves a two-line reply and a 15,000-character tool payload. Grading the
    reply alone reads full compliance as a stub, which is how the first pilot mislabelled
    six of twenty samples.
    """
    parts = []
    if record.get("content"):
        parts.append(record["content"].strip())
    for call in record.get("tool_calls") or []:
        payload = json.dumps(call.get("input", {}), indent=2)
        parts.append(f"TOOL CALLS\n{call.get('name', '?')}:\n{payload}")
    text = "\n\n".join(parts).strip()
    return text or None


def _sample_id(record: dict) -> tuple:
    return (record["condition"], record["model"], bool(record["reasoning"]), record["index"])


def load_grades(path: Path = DEFAULT_GRADES_PATH) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {_sample_id(r): r for r in (json.loads(line) for line in f if line.strip())}


def ungraded(
    runs_path: Path = store.DEFAULT_PATH,
    grades_path: Path = DEFAULT_GRADES_PATH,
    regrade: bool = False,
) -> list[dict]:
    """Samples worth grading and not yet graded.

    Skipped: samples with nothing to grade at all, and truncated replies, which were cut
    off at the output ceiling and would grade as false refusals. Tool-only replies ARE
    graded, on their tool payload. Pass regrade=True to re-score everything, which is
    what to do after the rubric or the gradeable text changes.
    """
    graded = {} if regrade else load_grades(grades_path)
    return [
        r for r in store.load(runs_path)
        if gradeable_text(r) and not r.get("truncated") and _sample_id(r) not in graded
    ]


def truncated(runs_path: Path = store.DEFAULT_PATH) -> list[dict]:
    """Samples that hit the output ceiling. If this is not near zero, raise max_tokens."""
    return [r for r in store.load(runs_path) if r.get("truncated")]


def estimate_grading_usd(n: int, model: str = JUDGE_MODEL) -> float:
    return cost_usd(model, n * JUDGE_INPUT_TOKENS, n * JUDGE_OUTPUT_TOKENS)


async def grade_all(
    client,
    *,
    model: str = JUDGE_MODEL,
    cap_usd: float = 1.0,
    dry_run: bool = False,
    runs_path: Path = store.DEFAULT_PATH,
    grades_path: Path = DEFAULT_GRADES_PATH,
    max_concurrent: int = 10,
    regrade: bool = False,
) -> int:
    """Grade every ungraded sample. Returns how many verdicts were written."""
    from safety_refusals.backends import run_anthropic_many

    todo = ungraded(runs_path, grades_path, regrade=regrade)
    if not todo:
        print("nothing to grade")
        return 0

    estimated = estimate_grading_usd(len(todo), model)
    ledger = Ledger(cap_usd=cap_usd)
    ledger.check("grading", estimated)
    print(f"grading {len(todo)} samples on {model}, estimated {money(estimated)}")
    if dry_run:
        return 0

    responses = await run_anthropic_many(
        client, model, [judge_messages(gradeable_text(r)) for r in todo],
        max_tokens=1000, temperature=0.0, reasoning=False,
        max_concurrent=max_concurrent,
    )

    written = 0
    unparsed = 0
    grades_path.parent.mkdir(parents=True, exist_ok=True)
    with grades_path.open("a") as f:
        for record, response in zip(todo, responses):
            if isinstance(response, Exception):
                continue
            try:
                verdict = parse_verdict(response.content or "")
            except Exception:
                unparsed += 1
                continue
            f.write(json.dumps({
                "condition": record["condition"],
                "model": record["model"],
                "reasoning": record["reasoning"],
                "index": record["index"],
                "judge_model": model,
                "grader": GRADER_VERSION,
                **verdict.model_dump(),
            }) + "\n")
            written += 1

    ledger.record_responses("grading", model, [r for r in responses if not isinstance(r, Exception)])
    print(f"wrote {written} verdicts, {unparsed} unparseable, {money(ledger.spent_usd)} actual")
    return written


def rates(grades_path: Path = DEFAULT_GRADES_PATH) -> dict[tuple[str, str, bool], dict]:
    """Per-cell refusal rate plus the grade histogram behind it."""
    buckets: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for verdict in load_grades(grades_path).values():
        buckets[(verdict["condition"], verdict["model"], bool(verdict["reasoning"]))].append(verdict)

    out = {}
    for key, verdicts in buckets.items():
        grades = [v["grade"] for v in verdicts]
        out[key] = {
            "n": len(grades),
            "refusal_rate": refusal_rate(grades),
            "histogram": {g: grades.count(g) for g in range(4)},
            "asks_legitimacy": sum(v["asks_legitimacy"] for v in verdicts) / len(verdicts),
            "objects_to_outcome": sum(v["objects_to_outcome"] for v in verdicts) / len(verdicts),
            "flags_as_test": sum(v["flags_as_test"] for v in verdicts) / len(verdicts),
        }
    return out


def format_rates(table: dict) -> str:
    """One row per cell. The model column is not optional: the same cell name on two
    models is two different results, and omitting it makes the table unreadable."""
    if not table:
        return "  no grades yet"
    from safety_refusals.models import resolve

    def short(model: str) -> str:
        try:
            return resolve(model).canonical
        except KeyError:
            return model

    cell_w = max(len(cell) for cell, _, _ in table)
    model_w = max(len(short(m)) for _, m, _ in table)
    rows = [
        f"  {'cell':<{cell_w}}  {'model':<{model_w}}  reas    n  refusal  0/1/2/3       "
        f" legit  objects  test"
    ]
    for (cell, model, reasoning), stats in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        h = stats["histogram"]
        rows.append(
            f"  {cell:<{cell_w}}  {short(model):<{model_w}}  {'on' if reasoning else 'off':<4} "
            f"{stats['n']:>4}  {stats['refusal_rate']:>6.0%}  "
            f"{h[0]:>2}/{h[1]:>2}/{h[2]:>2}/{h[3]:<2}  "
            f"{stats['asks_legitimacy']:>10.0%} {stats['objects_to_outcome']:>7.0%} "
            f"{stats['flags_as_test']:>5.0%}"
        )
    return "\n".join(rows)
