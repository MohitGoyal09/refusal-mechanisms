"""Runs a condition matrix: estimate, guard, call, persist, report.

Three invariants:
  1. Nothing is sent until the ledger has cleared the estimate.
  2. `max_tokens` is capped, because the upstream default of 16000 is a six-fold cost tail.
  3. Every completed sample is written to the store immediately, so an interrupted sweep
     resumes instead of re-billing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from safety_refusals import store
from safety_refusals.backends import BACKENDS, Completion
from safety_refusals.budget import (
    MAX_TOKENS_CAP,
    BudgetExceeded,
    Ledger,
    default_max_tokens,
    estimate_usd,
    money,
)
from safety_refusals.conditions import Condition, build_messages
from safety_refusals.models import resolve
from safety_refusals.prompts import TOOLS

DEFAULT_BACKEND = "anthropic"


@dataclass(frozen=True)
class CellPlan:
    condition: str
    model: str
    reasoning: bool
    have: int
    need: int
    estimated_usd: float


def plan_cell(cond: Condition, model: str, n: int, path=store.DEFAULT_PATH) -> CellPlan:
    key = store.RunKey(cond.name, model, cond.reasoning)
    have = store.count_for(key, path)
    need = max(0, n - have)
    return CellPlan(
        condition=cond.name,
        model=model,
        reasoning=cond.reasoning,
        have=have,
        need=need,
        estimated_usd=estimate_usd(model, need, cond.reasoning),
    )


def plan_matrix(conditions: list[Condition], model: str, n: int, path=store.DEFAULT_PATH) -> list[CellPlan]:
    return [plan_cell(c, model, n, path) for c in conditions]


def format_plan(plans: list[CellPlan]) -> str:
    width = max((len(p.condition) for p in plans), default=10)
    rows = [f"  {'cell':<{width}}  {'reas':<5} {'have':>4} {'need':>5}  estimate"]
    for p in plans:
        rows.append(
            f"  {p.condition:<{width}}  {'on' if p.reasoning else 'off':<5} "
            f"{p.have:>4} {p.need:>5}  {money(p.estimated_usd)}"
        )
    total = sum(p.estimated_usd for p in plans)
    rows.append(f"  {'TOTAL':<{width}}  {'':<5} {'':>4} "
                f"{sum(p.need for p in plans):>5}  {money(total)}")
    return "\n".join(rows)


def _record(cond: Condition, model: str, index: int, c: Completion, backend: str) -> dict:
    return {
        "condition": cond.name,
        "model": model,
        "backend": backend,
        "reasoning": cond.reasoning,
        "index": index,
        "content": c.content,
        "thinking": c.thinking,
        "tool_calls": c.tool_calls,
        "finish_reason": c.finish_reason,
        "truncated": c.truncated,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valence": str(cond.valence),
        "names": str(cond.names),
        "system_prompt": str(cond.system_prompt),
        "trust": asdict(cond.trust),
        "consequence": asdict(cond.consequence),
    }


async def run_cell(
    client,
    cond: Condition,
    model: str,
    n: int,
    ledger: Ledger,
    *,
    dry_run: bool = False,
    backend: str = DEFAULT_BACKEND,
    max_tokens: int | None = None,
    path=store.DEFAULT_PATH,
    max_concurrent: int = 8,
) -> list[dict]:
    """Top a cell up to n samples. Returns every sample for the cell, old and new."""
    if max_tokens is None:
        max_tokens = default_max_tokens(cond.reasoning)
    if max_tokens > MAX_TOKENS_CAP:
        raise ValueError(
            f"max_tokens={max_tokens} exceeds the cap of {MAX_TOKENS_CAP}. Raise "
            f"MAX_TOKENS_CAP deliberately if you mean it; this guard exists because the "
            f"upstream default of 16000 multiplies the bill several-fold."
        )
    if backend not in BACKENDS:
        raise KeyError(f"Unknown backend {backend!r}. Known: {sorted(BACKENDS)}")

    key = store.RunKey(cond.name, model, cond.reasoning)
    need = store.missing(key, n, path)
    if need == 0:
        return store.samples_for(key, path)

    estimated = estimate_usd(model, need, cond.reasoning)
    ledger.check(cond.name, estimated)
    if dry_run:
        print(f"[dry-run] {cond.name}: would run {need} calls, {money(estimated)}")
        return store.samples_for(key, path)

    responses = await BACKENDS[backend](
        client,
        model,
        build_messages(cond),
        need,
        tools=TOOLS,
        max_tokens=max_tokens,
        temperature=1.0,
        reasoning=cond.reasoning,
        max_concurrent=max_concurrent,
    )

    ok = [r for r in responses if not isinstance(r, Exception)]
    failed = len(responses) - len(ok)
    if failed:
        print(f"  {cond.name}: {failed}/{len(responses)} calls failed, keeping the rest")

    offset = store.count_for(key, path)
    store.append([_record(cond, model, offset + i, r, backend) for i, r in enumerate(ok)], path)
    spent = ledger.record_responses(cond.name, model, ok)
    cut = sum(r.truncated for r in ok)
    warning = f", {cut} TRUNCATED" if cut else ""
    print(f"  {cond.name}: +{len(ok)} samples, {money(spent)} actual{warning}")
    return store.samples_for(key, path)


async def run_matrix(
    client,
    conditions: list[Condition],
    model: str,
    n: int,
    cap_usd: float,
    *,
    dry_run: bool = False,
    backend: str = DEFAULT_BACKEND,
    path=store.DEFAULT_PATH,
) -> Ledger:
    plans = plan_matrix(conditions, model, n, path)
    print(format_plan(plans))
    total = sum(p.estimated_usd for p in plans)
    if total > cap_usd:
        raise BudgetExceeded(
            f"Planned spend {money(total)} exceeds the cap {money(cap_usd)}. "
            f"Lower n or raise the cap on purpose."
        )

    ledger = Ledger(cap_usd=cap_usd)
    for cond in conditions:
        await run_cell(
            client, cond, model, n, ledger,
            dry_run=dry_run, backend=backend, path=path,
        )
    if not dry_run:
        print("\nspend:")
        print(ledger.report())
    return ledger
