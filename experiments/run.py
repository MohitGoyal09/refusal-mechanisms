#!/usr/bin/env python
"""CLI for the condition matrix.

    uv run experiments/run.py pilot --dry-run
    uv run experiments/run.py pilot --n 10 --cap 1
    uv run experiments/run.py exp1 --n 30 --model opus-4.5 --cap 5
    uv run experiments/run.py exp1 --n 30 --backend openrouter

--dry-run needs no API key. It prints the matrix, what is already stored, and the
estimated spend, and exits without sending anything.

The default backend is the native Anthropic API, which gives exact control of extended
thinking. Use --backend openrouter to reproduce upstream's path.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from safety_refusals.backends import BACKENDS
from safety_refusals.budget import money
from safety_refusals.experiments import REGISTRY
from safety_refusals.models import choices
from safety_refusals.runner import DEFAULT_BACKEND, format_plan, plan_matrix, run_matrix

DEFAULT_MODEL = "haiku-4.5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=sorted(REGISTRY), help="which condition set to run")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=choices(), help="target model")
    p.add_argument("--backend", default=DEFAULT_BACKEND, choices=sorted(BACKENDS),
                   help="anthropic (native, exact thinking control) or openrouter (upstream path)")
    p.add_argument("--n", type=int, default=30, help="samples per cell (tops up, never re-bills)")
    p.add_argument("--cap", type=float, default=5.0, help="hard spend cap for this run, USD")
    p.add_argument("--dry-run", action="store_true", help="plan and price it, send nothing")
    p.add_argument("--cell", default=None,
                   help="run only cells whose name contains this substring")
    p.add_argument("--reasoning", default="both", choices=["off", "on", "both"],
                   help="restrict to reasoning-off or reasoning-on cells. The no-reasoning "
                        "contrast is the clean one upstream, and half the price")
    return p.parse_args(argv)


def select(conditions: list, reasoning: str) -> list:
    if reasoning == "both":
        return conditions
    want = reasoning == "on"
    return [c for c in conditions if c.reasoning is want]


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conditions = select(REGISTRY[args.experiment](), args.reasoning)
    if args.cell:
        conditions = [c for c in conditions if args.cell in c.name]
    if not conditions:
        print(f"{args.experiment} has no reasoning-{args.reasoning} cells")
        return 1

    if args.dry_run:
        plans = plan_matrix(conditions, args.model, args.n)
        print(f"{args.experiment} on {args.model} via {args.backend}, n={args.n} per cell\n")
        print(format_plan(plans))
        total = sum(p.estimated_usd for p in plans)
        print(f"\ncap {money(args.cap)}; planned {money(total)}; "
              f"{'WITHIN cap' if total <= args.cap else 'OVER cap, run would refuse'}")
        return 0

    from safety_refusals.backends import get_client

    client = get_client(args.backend)
    await run_matrix(client, conditions, args.model, args.n, args.cap, backend=args.backend)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
