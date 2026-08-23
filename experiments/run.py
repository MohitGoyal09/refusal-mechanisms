#!/usr/bin/env python
"""CLI for the condition matrix.

    uv run experiments/run.py pilot --dry-run
    uv run experiments/run.py pilot --n 10 --cap 1
    uv run experiments/run.py exp1 --n 30 --model anthropic/claude-haiku-4.5 --cap 5

--dry-run needs no API key. It prints the matrix, what is already stored, and the
estimated spend, and exits without sending anything.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from safety_refusals.budget import PRICES, money
from safety_refusals.experiments import REGISTRY
from safety_refusals.runner import format_plan, plan_matrix, run_matrix

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=sorted(REGISTRY), help="which condition set to run")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES), help="target model")
    p.add_argument("--n", type=int, default=30, help="samples per cell (tops up, never re-bills)")
    p.add_argument("--cap", type=float, default=5.0, help="hard spend cap for this run, USD")
    p.add_argument("--dry-run", action="store_true", help="plan and price it, send nothing")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conditions = REGISTRY[args.experiment]()

    if args.dry_run:
        plans = plan_matrix(conditions, args.model, args.n)
        print(f"{args.experiment} on {args.model}, n={args.n} per cell\n")
        print(format_plan(plans))
        total = sum(p.estimated_usd for p in plans)
        print(f"\ncap {money(args.cap)}; planned {money(total)}; "
              f"{'WITHIN cap' if total <= args.cap else 'OVER cap, run would refuse'}")
        return 0

    from safety_refusals.api import get_openrouter_client

    client = get_openrouter_client()
    await run_matrix(client, conditions, args.model, args.n, args.cap)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
