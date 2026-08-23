#!/usr/bin/env python
"""Grade stored responses, then print per-cell refusal rates.

    uv run experiments/grade.py --dry-run
    uv run experiments/grade.py --cap 1
    uv run experiments/grade.py --report-only

Grading re-reads responses off disk, so revising the rubric costs judge calls only,
never new completions.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from safety_refusals.budget import PRICES
from safety_refusals.grading import JUDGE_MODEL, format_rates, grade_all, rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=JUDGE_MODEL, choices=sorted(PRICES), help="judge model")
    p.add_argument("--cap", type=float, default=1.0, help="hard spend cap for grading, USD")
    p.add_argument("--dry-run", action="store_true", help="count and price it, send nothing")
    p.add_argument("--report-only", action="store_true", help="print existing rates, grade nothing")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.report_only:
        client = None
        if not args.dry_run:
            from safety_refusals.api import get_openrouter_client

            client = get_openrouter_client()
        await grade_all(client, model=args.model, cap_usd=args.cap, dry_run=args.dry_run)

    print()
    print(format_rates(rates()))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
