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

from safety_refusals.grading import JUDGE_MODEL, format_rates, grade_all, rates, truncated
from safety_refusals.models import choices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=JUDGE_MODEL, choices=choices(), help="judge model")
    p.add_argument("--cap", type=float, default=1.0, help="hard spend cap for grading, USD")
    p.add_argument("--dry-run", action="store_true", help="count and price it, send nothing")
    p.add_argument("--report-only", action="store_true", help="print existing rates, grade nothing")
    p.add_argument("--regrade", action="store_true",
                   help="re-score every sample, ignoring existing verdicts (use after a rubric change)")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.report_only:
        client = None
        if not args.dry_run:
            from safety_refusals.backends import get_anthropic_client

            client = get_anthropic_client()
        await grade_all(client, model=args.model, cap_usd=args.cap,
                        dry_run=args.dry_run, regrade=args.regrade)

    print()
    print(format_rates(rates()))
    cut = truncated()
    if cut:
        print(f"\n  WARNING: {len(cut)} stored samples were truncated and are excluded "
              f"from grading. Raise max_tokens and re-run those cells.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
