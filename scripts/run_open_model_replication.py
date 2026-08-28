"""Replicate the two-mechanism split on an open-weight model via OpenRouter.

Built for the MATS Winter 2027 (Neel Nanda stream) application: does the
uncertainty/objection split found on Claude 4.5 hold on a model whose weights
we could, in principle, look inside? This script only runs the behavioral
replication; it does not touch activations.

Usage:
    uv run python scripts/run_open_model_replication.py --model llama-3.3-70b --pilot
    uv run python scripts/run_open_model_replication.py --model llama-3.3-70b --full --cap 5.00
"""

from __future__ import annotations

import argparse
import asyncio

from safety_refusals import experiments, store
from safety_refusals.api import get_openrouter_client
from safety_refusals.runner import run_matrix


def full_conditions():
    """The same 17-condition set used for the Claude sweep, minus the reasoning-off/on
    duplication where a condition function already expands it."""
    conds = []
    for fn in (
        experiments.exp1_trust_consequence,
        experiments.exp2_valence,
        experiments.exp3_components,
        experiments.exp4_names,
        experiments.controls,
    ):
        conds.extend(fn())
    return conds


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama-3.3-70b", "qwen-2.5-72b"])
    ap.add_argument("--pilot", action="store_true", help="2-condition cheapest check")
    ap.add_argument("--full", action="store_true", help="full condition set")
    ap.add_argument("--n", type=int, default=10, help="samples per cell")
    ap.add_argument("--cap", type=float, default=1.00, help="USD spend cap for this run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conditions = experiments.pilot() if args.pilot else full_conditions()
    client = get_openrouter_client()
    ledger = await run_matrix(
        client, conditions, args.model, args.n, args.cap,
        dry_run=args.dry_run, backend="openrouter",
        path=store.DEFAULT_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
