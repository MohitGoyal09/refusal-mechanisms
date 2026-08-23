#!/usr/bin/env python
"""One live call, to prove the wiring before a sweep spends anything.

    uv run experiments/smoke.py
    uv run experiments/smoke.py --reasoning --model opus-4.5

Costs roughly one call. Everything it checks is otherwise only checkable by spending.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from safety_refusals.backends import get_client, run_anthropic, run_openrouter
from safety_refusals.budget import cost_usd, money
from safety_refusals.conditions import Condition, Consequence, build_messages
from safety_refusals.models import choices
from safety_refusals.prompts import TOOLS

RUNNERS = {"anthropic": run_anthropic, "openrouter": run_openrouter}


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="haiku-4.5", choices=choices())
    p.add_argument("--backend", default="anthropic", choices=sorted(RUNNERS))
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--chars", type=int, default=900, help="how much of the response to print")
    args = p.parse_args()

    cond = Condition("smoke", consequence=Consequence.upstream_absent(), reasoning=args.reasoning)
    max_tokens = 8000 if args.reasoning else 4000

    out = await RUNNERS[args.backend](
        get_client(args.backend), args.model, build_messages(cond), 1,
        tools=TOOLS, max_tokens=max_tokens, reasoning=args.reasoning,
    )
    result = out[0]
    if isinstance(result, Exception):
        print(f"FAILED: {type(result).__name__}: {str(result)[:800]}")
        return 1

    spent = cost_usd(args.model, result.prompt_tokens, result.completion_tokens)
    print(
        f"OK  model={args.model} backend={args.backend} reasoning={args.reasoning}\n"
        f"    stop_reason={result.finish_reason}  truncated={result.truncated}\n"
        f"    tokens in/out={result.prompt_tokens}/{result.completion_tokens}  cost={money(spent)}\n"
        f"    tool_calls={len(result.tool_calls)}  text_chars={len(result.content or '')}"
        f"  thinking_chars={len(result.thinking or '')}"
    )
    print("\n--- response ---")
    print((result.content or "<no text block>")[: args.chars])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
