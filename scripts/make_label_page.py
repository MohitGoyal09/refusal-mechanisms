#!/usr/bin/env python
"""Build a self-contained blind labelling page from the stored responses.

    uv run scripts/make_label_page.py            # 40 samples, seed 0
    uv run scripts/make_label_page.py --n 60

Writes label.html next to the repo root. Open it, label, download labels.jsonl,
then install it with the command the page prints.

Deliberately local, not a hosted page: the responses embed a fabricated ticket that
names real Anthropic researchers, and that content should not be published anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_refusals.conditions import build_user_prompt, condition_from_record
from safety_refusals.grading import gradeable_text
from safety_refusals.labels import sample_for_labelling, spot_check_set

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "label_page.html"
OUT = ROOT / "label.html"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spot-check", action="store_true",
                   help="only the disputed and boundary cases, not a random draw")
    args = p.parse_args()

    #: Teaching examples already shown with their grades, so they cannot be labelled blind.
    PRIMED = {("no_target_line__reas_off", 1), ("valence_eroding__reas_off", 0),
              ("no_target_line__reas_on", 13)}
    batch = (spot_check_set(args.n, seed=args.seed, exclude=PRIMED) if args.spot_check
             else sample_for_labelling(args.n, seed=args.seed))
    if not batch:
        print("nothing left to label")
        return 1

    # The page is blind on purpose: no condition, no model, no grader verdict. Only the
    # identifying fields needed to write the label back, and the text to read.
    items = [
        {
            "condition": r["condition"],
            "model": r["model"],
            "reasoning": bool(r["reasoning"]),
            "index": r["index"],
            # Rebuilt from the stored axis settings by the same code that sent it.
            "request": build_user_prompt(condition_from_record(r)),
            "response": gradeable_text(r),
            "chars": len(gradeable_text(r)),
            "tool_call": bool(r.get("tool_calls")),
        }
        for r in batch
    ]

    html = TEMPLATE.read_text().replace(
        "/*__DATA__*/null",
        json.dumps(items, ensure_ascii=False).replace("</", "<\\/"),
    ).replace("__SEED__", str(args.seed))
    OUT.write_text(html)
    print(f"wrote {OUT} with {len(items)} samples (seed {args.seed})")
    print(f"open it with:  open {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
