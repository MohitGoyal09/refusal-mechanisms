#!/usr/bin/env python
"""Generate the appendix results table from the grades, so it cannot drift from the data."""

from __future__ import annotations

from pathlib import Path

from safety_refusals.grading import rates
from safety_refusals.models import resolve

DESCRIPTIONS = {
    "full_spec": "complete ticket, benign target",
    "no_target_line": "complete, minus the training-target line",
    "trust_hi__conseq_lo": "approval kept, all consequence detail removed",
    "trust_lo__conseq_hi": "consequence detail kept, all approval removed",
    "trust_lo__conseq_lo": "both removed",
    "valence_eroding": "complete ticket, value-eroding target",
    "eroding_max_authority": "value-eroding, plus legal and executive sign-off",
    "ends_on_eval_all_present": "complete, reordered to end on the eval line",
    "drop_approval": "complete, minus the approval line",
    "drop_reviewers": "complete, minus the reviewer names",
    "drop_incidents": "complete, minus the incident detail",
    "drop_holdout_eval": "complete, minus the held-out eval",
    "drop_red_team": "complete, minus the red team plan",
}


def main() -> int:
    table = rates()
    rows = []
    for (cell, model, reasoning), s in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        base = cell.replace("__reas_off", "").replace("__reas_on", "")
        h = s["histogram"]
        rows.append(
            f"| {DESCRIPTIONS.get(base, base)} | {resolve(model).canonical} | "
            f"{'on' if reasoning else 'off'} | {s['n']} | **{s['refusal_rate']:.0%}** | "
            f"{h[0]}/{h[1]}/{h[2]}/{h[3]} | {s['asks_legitimacy']:.0%} | "
            f"{s['objects_to_outcome']:.0%} | {s['flags_as_test']:.0%} |"
        )
    header = (
        "| Condition | Model | Reasoning | n | Withheld | Grades 0/1/2/3 | Asks legitimacy | "
        "Objects to outcome | Flags as test |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    total = sum(s["n"] for s in table.values())
    out = header + "\n" + "\n".join(rows) + f"\n\nTotal graded responses: {total}.\n"
    Path("results/appendix.md").write_text(out)
    print(f"wrote results/appendix.md, {len(rows)} rows, {total} responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
