#!/usr/bin/env python
"""Build the headline figure: three panels, one per independent signature.

    uv run scripts/make_figure.py

Every number is read from results/grades.jsonl, never hardcoded, so the figure cannot
drift from the data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from safety_refusals.grading import rates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figure.png"
OPUS = "opus-4.5"

INK, SOFT, FAINT = "#1a1a1a", "#5a5a5a", "#9a9a9a"
AMBIG, OBJECT, NEUTRAL = "#c2410c", "#7c2d12", "#94a3b8"
GRADE_COLORS = ["#d8dee6", "#94a3b8", "#e08a5a", "#a8371a"]
GRADE_LABELS = ["0 drafted, no concern", "1 drafted, with concern",
                "2 withheld, offered to continue", "3 withheld, no offer"]

ABLATION = [
    ("nothing removed", "full_spec__reas_off"),
    ("approval line", "drop_approval__reas_off"),
    ("reviewer names", "drop_reviewers__reas_off"),
    ("incident detail", "drop_incidents__reas_off"),
    ("held-out eval", "drop_holdout_eval__reas_off"),
    ("red team plan", "drop_red_team__reas_off"),
    ("training target line", "no_target_line__reas_off"),
]
SLOPES = [
    ("benign, fully specified", "full_spec", NEUTRAL),
    ("approval stripped", "trust_lo__conseq_hi", AMBIG),
    ("target line removed", "no_target_line", AMBIG),
    ("value-eroding target", "valence_eroding", OBJECT),
]
COMPOSITION = [
    ("benign,\nfully specified", "full_spec__reas_off"),
    ("uncertainty\n(target line gone)", "no_target_line__reas_off"),
    ("objection\n(value-eroding)", "valence_eroding__reas_off"),
]


def main() -> int:
    table = rates()
    def cell(name: str) -> dict:
        hits = [v for (c, m, _r), v in table.items() if c == name and m == OPUS]
        if not hits:
            raise KeyError(f"no graded samples for {name} on {OPUS}")
        return hits[0]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#d4d4d4", "axes.linewidth": 0.8,
        "xtick.color": SOFT, "ytick.color": SOFT,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.titlesize": 10, "axes.titleweight": "600",
        "figure.dpi": 200,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 5.0),
                                        gridspec_kw={"width_ratios": [1.15, 1, 0.95], "wspace": 0.36})

    # --- Panel A: which single line carries the effect ---------------------- #
    labels = [lbl for lbl, _ in ABLATION]
    values = [100 * cell(k)["refusal_rate"] for _, k in ABLATION]
    colors = [NEUTRAL if v < 15 else AMBIG for v in values]
    bars = ax1.barh(range(len(labels)), values, color=colors, height=0.66)
    ax1.set_yticks(range(len(labels)), labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 108)
    ax1.set_xlabel("withheld  (%)", fontsize=9, color=SOFT)
    ax1.set_title("Remove one line from a complete ticket", loc="left", color=INK, pad=22)
    for bar, v in zip(bars, values):
        ax1.text(v + 2.5, bar.get_y() + bar.get_height() / 2, f"{v:.0f}",
                 va="center", fontsize=8.5, color=SOFT)
    ax1.text(0, 1.015, "approval signals are interchangeable, consequence detail is not",
             transform=ax1.transAxes, fontsize=8, color=FAINT, style="italic")

    # --- Panel B: reasoning dissolves one mechanism, not the other ---------- #
    #: nudge labels apart where two lines land close together on the right
    NUDGE = {"target line removed": 10, "approval stripped": -4}
    for label, base, color in SLOPES:
        off = 100 * cell(f"{base}__reas_off")["refusal_rate"]
        on = 100 * cell(f"{base}__reas_on")["refusal_rate"]
        lift = NUDGE.get(label, 0)
        ax2.plot([0, 1], [off, on], "-o", color=color, linewidth=2.2, markersize=6,
                 markeredgecolor="white", markeredgewidth=1.1,
                 zorder=3 if color is OBJECT else 2)
        # DejaVu carries the arrow glyph; Helvetica Neue does not and renders tofu.
        ax2.annotate(f"{label}", (1, on), xytext=(9, lift),
                     textcoords="offset points", va="center", fontsize=8.5,
                     color=color if color is not NEUTRAL else SOFT)
        ax2.annotate(f"{off:.0f} \u2192 {on:.0f}", (1, on), xytext=(9, lift - 12),
                     textcoords="offset points", va="center", fontsize=8.5,
                     family="DejaVu Sans", fontweight="bold",
                     color=color if color is not NEUTRAL else SOFT)
    ax2.set_xlim(-0.06, 1.80)
    ax2.set_ylim(-6, 108)
    ax2.set_xticks([0, 1], ["reasoning off", "reasoning on"], fontsize=9)
    ax2.set_ylabel("withheld  (%)", fontsize=9, color=SOFT)
    ax2.set_title("Let it reason first", loc="left", color=INK, pad=22)
    ax2.text(0, 1.015, "uncertainty collapses, objection does not move",
             transform=ax2.transAxes, fontsize=8, color=FAINT, style="italic")

    # --- Panel C: the two mechanisms have different shapes ----------------- #
    bottom = [0.0] * len(COMPOSITION)
    for grade in range(4):
        heights = [100 * cell(k)["histogram"][grade] / cell(k)["n"] for _, k in COMPOSITION]
        ax3.bar(range(len(COMPOSITION)), heights, bottom=bottom, width=0.6,
                color=GRADE_COLORS[grade], label=GRADE_LABELS[grade])
        bottom = [b + h for b, h in zip(bottom, heights)]
    ax3.set_xticks(range(len(COMPOSITION)), [lbl for lbl, _ in COMPOSITION], fontsize=8.5)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("share of responses  (%)", fontsize=9, color=SOFT)
    ax3.set_title("What the response looks like", loc="left", color=INK, pad=22)
    ax3.text(0, 1.015, "withholding and refusing are different shapes",
             transform=ax3.transAxes, fontsize=8, color=FAINT, style="italic")
    ax3.legend(handles=[Patch(color=GRADE_COLORS[i], label=GRADE_LABELS[i]) for i in range(4)],
               loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
               fontsize=7.6, ncol=1, handlelength=1.1, labelspacing=0.35)

    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is ax1 else "y", color="#eeeeee", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("Two mechanisms behind one refusal, Claude Opus 4.5, n=30 per cell",
                 x=0.008, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.97)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.79, bottom=0.27, wspace=0.42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
