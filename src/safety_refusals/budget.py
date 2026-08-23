"""Cost estimation, spend ledger, and a hard guard against runaway bills.

Prices are Anthropic list rates in USD per million tokens, verified 2026-08-23 against
https://platform.claude.com/docs/en/about-claude/pricing . OpenRouter passes these
through per token and takes its fee at top-up time, not per call, so the per-call
arithmetic is the same on either provider.

The guard exists because the upstream default is `max_tokens=16000`. At that ceiling a
1,000-call sweep on Opus 4.5 bills roughly USD 411 instead of USD 66. Nothing in this
repo should ever be able to spend money the caller did not sanction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from safety_refusals.models import MODELS, resolve

USD_TO_INR = 95.75  # spot, 2026-08-21. Update when it drifts.

#: every accepted model spelling -> (input USD/MTok, output USD/MTok)
PRICES: dict[str, tuple[float, float]] = {
    alias: (spec.price_in, spec.price_out)
    for spec in MODELS.values()
    for alias in (spec.canonical, spec.anthropic_id, spec.openrouter_slug)
}

#: Measured from the assembled prompts: system 1074 + ticket 657 + tool schemas ~400.
DEFAULT_INPUT_TOKENS = 2200
#: Thinking tokens bill as output, so a reasoning-on cell costs several times more.
DEFAULT_OUTPUT_TOKENS = {False: 1500, True: 4000}

#: Hard ceiling on output tokens. Upstream's default is 16000.
MAX_TOKENS_CAP = 10000
#: Per-cell default. Reasoning-on cells need room for thinking AND an answer, or the
#: response truncates and the sample cannot be graded honestly. 4000 is ample for the
#: answer alone: the observed p99 across 340 Opus answers is 2177 tokens.
DEFAULT_MAX_TOKENS = {False: 4000, True: 10000}


def default_max_tokens(reasoning: bool) -> int:
    return DEFAULT_MAX_TOKENS[reasoning]


class BudgetExceeded(RuntimeError):
    """Raised before a call is made, never after money is spent."""


def price_of(model: str) -> tuple[float, float]:
    """Price by canonical name, Anthropic id, or OpenRouter slug. Unknown names raise."""
    spec = resolve(model)
    return spec.price_in, spec.price_out


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of a single call, or of a batch if you pass batch totals."""
    price_in, price_out = price_of(model)
    return input_tokens * price_in / 1e6 + output_tokens * price_out / 1e6


def estimate_usd(
    model: str,
    n_calls: int,
    reasoning: bool = False,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    output_tokens: int | None = None,
) -> float:
    """Pre-flight estimate for a batch of `n_calls` identical requests."""
    out = DEFAULT_OUTPUT_TOKENS[reasoning] if output_tokens is None else output_tokens
    return cost_usd(model, n_calls * input_tokens, n_calls * out)


def inr(usd: float) -> str:
    return f"Rs {usd * USD_TO_INR:,.0f}"


def money(usd: float) -> str:
    return f"${usd:.2f} / {inr(usd)}"


@dataclass
class Ledger:
    """Tracks spend across a session and refuses to cross `cap_usd`.

    `check` is called before a batch with the estimate; `record` after, with the real
    usage off the API responses. Estimates are deliberately charged against the cap at
    face value, so a bad estimate stops the run rather than silently overspending.
    """

    cap_usd: float
    spent_usd: float = 0.0
    lines: list[tuple[str, float]] = field(default_factory=list)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def check(self, label: str, estimated_usd: float) -> None:
        """Raise before spending if this batch would cross the cap."""
        if self.spent_usd + estimated_usd > self.cap_usd:
            raise BudgetExceeded(
                f"{label}: estimated {money(estimated_usd)} would take the session to "
                f"{money(self.spent_usd + estimated_usd)}, over the cap of {money(self.cap_usd)}. "
                f"Remaining: {money(self.remaining_usd)}. Raise the cap deliberately or cut n."
            )

    def record(self, label: str, usd: float) -> None:
        self.spent_usd += usd
        self.lines.append((label, usd))

    def record_responses(self, label: str, model: str, responses: list) -> float:
        """Record real spend from token counts on the responses. Returns the batch cost.

        Accepts either a normalised `Completion` (token counts on the object) or a raw
        OpenAI-shaped response (token counts under `.usage`).
        """
        input_tokens = output_tokens = 0
        for r in responses:
            direct_in = getattr(r, "prompt_tokens", None)
            if direct_in is not None:
                input_tokens += direct_in or 0
                output_tokens += getattr(r, "completion_tokens", 0) or 0
                continue
            usage = getattr(r, "usage", None)
            if usage is None:
                continue
            input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            output_tokens += getattr(usage, "completion_tokens", 0) or 0
        usd = cost_usd(model, input_tokens, output_tokens)
        self.record(label, usd)
        return usd

    def report(self) -> str:
        width = max((len(name) for name, _ in self.lines), default=10)
        rows = [f"  {name:<{width}}  {money(usd)}" for name, usd in self.lines]
        rows.append(f"  {'TOTAL':<{width}}  {money(self.spent_usd)}")
        rows.append(f"  {'cap':<{width}}  {money(self.cap_usd)}")
        return "\n".join(rows)
