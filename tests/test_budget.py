"""The guard is the whole point: it must refuse before money moves."""

from dataclasses import dataclass

import pytest

from safety_refusals.budget import (
    MAX_TOKENS_CAP,
    BudgetExceeded,
    Ledger,
    cost_usd,
    estimate_usd,
    inr,
    price_of,
)

HAIKU = "anthropic/claude-haiku-4.5"
OPUS = "anthropic/claude-opus-4.5"


def test_cost_matches_published_rates():
    # Haiku 4.5 is $1/MTok in, $5/MTok out
    assert cost_usd(HAIKU, 1_000_000, 0) == pytest.approx(1.0)
    assert cost_usd(HAIKU, 0, 1_000_000) == pytest.approx(5.0)
    # Opus 4.5 is $5/$25
    assert cost_usd(OPUS, 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        price_of("anthropic/claude-made-up-9")


def test_reasoning_on_costs_more_than_reasoning_off():
    off = estimate_usd(HAIKU, 50, reasoning=False)
    on = estimate_usd(HAIKU, 50, reasoning=True)
    assert on > off


def test_estimate_scales_linearly_in_calls():
    one = estimate_usd(OPUS, 1)
    hundred = estimate_usd(OPUS, 100)
    assert hundred == pytest.approx(one * 100)


def test_ledger_allows_a_batch_inside_the_cap():
    ledger = Ledger(cap_usd=1.0)
    ledger.check("cell", 0.5)  # must not raise
    ledger.record("cell", 0.5)
    assert ledger.remaining_usd == pytest.approx(0.5)


def test_ledger_refuses_a_batch_that_would_cross_the_cap():
    ledger = Ledger(cap_usd=1.0)
    ledger.record("first", 0.9)
    with pytest.raises(BudgetExceeded) as excinfo:
        ledger.check("second", 0.2)
    assert "over the cap" in str(excinfo.value)


def test_ledger_remaining_never_goes_negative():
    ledger = Ledger(cap_usd=1.0)
    ledger.record("overshoot", 5.0)
    assert ledger.remaining_usd == 0.0


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Response:
    usage: _Usage


def test_record_responses_uses_real_usage_not_the_estimate():
    ledger = Ledger(cap_usd=10.0)
    responses = [_Response(_Usage(2000, 1000)) for _ in range(3)]

    spent = ledger.record_responses("cell", HAIKU, responses)

    # 6000 input tokens at $1/MTok + 3000 output at $5/MTok
    assert spent == pytest.approx(6000 / 1e6 + 3000 * 5 / 1e6)
    assert ledger.spent_usd == pytest.approx(spent)


def test_record_responses_tolerates_a_response_with_no_usage():
    ledger = Ledger(cap_usd=10.0)

    @dataclass
    class Bare:
        usage: None = None

    assert ledger.record_responses("cell", HAIKU, [Bare()]) == 0.0


def test_inr_conversion_is_applied():
    assert inr(1.0).startswith("Rs ")
    assert "95" in inr(1.0) or "96" in inr(1.0)


def test_max_tokens_cap_is_well_below_the_upstream_default():
    assert MAX_TOKENS_CAP < 16000
