"""Planning must be honest about what is already stored, and dry runs must spend nothing."""

import pytest

from safety_refusals import store
from safety_refusals.budget import Ledger
from safety_refusals.conditions import Condition
from safety_refusals.experiments import REGISTRY
from safety_refusals.runner import plan_cell, plan_matrix, run_cell

HAIKU = "anthropic/claude-haiku-4.5"


def test_plan_counts_what_is_missing(tmp_path):
    path = tmp_path / "runs.jsonl"
    cond = Condition("cell_a")
    store.append(
        [{"condition": "cell_a", "model": HAIKU, "reasoning": False, "index": i} for i in range(12)],
        path,
    )

    plan = plan_cell(cond, HAIKU, 30, path)

    assert plan.have == 12
    assert plan.need == 18
    assert plan.estimated_usd > 0


def test_plan_for_a_finished_cell_costs_nothing(tmp_path):
    path = tmp_path / "runs.jsonl"
    cond = Condition("cell_a")
    store.append(
        [{"condition": "cell_a", "model": HAIKU, "reasoning": False, "index": i} for i in range(30)],
        path,
    )

    plan = plan_cell(cond, HAIKU, 30, path)

    assert plan.need == 0
    assert plan.estimated_usd == 0.0


async def test_dry_run_spends_nothing_and_calls_nothing(tmp_path):
    path = tmp_path / "runs.jsonl"
    ledger = Ledger(cap_usd=100.0)

    result = await run_cell(None, Condition("cell_a"), HAIKU, 10, ledger, dry_run=True, path=path)

    assert result == []
    assert ledger.spent_usd == 0.0
    assert not path.exists()


async def test_max_tokens_above_the_cap_is_refused(tmp_path):
    ledger = Ledger(cap_usd=100.0)
    with pytest.raises(ValueError) as excinfo:
        await run_cell(
            None, Condition("cell_a"), HAIKU, 1, ledger,
            max_tokens=16000, path=tmp_path / "runs.jsonl",
        )
    assert "exceeds the cap" in str(excinfo.value)


async def test_a_finished_cell_makes_no_call_even_without_dry_run(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append(
        [{"condition": "cell_a", "model": HAIKU, "reasoning": False, "index": i} for i in range(5)],
        path,
    )
    ledger = Ledger(cap_usd=100.0)

    # client is None, so any real call would raise
    result = await run_cell(None, Condition("cell_a"), HAIKU, 5, ledger, path=path)

    assert len(result) == 5
    assert ledger.spent_usd == 0.0


async def test_budget_guard_fires_before_any_call(tmp_path):
    from safety_refusals.budget import BudgetExceeded

    ledger = Ledger(cap_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        await run_cell(None, Condition("cell_a"), HAIKU, 50, ledger, path=tmp_path / "runs.jsonl")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_experiment_plans_cleanly(name, tmp_path):
    conditions = REGISTRY[name]()
    assert conditions, f"{name} produced no cells"
    names = [c.name for c in conditions]
    assert len(names) == len(set(names)), f"{name} has duplicate cell names: {names}"

    plans = plan_matrix(conditions, HAIKU, 5, tmp_path / "runs.jsonl")
    assert all(p.need == 5 for p in plans)


async def test_matrix_refuses_when_the_whole_plan_exceeds_the_cap(tmp_path):
    from safety_refusals.budget import BudgetExceeded
    from safety_refusals.experiments import exp1_trust_consequence
    from safety_refusals.runner import run_matrix

    with pytest.raises(BudgetExceeded):
        await run_matrix(
            None, exp1_trust_consequence(), "anthropic/claude-opus-4.5",
            50, cap_usd=1.0, path=tmp_path / "runs.jsonl",
        )


def test_exp3_control_shares_a_cell_with_exp1(tmp_path):
    """The control must not be billed twice under a second name."""
    from safety_refusals.experiments import exp1_trust_consequence, exp3_components

    exp1_names = {c.name for c in exp1_trust_consequence()}
    exp3_names = {c.name for c in exp3_components()}
    assert "full_spec__reas_off" in exp1_names & exp3_names


def test_reasoning_filter_selects_only_matching_cells():
    import sys
    sys.path.insert(0, "experiments")
    from run import select

    from safety_refusals.experiments import exp1_trust_consequence

    cells = exp1_trust_consequence()
    off = select(cells, "off")
    on = select(cells, "on")

    assert len(off) == 4 and all(not c.reasoning for c in off)
    assert len(on) == 4 and all(c.reasoning for c in on)
    assert len(select(cells, "both")) == 8


def test_reasoning_filter_can_empty_a_set():
    import sys
    sys.path.insert(0, "experiments")
    from run import select

    from safety_refusals.experiments import exp3_components

    # exp3 is reasoning-off only by design
    assert select(exp3_components(), "on") == []
