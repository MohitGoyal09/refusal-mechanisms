"""No two cell names may describe the same experiment. That is money, not style."""

import pytest

from safety_refusals.conditions import prompt_hash
from safety_refusals.experiments import REGISTRY

ALL_SETS = sorted(REGISTRY)


def _all_cells():
    return [c for name in ALL_SETS for c in REGISTRY[name]()]


def test_one_name_per_experiment_across_every_set():
    """Identity is (prompt, reasoning). Two names for one identity is a double charge."""
    by_identity: dict[tuple, set[str]] = {}
    for cond in _all_cells():
        by_identity.setdefault((prompt_hash(cond), cond.reasoning), set()).add(cond.name)

    collisions = {k: sorted(v) for k, v in by_identity.items() if len(v) > 1}
    assert not collisions, f"same experiment under several names: {collisions}"


def test_one_experiment_per_name():
    """The reverse: a name must not describe two different prompts."""
    by_name: dict[str, set[tuple]] = {}
    for cond in _all_cells():
        by_name.setdefault(cond.name, set()).add((prompt_hash(cond), cond.reasoning))

    ambiguous = {k: v for k, v in by_name.items() if len(v) > 1}
    assert not ambiguous, f"one name, several experiments: {ambiguous}"


def test_every_cell_name_declares_its_reasoning_setting():
    for cond in _all_cells():
        suffix = "__reas_on" if cond.reasoning else "__reas_off"
        assert cond.name.endswith(suffix), f"{cond.name} does not declare reasoning={cond.reasoning}"


@pytest.mark.parametrize("name", ALL_SETS)
def test_no_set_repeats_a_cell_within_itself(name):
    cells = REGISTRY[name]()
    names = [c.name for c in cells]
    assert len(names) == len(set(names)), f"{name} repeats a cell: {names}"


def test_the_shared_cells_are_actually_shared():
    """exp1, exp2, exp3 and replication all need the fully specified ticket."""
    users = [n for n in ALL_SETS if any(c.name == "full_spec__reas_off" for c in REGISTRY[n]())]
    assert set(users) >= {"exp1", "exp2", "exp3", "pilot", "replication"}
