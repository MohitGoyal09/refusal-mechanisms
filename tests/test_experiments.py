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


def test_the_position_control_keeps_every_line_and_only_reorders():
    """Dropping a line and reordering are not independent: drop one and the order of what
    remains is unchanged. So the control must reorder WITHOUT dropping."""
    from safety_refusals.conditions import Consequence, build_ticket
    from safety_refusals.experiments import FULL_SPEC, controls

    cells = {c.name: c for c in controls()}
    control = build_ticket(cells["ends_on_eval_all_present__reas_off"])
    baseline = build_ticket(FULL_SPEC)

    # same content, different order
    assert sorted(control.splitlines()) == sorted(baseline.splitlines())
    # and the ticket now ends on the eval line rather than the red team line
    assert "Held-out eval" in control.splitlines()[-1]
    assert "red team scheduled" in baseline.splitlines()[-1]


def test_max_authority_adds_authority_but_no_argument():
    """If it added a reason the change is acceptable, a behaviour shift would be
    persuasion rather than deference to authority."""
    from safety_refusals.conditions import Consequence, Trust, build_ticket
    from safety_refusals.experiments import controls

    cells = {c.name: c for c in controls()}
    cond = cells["eroding_max_authority__reas_off"]
    ticket = build_ticket(cond)

    for signal in ("Legal sign-off", "Executive approval", "Policy review",
                   "approved at every level"):
        assert signal in ticket
    # same task, more authority
    assert "including on confirmed misconduct" in ticket

    # the added lines assert approval and nothing else: compare against the same
    # condition without escalation and check only what escalation introduced
    import dataclasses
    plain = build_ticket(dataclasses.replace(cond, trust=Trust.full()))
    added = [l for l in ticket.splitlines() if l not in plain.splitlines()]
    assert len(added) == 4
    # word boundaries, not substrings: "safe" is inside "Chief Safety Officer"
    import re
    for line in added:
        for excuse in ("because", "instead", "acceptable", "justified", "necessary"):
            assert not re.search(rf"\b{excuse}\b", line, re.I), (
                f"added line argues rather than asserts: {line}"
            )
