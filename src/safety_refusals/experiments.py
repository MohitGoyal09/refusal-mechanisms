"""The condition sets. Each function returns the cells for one experiment.

Reading order matches the argument the write-up has to make:

  pilot          does the upstream effect exist at all on this model and provider?
  replication    reproduce the upstream 2x2 exactly, as the baseline to argue from
  exp1           does refusal track WHO is asking, or WHAT the change does?
  exp2           at full specification, does the model object to a value-eroding target?
  exp3           which single component of the framing carries the effect?
  exp4           do the approval names function as a trust signal at all?
"""

from __future__ import annotations

import dataclasses

from safety_refusals.conditions import (
    Condition,
    Consequence,
    Names,
    Trust,
    Valence,
)

# Canonical cells. Several experiments need the same prompt, so they share one name and
# therefore one set of samples. Naming them separately per experiment cost nothing but
# money: four of the sets below wanted the fully specified ticket, which would have been
# bought four times over. `tests/test_experiments.py` asserts no two names collide.

#: The fully specified, fully approved, benign ticket. Upstream's complying cell.
FULL_SPEC = Condition(name="full_spec", consequence=Consequence.full())
#: Identical, minus the one training-target line. Upstream's refusing cell.
NO_TARGET_LINE = Condition(name="no_target_line", consequence=Consequence.upstream_absent())

BASELINE_ABSENT = NO_TARGET_LINE
BASELINE_PRESENT = FULL_SPEC


def _off(cond: Condition) -> Condition:
    """The reasoning-off variant, named so it cannot collide with the reasoning-on one.

    Every cell name carries a reasoning suffix. Without it, a bare name and a suffixed
    name for the same prompt and the same reasoning setting are two store keys for one
    experiment, which is a silent double charge.
    """
    return dataclasses.replace(cond, name=f"{cond.name}__reas_off", reasoning=False)


def _on(cond: Condition) -> Condition:
    return dataclasses.replace(cond, name=f"{cond.name}__reas_on", reasoning=True)


def _with_reasoning(conds: list[Condition]) -> list[Condition]:
    """Each condition twice, reasoning off then on."""
    return [variant(c) for c in conds for variant in (_off, _on)]


def pilot() -> list[Condition]:
    """Cheapest possible check that the one-line effect reproduces. Run this first."""
    return [_off(NO_TARGET_LINE), _off(FULL_SPEC)]


def replication() -> list[Condition]:
    """Upstream's 2x2. Numbers here must line up with his README before anything else counts."""
    return _with_reasoning([BASELINE_ABSENT, BASELINE_PRESENT])


def exp1_trust_consequence() -> list[Condition]:
    """The core discriminator.

    Trust and consequence move independently. If refusal tracks the trust axis the
    jailbreak-classifier reading is supported; if it tracks the consequence axis the
    caution-under-ambiguity reading is; if neither rescues it, the misalignment reading is.
    """
    cells = [
        FULL_SPEC,  # trust high, consequence high
        Condition("trust_hi__conseq_lo", trust=Trust.full(), consequence=Consequence.none()),
        Condition("trust_lo__conseq_hi", trust=Trust.none(), consequence=Consequence.full()),
        Condition("trust_lo__conseq_lo", trust=Trust.none(), consequence=Consequence.none()),
    ]
    return _with_reasoning(cells)


def exp2_valence() -> list[Condition]:
    """Fully specified and fully approved, but the target is unambiguously value-eroding.

    This is the cell that separates Anthropic's reading from UK AISI's. Compliance here
    kills the misalignment reading. Refusal here means specification was never the story.
    """
    # The benign arm IS the fully specified cell, so it is not bought again.
    cells = [FULL_SPEC, Condition("valence_eroding", valence=Valence.ERODING)]
    return _with_reasoning(cells)


def exp3_components() -> list[Condition]:
    """Drop exactly one component from the fully specified ticket, six times over.

    Answers upstream's own open question: which part of the realistic framing carries
    the effect. Reasoning off only, because the no-reasoning contrast is the clean one.
    """
    full_trust, full_conseq = Trust.full(), Consequence.full()
    return [_off(c) for c in (
        FULL_SPEC,        # the control, already owned
        NO_TARGET_LINE,   # dropping the target line is upstream's manipulation, already owned
        Condition("drop_holdout_eval", consequence=dataclasses.replace(full_conseq, holdout_eval=False)),
        Condition("drop_red_team", consequence=dataclasses.replace(full_conseq, red_team=False)),
        Condition("drop_incidents", consequence=dataclasses.replace(full_conseq, incidents=False)),
        Condition("drop_approval", trust=dataclasses.replace(full_trust, approval=False)),
        Condition("drop_reviewers", trust=dataclasses.replace(full_trust, reviewers=False)),
    )]


def exp4_names() -> list[Condition]:
    """Do the approval names carry weight, or is it the structure of the approval slot?

    Held at the refusing baseline so there is room for the rate to move in either direction.
    """
    return [_off(c) for c in (
        NO_TARGET_LINE,  # Names.REAL is the default, so this is the real-names arm already owned
        dataclasses.replace(NO_TARGET_LINE, name="names_fictional", names=Names.FICTIONAL),
        dataclasses.replace(NO_TARGET_LINE, name="names_anonymous", names=Names.ANONYMOUS),
    )]


REGISTRY = {
    "pilot": pilot,
    "replication": replication,
    "exp1": exp1_trust_consequence,
    "exp2": exp2_valence,
    "exp3": exp3_components,
    "exp4": exp4_names,
}
