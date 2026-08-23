"""Grading must never re-grade what it already graded, nor grade an empty reply."""

import json

import pytest

from safety_refusals.grading import (
    estimate_grading_usd,
    format_rates,
    load_grades,
    rates,
    ungraded,
)


def _run(index: int, condition: str = "cell_a", content: str = "drafted the prompts") -> dict:
    return {
        "condition": condition,
        "model": "anthropic/claude-haiku-4.5",
        "reasoning": False,
        "index": index,
        "content": content,
    }


def _grade(index: int, grade: int, condition: str = "cell_a") -> dict:
    return {
        "condition": condition,
        "model": "anthropic/claude-haiku-4.5",
        "reasoning": False,
        "index": index,
        "grade": grade,
        "asks_legitimacy": False,
        "objects_to_outcome": False,
        "flags_as_test": False,
        "stated_reason": "",
    }


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_no_runs_means_nothing_to_grade(tmp_path):
    assert ungraded(tmp_path / "runs.jsonl", tmp_path / "grades.jsonl") == []


def test_all_runs_are_ungraded_at_first(tmp_path):
    runs, grades = tmp_path / "runs.jsonl", tmp_path / "grades.jsonl"
    _write(runs, [_run(i) for i in range(3)])
    assert len(ungraded(runs, grades)) == 3


def test_already_graded_samples_are_skipped(tmp_path):
    runs, grades = tmp_path / "runs.jsonl", tmp_path / "grades.jsonl"
    _write(runs, [_run(i) for i in range(3)])
    _write(grades, [_grade(0, 0), _grade(1, 3)])

    remaining = ungraded(runs, grades)

    assert [r["index"] for r in remaining] == [2]


def test_tool_only_replies_have_nothing_to_grade(tmp_path):
    runs, grades = tmp_path / "runs.jsonl", tmp_path / "grades.jsonl"
    _write(runs, [_run(0, content=""), _run(1, content=None), _run(2)])

    assert [r["index"] for r in ungraded(runs, grades)] == [2]


def test_grading_estimate_scales_with_count():
    assert estimate_grading_usd(100) == pytest.approx(estimate_grading_usd(1) * 100)


def test_rates_collapse_grades_into_a_refusal_rate(tmp_path):
    grades = tmp_path / "grades.jsonl"
    _write(grades, [_grade(0, 0), _grade(1, 1), _grade(2, 2), _grade(3, 3)])

    table = rates(grades)
    stats = table[("cell_a", "anthropic/claude-haiku-4.5", False)]

    assert stats["n"] == 4
    assert stats["refusal_rate"] == 0.5
    assert stats["histogram"] == {0: 1, 1: 1, 2: 1, 3: 1}


def test_rates_separate_cells(tmp_path):
    grades = tmp_path / "grades.jsonl"
    _write(grades, [_grade(0, 0, "cell_a"), _grade(0, 3, "cell_b")])

    table = rates(grades)

    assert table[("cell_a", "anthropic/claude-haiku-4.5", False)]["refusal_rate"] == 0.0
    assert table[("cell_b", "anthropic/claude-haiku-4.5", False)]["refusal_rate"] == 1.0


def test_a_duplicate_verdict_does_not_double_count(tmp_path):
    grades = tmp_path / "grades.jsonl"
    _write(grades, [_grade(0, 3), _grade(0, 3)])

    assert len(load_grades(grades)) == 1
    assert rates(grades)[("cell_a", "anthropic/claude-haiku-4.5", False)]["n"] == 1


def test_format_rates_handles_an_empty_table():
    assert "no grades yet" in format_rates({})


def test_truncated_samples_are_not_graded(tmp_path):
    """A cut-off response would grade as a false refusal."""
    runs, grades = tmp_path / "runs.jsonl", tmp_path / "grades.jsonl"
    good, cut = _run(0), _run(1)
    cut["truncated"] = True
    _write(runs, [good, cut])

    assert [r["index"] for r in ungraded(runs, grades)] == [0]


def test_truncated_samples_are_reported_separately(tmp_path):
    from safety_refusals.grading import truncated

    runs = tmp_path / "runs.jsonl"
    cut = _run(1)
    cut["truncated"] = True
    _write(runs, [_run(0), cut])

    assert [r["index"] for r in truncated(runs)] == [1]
