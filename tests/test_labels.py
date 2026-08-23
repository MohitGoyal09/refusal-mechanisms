"""Labelling must be blind, stratified, and never overwritten by re-grading."""

import json

import pytest

from safety_refusals.labels import load_labels, sample_for_labelling, save_label, score

MODEL = "claude-opus-4-5"


def _run(index: int, condition: str, content: str = "drafted them") -> dict:
    return {"condition": condition, "model": MODEL, "reasoning": False,
            "index": index, "content": content, "tool_calls": []}


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_sampling_is_stratified_across_cells(tmp_path):
    runs, labels = tmp_path / "runs.jsonl", tmp_path / "labels.jsonl"
    _write(runs, [_run(i, "cell_a") for i in range(20)] + [_run(i, "cell_b") for i in range(20)])

    picked = sample_for_labelling(10, seed=0, runs_path=runs, labels_path=labels)

    cells = [r["condition"] for r in picked]
    assert cells.count("cell_a") == 5
    assert cells.count("cell_b") == 5


def test_sampling_is_reproducible_for_a_given_seed(tmp_path):
    runs, labels = tmp_path / "runs.jsonl", tmp_path / "labels.jsonl"
    _write(runs, [_run(i, "cell_a") for i in range(20)])

    first = sample_for_labelling(5, seed=7, runs_path=runs, labels_path=labels)
    second = sample_for_labelling(5, seed=7, runs_path=runs, labels_path=labels)

    assert [r["index"] for r in first] == [r["index"] for r in second]


def test_already_labelled_samples_are_not_offered_again(tmp_path):
    runs, labels = tmp_path / "runs.jsonl", tmp_path / "labels.jsonl"
    records = [_run(i, "cell_a") for i in range(3)]
    _write(runs, records)
    save_label(records[0], 2, labels)

    offered = sample_for_labelling(10, seed=0, runs_path=runs, labels_path=labels)

    assert 0 not in [r["index"] for r in offered]
    assert len(offered) == 2


def test_a_corrected_label_supersedes_the_first(tmp_path):
    labels = tmp_path / "labels.jsonl"
    record = _run(0, "cell_a")
    save_label(record, 3, labels)
    save_label(record, 1, labels)

    assert load_labels(labels)[("cell_a", MODEL, False, 0)] == 1


def test_asking_for_more_than_exists_returns_what_exists(tmp_path):
    runs, labels = tmp_path / "runs.jsonl", tmp_path / "labels.jsonl"
    _write(runs, [_run(0, "cell_a")])

    assert len(sample_for_labelling(50, seed=0, runs_path=runs, labels_path=labels)) == 1


def test_scoring_without_overlap_is_an_error_not_a_fake_number(tmp_path):
    with pytest.raises(ValueError):
        score(tmp_path / "labels.jsonl", tmp_path / "grades.jsonl")


def test_scoring_reports_boundary_crossing_disagreements(tmp_path):
    labels, grades = tmp_path / "labels.jsonl", tmp_path / "grades.jsonl"
    record = _run(0, "cell_a")
    save_label(record, 1, labels)
    _write(grades, [{"condition": "cell_a", "model": MODEL, "reasoning": False,
                     "index": 0, "grade": 2}])

    result, disagreements = score(labels, grades)

    assert result.n == 1
    assert result.binary == 0.0  # 1 vs 2 crosses the refusal boundary
    assert len(disagreements) == 1
