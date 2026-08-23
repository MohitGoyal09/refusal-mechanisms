"""A cell must never be billed twice. That is the store's only job."""

from safety_refusals import store
from safety_refusals.store import RunKey

KEY = RunKey("cell_a", "anthropic/claude-haiku-4.5", False)


def _record(index: int, condition: str = "cell_a", reasoning: bool = False) -> dict:
    return {
        "condition": condition,
        "model": "anthropic/claude-haiku-4.5",
        "reasoning": reasoning,
        "index": index,
        "content": f"response {index}",
    }


def test_empty_store_reports_nothing(tmp_path):
    path = tmp_path / "runs.jsonl"
    assert store.load(path) == []
    assert store.count_for(KEY, path) == 0
    assert store.missing(KEY, 10, path) == 10


def test_append_then_count(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append([_record(i) for i in range(4)], path)
    assert store.count_for(KEY, path) == 4
    assert store.missing(KEY, 10, path) == 6


def test_topping_up_only_needs_the_shortfall(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append([_record(i) for i in range(30)], path)
    assert store.missing(KEY, 30, path) == 0


def test_missing_never_goes_negative(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append([_record(i) for i in range(50)], path)
    assert store.missing(KEY, 30, path) == 0


def test_cells_are_keyed_separately(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append([_record(0, condition="cell_a"), _record(0, condition="cell_b")], path)
    assert store.count_for(RunKey("cell_a", KEY.model, False), path) == 1
    assert store.count_for(RunKey("cell_b", KEY.model, False), path) == 1


def test_reasoning_flag_is_part_of_the_key(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append([_record(0, reasoning=False), _record(0, reasoning=True)], path)
    assert store.count_for(RunKey("cell_a", KEY.model, False), path) == 1
    assert store.count_for(RunKey("cell_a", KEY.model, True), path) == 1


def test_appending_nothing_does_not_create_a_file(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append([], path)
    assert not path.exists()


def test_samples_are_returned_in_write_order(tmp_path):
    path = tmp_path / "runs.jsonl"
    store.append([_record(i) for i in range(3)], path)
    assert [r["index"] for r in store.samples_for(KEY, path)] == [0, 1, 2]
