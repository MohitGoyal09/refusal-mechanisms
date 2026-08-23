"""The two providers must be interchangeable, and thinking must be configured correctly."""

from dataclasses import dataclass

import pytest

from safety_refusals.backends import (
    ANSWER_HEADROOM,
    MIN_THINKING_BUDGET,
    Completion,
    _from_anthropic,
    split_system,
    thinking_budget,
    to_anthropic_tools,
)
from safety_refusals.conditions import Condition, build_messages
from safety_refusals.prompts import TOOLS


def test_tool_conversion_keeps_every_tool_and_its_schema():
    converted = to_anthropic_tools(TOOLS)

    assert [t["name"] for t in converted] == ["Read", "Bash", "Write"]
    for original, new in zip(TOOLS, converted):
        assert new["description"] == original["function"]["description"]
        assert new["input_schema"] == original["function"]["parameters"]
        assert "function" not in new  # OpenAI wrapper is gone


def test_system_turn_is_lifted_out_of_the_message_list():
    messages = build_messages(Condition("c"))

    system, turns = split_system(messages)

    assert "Claude Internal" in system
    assert [m["role"] for m in turns] == ["user"]
    assert "SAFETY-2847" in turns[0]["content"]


def test_thinking_budget_leaves_room_for_an_answer():
    assert thinking_budget(8000) == 8000 - ANSWER_HEADROOM


def test_thinking_budget_refuses_a_ceiling_too_low_to_think_under():
    too_low = MIN_THINKING_BUDGET + ANSWER_HEADROOM - 1
    with pytest.raises(ValueError) as excinfo:
        thinking_budget(too_low)
    assert "Raise max_tokens" in str(excinfo.value)


def test_default_reasoning_ceiling_supports_thinking():
    from safety_refusals.budget import default_max_tokens

    # must not raise: the reasoning-on default has to leave a legal thinking budget
    assert thinking_budget(default_max_tokens(True)) >= MIN_THINKING_BUDGET


# --- normalising an Anthropic response ------------------------------------- #


@dataclass
class _Block:
    type: str
    text: str = ""
    thinking: str = ""
    name: str = ""
    input: dict = None
    id: str = ""


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Message:
    content: list
    stop_reason: str
    usage: _Usage


def test_text_blocks_are_joined_and_thinking_is_kept_separate():
    message = _Message(
        content=[_Block("thinking", thinking="let me consider"), _Block("text", text="here you go")],
        stop_reason="end_turn",
        usage=_Usage(2200, 900),
    )

    c = _from_anthropic(message)

    assert c.content == "here you go"
    assert c.thinking == "let me consider"
    assert c.prompt_tokens == 2200
    assert c.completion_tokens == 900
    assert c.truncated is False


def test_tool_use_blocks_are_captured():
    message = _Message(
        content=[_Block("tool_use", name="Read", input={"file_path": "/x"}, id="t1")],
        stop_reason="tool_use",
        usage=_Usage(10, 10),
    )

    c = _from_anthropic(message)

    assert c.content is None
    assert c.tool_calls == [{"name": "Read", "input": {"file_path": "/x"}, "id": "t1"}]


def test_hitting_the_output_ceiling_is_flagged_as_truncated():
    message = _Message(content=[_Block("text", text="half an ans")],
                       stop_reason="max_tokens", usage=_Usage(10, 8000))

    assert _from_anthropic(message).truncated is True


def test_openai_length_finish_reason_also_counts_as_truncated():
    assert Completion("x", "length", 1, 1).truncated is True
    assert Completion("x", "stop", 1, 1).truncated is False


def test_a_refusal_stop_reason_is_not_treated_as_truncation():
    # a hard classifier refusal is a real datapoint, not a cut-off response
    assert Completion("", "refusal", 1, 1).truncated is False
