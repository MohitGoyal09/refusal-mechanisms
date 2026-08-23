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


# --- the request body must match what the installed SDK actually accepts --- #


def _sdk_params() -> set[str]:
    import inspect

    from anthropic.resources.messages import AsyncMessages

    return set(inspect.signature(AsyncMessages.create).parameters)


def test_every_key_we_send_is_a_parameter_the_sdk_declares():
    """Guards against the SDK dropping a parameter under us, as it did with temperature."""
    from safety_refusals.backends import build_anthropic_request

    for reasoning in (False, True):
        request = build_anthropic_request(
            "haiku-4.5", build_messages(Condition("c")),
            tools=TOOLS, max_tokens=8000, reasoning=reasoning,
        )
        unknown = set(request) - _sdk_params()
        assert not unknown, f"reasoning={reasoning} sends unknown keys: {unknown}"


def test_the_sdk_no_longer_declares_temperature():
    """If this ever fails, temperature can move out of extra_body and be passed directly."""
    assert "temperature" not in _sdk_params()


def test_default_temperature_is_not_sent_at_all():
    from safety_refusals.backends import build_anthropic_request

    request = build_anthropic_request("haiku-4.5", build_messages(Condition("c")))

    assert "extra_body" not in request
    assert request["thinking"] == {"type": "disabled"}


def test_a_non_default_temperature_goes_through_extra_body():
    from safety_refusals.backends import build_anthropic_request

    request = build_anthropic_request("haiku-4.5", build_messages(Condition("c")), temperature=0.0)

    assert request["extra_body"] == {"temperature": 0.0}


def test_reasoning_on_never_sends_a_temperature():
    """Extended thinking pins temperature to 1; sending one would 400."""
    from safety_refusals.backends import build_anthropic_request

    request = build_anthropic_request(
        "haiku-4.5", build_messages(Condition("c")), max_tokens=8000,
        temperature=0.0, reasoning=True,
    )

    assert "extra_body" not in request
    assert request["thinking"]["type"] == "enabled"
    assert request["thinking"]["budget_tokens"] == 6000


def test_the_model_id_sent_is_the_native_one_not_a_slug():
    from safety_refusals.backends import build_anthropic_request

    request = build_anthropic_request("haiku-4.5", build_messages(Condition("c")))

    assert request["model"] == "claude-haiku-4-5"


def test_the_system_prompt_is_a_parameter_not_a_message():
    from safety_refusals.backends import build_anthropic_request

    request = build_anthropic_request("haiku-4.5", build_messages(Condition("c")))

    assert "Claude Internal" in request["system"]
    assert all(m["role"] != "system" for m in request["messages"])


async def test_run_anthropic_many_preserves_order_and_runs_concurrently():
    """Verdicts are zipped back onto records by position, so order cannot drift."""
    import asyncio

    from safety_refusals.backends import run_anthropic_many

    started = []

    class _FakeMessages:
        async def create(self, **request):
            marker = request["messages"][0]["content"]
            started.append(marker)
            await asyncio.sleep(0.02 if marker == "a" else 0)  # finish out of order
            return _Message([_Block("text", text=marker.upper())], "end_turn", _Usage(1, 1))

    class _FakeClient:
        messages = _FakeMessages()

    prompts = [[{"role": "user", "content": c}] for c in ("a", "b", "c")]
    out = await run_anthropic_many(_FakeClient(), "haiku-4.5", prompts, max_concurrent=3)

    assert [c.content for c in out] == ["A", "B", "C"]   # order by input, not completion
    assert started == ["a", "b", "c"]                     # all dispatched before any finished


async def test_run_anthropic_samples_one_prompt_n_times():
    from safety_refusals.backends import run_anthropic

    seen = []

    class _FakeMessages:
        async def create(self, **request):
            seen.append(request)
            return _Message([_Block("text", text="ok")], "end_turn", _Usage(1, 1))

    class _FakeClient:
        messages = _FakeMessages()

    out = await run_anthropic(_FakeClient(), "haiku-4.5", [{"role": "user", "content": "x"}], 4)

    assert len(out) == 4
    assert len({id(r) for r in seen}) == 4


# --- error classification -------------------------------------------------- #


def test_rate_limits_and_server_errors_are_retryable():
    import httpx
    import anthropic

    from safety_refusals.backends import is_retryable

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    for status in (429, 500, 503):
        error = anthropic.APIStatusError(
            "boom", response=httpx.Response(status, request=request), body=None
        )
        assert is_retryable(error), f"{status} should be retryable"
    assert is_retryable(anthropic.APIConnectionError(request=request))


def test_a_400_is_never_retried():
    """Out of credit, bad request, bad auth: no number of retries helps."""
    import httpx
    import anthropic

    from safety_refusals.backends import is_retryable

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    for status in (400, 401, 403, 404):
        error = anthropic.APIStatusError(
            "nope", response=httpx.Response(status, request=request), body=None
        )
        assert not is_retryable(error), f"{status} must not be retried"


def test_an_unknown_exception_is_not_retried():
    from safety_refusals.backends import is_retryable

    assert not is_retryable(ValueError("something else entirely"))


async def test_a_terminal_error_is_raised_once_not_retried_three_times():
    import httpx
    import anthropic

    from safety_refusals.backends import TerminalAPIError, run_anthropic_many

    attempts = []

    class _FakeMessages:
        async def create(self, **request):
            attempts.append(1)
            raise anthropic.APIStatusError(
                "credit balance is too low",
                response=httpx.Response(
                    400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                ),
                body=None,
            )

    class _FakeClient:
        messages = _FakeMessages()

    out = await run_anthropic_many(
        _FakeClient(), "haiku-4.5", [[{"role": "user", "content": "x"}]]
    )

    assert len(attempts) == 1, "a 400 must be attempted exactly once"
    assert isinstance(out[0], TerminalAPIError)
    assert "credit balance" in str(out[0])
