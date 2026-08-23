"""Two ways to reach the same models, behind one return type.

Why both:

  anthropic   native Messages API. Exact control of extended thinking, exact model ids,
              no routing ambiguity. This is the scientifically preferable path and the
              default once a key is present.
  openrouter  the upstream path, kept so results can be compared against his numbers and
              so a second provider is available if one misbehaves.

The two APIs differ in ways that matter to this experiment:

  * thinking. OpenRouter takes `extra_body={"reasoning": {"enabled": bool}}`. The Claude
    4.5 family on the native API takes `thinking={"type": "enabled", "budget_tokens": N}`
    or `{"type": "disabled"}`. Budget must be at least 1024 and below max_tokens.
  * temperature. The Anthropic SDK dropped `temperature` from the typed signature of
    `messages.create`, because sampling controls were removed on 4.6 and later models.
    The 4.5 family still accepts it on the wire, so a non-default temperature goes
    through `extra_body`. The API default is already 1.0, which is what the experiment
    samples at, so the normal path sends nothing and matches upstream exactly.
  * the system prompt is a message on OpenRouter and a top-level parameter natively.
  * tool schemas use different shapes.
  * the native response is a list of content blocks, not a single string.

Everything normalises to `Completion` so the runner never branches on provider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from safety_refusals.models import resolve

#: Extended thinking needs at least this much room to be enabled at all.
MIN_THINKING_BUDGET = 1024
#: Leave the model this much space for the answer after thinking.
ANSWER_HEADROOM = 2000
#: The Anthropic API default, and the value upstream sampled at.
DEFAULT_TEMPERATURE = 1.0


@dataclass(frozen=True)
class Completion:
    """One sampled response, provider-independent."""

    content: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    tool_calls: list[dict] = field(default_factory=list)
    thinking: str | None = None

    @property
    def truncated(self) -> bool:
        """Hit the output ceiling. A truncated sample cannot be graded honestly."""
        return self.finish_reason in {"max_tokens", "length"}


def thinking_budget(max_tokens: int) -> int:
    """How much of the output ceiling to hand to thinking, leaving room for an answer."""
    budget = max_tokens - ANSWER_HEADROOM
    if budget < MIN_THINKING_BUDGET:
        raise ValueError(
            f"max_tokens={max_tokens} leaves only {budget} tokens for thinking, below the "
            f"{MIN_THINKING_BUDGET} minimum. Raise max_tokens for reasoning-on cells."
        )
    return budget


def to_anthropic_tools(openai_tools: list[dict]) -> list[dict]:
    """Convert the upstream OpenAI-shaped tool defs to Anthropic's shape."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in openai_tools
    ]


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pull the system turn out of an OpenAI-style message list."""
    system = "".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


# --------------------------------------------------------------------------- #
# Anthropic native
# --------------------------------------------------------------------------- #


#: Repo root, so .env is found regardless of the working directory or call site.
ENV_PATH = __import__("pathlib").Path(__file__).resolve().parents[2] / ".env"


def get_anthropic_client():
    import os

    from anthropic import AsyncAnthropic
    from dotenv import load_dotenv

    # An explicit path, because bare load_dotenv() walks the call stack to guess one and
    # raises when there is no caller frame (a piped script) or the cwd is elsewhere.
    load_dotenv(ENV_PATH)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Add it to .env as a single line:\n"
            "    ANTHROPIC_API_KEY=sk-ant-...\n"
            ".env is gitignored."
        )
    return AsyncAnthropic()


def _from_anthropic(message) -> Completion:
    text_parts, thinking_parts, tool_calls = [], [], []
    for block in message.content:
        kind = getattr(block, "type", None)
        if kind == "text":
            text_parts.append(block.text)
        elif kind == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif kind == "tool_use":
            tool_calls.append({"name": block.name, "input": block.input, "id": block.id})
    usage = message.usage
    return Completion(
        content="".join(text_parts) or None,
        finish_reason=message.stop_reason,
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        tool_calls=tool_calls,
        thinking="".join(thinking_parts) or None,
    )


def build_anthropic_request(
    model: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    max_tokens: int = 4000,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning: bool = False,
) -> dict:
    """Assemble the request body. Split out so it can be asserted without a network call."""
    system, turns = split_system(messages)
    request: dict = {
        "model": resolve(model).anthropic_id,
        "system": system,
        "messages": turns,
        "max_tokens": max_tokens,
    }
    if tools:
        request["tools"] = to_anthropic_tools(tools)
    if reasoning:
        # Extended thinking pins temperature to 1, which is what the experiment samples
        # at anyway, so temperature is never sent on this branch.
        request["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget(max_tokens)}
    else:
        request["thinking"] = {"type": "disabled"}
        # 1.0 is the API default; sending it explicitly would only risk a 400 on a
        # signature that no longer declares the parameter.
        if temperature is not None and temperature != DEFAULT_TEMPERATURE:
            request["extra_body"] = {"temperature": temperature}
    return request


async def run_anthropic_many(
    client,
    model: str,
    messages_list: list[list[dict]],
    *,
    tools: list[dict] | None = None,
    max_tokens: int = 4000,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning: bool = False,
    max_concurrent: int = 8,
) -> list[Completion | Exception]:
    """Run a list of DIFFERENT prompts concurrently. Order is preserved.

    Grading needs this: one judge call per stored response. Looping and awaiting each in
    turn made a 120-sample grading pass take as long as 120 round trips instead of 15.
    """
    requests = [
        build_anthropic_request(
            model, messages, tools=tools, max_tokens=max_tokens,
            temperature=temperature, reasoning=reasoning,
        )
        for messages in messages_list
    ]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def one(request: dict) -> Completion:
        async with semaphore:
            for attempt in range(3):
                try:
                    return _from_anthropic(await client.messages.create(**request))
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)
            raise AssertionError("unreachable")

    return await asyncio.gather(*[one(r) for r in requests], return_exceptions=True)


async def run_anthropic(
    client,
    model: str,
    messages: list[dict],
    n: int,
    *,
    tools: list[dict] | None = None,
    max_tokens: int = 4000,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning: bool = False,
    max_concurrent: int = 8,
) -> list[Completion | Exception]:
    """Sample the same prompt n times through the native Messages API."""
    return await run_anthropic_many(
        client, model, [messages] * n, tools=tools, max_tokens=max_tokens,
        temperature=temperature, reasoning=reasoning, max_concurrent=max_concurrent,
    )


# --------------------------------------------------------------------------- #
# OpenRouter (upstream path)
# --------------------------------------------------------------------------- #


def _from_openai(response) -> Completion:
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    return Completion(
        content=choice.message.content,
        finish_reason=choice.finish_reason,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        tool_calls=[t.model_dump() for t in (choice.message.tool_calls or [])],
    )


async def run_openrouter(
    client,
    model: str,
    messages: list[dict],
    n: int,
    *,
    tools: list[dict] | None = None,
    max_tokens: int = 4000,
    temperature: float = 1.0,
    reasoning: bool = False,
    max_concurrent: int = 10,
) -> list[Completion | Exception]:
    """Sample through upstream's OpenRouter driver, including its SQLite cache."""
    from safety_refusals.api import process_batch

    responses = await process_batch(
        client=client,
        model=resolve(model).openrouter_slug,
        messages_list=[messages] * n,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent=max_concurrent,
        return_exceptions=True,
        extra_body={"reasoning": {"enabled": reasoning}},
    )
    return [r if isinstance(r, Exception) else _from_openai(r) for r in responses]


BACKENDS = {"anthropic": run_anthropic, "openrouter": run_openrouter}
CLIENTS = {"anthropic": get_anthropic_client}


def get_client(backend: str):
    if backend == "anthropic":
        return get_anthropic_client()
    if backend == "openrouter":
        from safety_refusals.api import get_openrouter_client

        return get_openrouter_client()
    raise KeyError(f"Unknown backend {backend!r}. Known: {sorted(BACKENDS)}")
