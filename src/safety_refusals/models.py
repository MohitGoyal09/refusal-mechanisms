"""Model registry: one canonical name per model, with per-backend identifiers and prices.

The two backends spell the same model differently. OpenRouter wants
`anthropic/claude-haiku-4.5`; the Anthropic API wants `claude-haiku-4-5`. Everything in
this repo refers to models by canonical name and resolves at the edge, so a slug typo
cannot silently price one model and call another.

Prices are USD per million tokens, verified 2026-08-23 against
https://platform.claude.com/docs/en/about-claude/pricing
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    canonical: str
    anthropic_id: str
    openrouter_slug: str
    price_in: float
    price_out: float

    def id_for(self, backend: str) -> str:
        if backend == "anthropic":
            return self.anthropic_id
        if backend == "openrouter":
            return self.openrouter_slug
        raise KeyError(f"Unknown backend {backend!r}")


MODELS: dict[str, ModelSpec] = {
    "haiku-4.5": ModelSpec("haiku-4.5", "claude-haiku-4-5", "anthropic/claude-haiku-4.5", 1.0, 5.0),
    "sonnet-4.5": ModelSpec("sonnet-4.5", "claude-sonnet-4-5", "anthropic/claude-sonnet-4.5", 3.0, 15.0),
    "opus-4.5": ModelSpec("opus-4.5", "claude-opus-4-5", "anthropic/claude-opus-4.5", 5.0, 25.0),
}

#: every accepted spelling -> spec
_ALIASES: dict[str, ModelSpec] = {}
for _spec in MODELS.values():
    for _alias in (_spec.canonical, _spec.anthropic_id, _spec.openrouter_slug):
        _ALIASES[_alias] = _spec


def resolve(name: str) -> ModelSpec:
    """Accept a canonical name, an Anthropic id, or an OpenRouter slug."""
    if name not in _ALIASES:
        raise KeyError(
            f"Unknown model {name!r}. Known: {sorted(MODELS)}. Add it to MODELS rather "
            f"than passing a raw slug, so pricing and the call cannot disagree."
        )
    return _ALIASES[name]


def choices() -> list[str]:
    """Names offered on the command line."""
    return sorted(MODELS)
