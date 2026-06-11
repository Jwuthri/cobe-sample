"""Model resolution for the Agno port — one code path, env-driven.

A temperature becomes an :class:`agno.models.openai.OpenAIChat`. The provider
model is resolved from the same env chain v4_1 uses, so this package inherits
whatever the repo's ``.env`` configures:

    ``AGENT_AGNO_V4_1_MODEL`` → ``AGENT_V4_1_MODEL`` → ``AGENT_V2_OPENAI_MODEL``
    → ``gpt-4.1-mini``.

Newer OpenAI reasoning models (gpt-5.x) only accept the default temperature, so
we omit ``temperature`` for those rather than 400 on an unsupported value.
"""

from __future__ import annotations

import os

from agno.models.openai import OpenAIChat

_DEFAULT_MODEL = "gpt-4.1-mini"


def default_model_id() -> str:
    """Resolve the bare OpenAI model id from the env chain (no provider prefix)."""
    name = (
        os.environ.get("AGENT_AGNO_V4_1_MODEL")
        or os.environ.get("AGENT_V4_1_MODEL")
        or os.environ.get("AGENT_V2_OPENAI_MODEL")
        or _DEFAULT_MODEL
    )
    # Tolerate a "openai:gpt-..." style value by stripping the provider prefix.
    return name.split(":", 1)[1] if name.startswith("openai:") else name


def _supports_temperature(model_id: str) -> bool:
    # gpt-5.x / o-series reasoning models reject a custom temperature.
    head = model_id.lower()
    return not (head.startswith("gpt-5") or head.startswith("o1") or head.startswith("o3"))


def resolve_model(temperature: float = 0.0, model_id: str | None = None) -> OpenAIChat:
    """Build the chat model for a sub-agent / writer at the given temperature."""
    mid = model_id or default_model_id()
    if _supports_temperature(mid):
        return OpenAIChat(id=mid, temperature=temperature)
    return OpenAIChat(id=mid)


__all__ = ["default_model_id", "resolve_model"]
