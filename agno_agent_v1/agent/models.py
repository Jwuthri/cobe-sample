"""Model resolution — one code path to an ``agno`` chat model.

The model id is resolved from the env (``AGNO_AGENT_V1_MODEL`` →
``AGENT_V2_OPENAI_MODEL`` → a sane default), matching the repo's ``.env``. A
``provider:`` prefix is stripped (Agno's ``OpenAIChat`` wants a bare id).

gpt-5.x / o-series models reject a custom ``temperature`` (only the default is
allowed), so it is omitted for them — set it only where the model accepts it.
"""

from __future__ import annotations

import os

from agno.models.openai import OpenAIChat

_DEFAULT_MODEL_ID = "gpt-5.4-mini"


def default_model_id() -> str:
    name = os.environ.get("AGNO_AGENT_V1_MODEL") or os.environ.get("AGENT_V2_OPENAI_MODEL")
    if not name:
        return _DEFAULT_MODEL_ID
    return name.split(":", 1)[1] if name.startswith("openai:") else name


def _supports_custom_temperature(model_id: str) -> bool:
    mid = model_id.lower()
    return not (mid.startswith("gpt-5") or mid.startswith(("o1", "o3", "o4")))


def resolve_model(model_id: str | None = None, temperature: float | None = None) -> OpenAIChat:
    """Build the chat model. ``temperature`` is applied only where supported."""
    mid = model_id or default_model_id()
    kwargs: dict = {}
    if temperature is not None and _supports_custom_temperature(mid):
        kwargs["temperature"] = temperature
    return OpenAIChat(id=mid, **kwargs)


__all__ = ["default_model_id", "resolve_model"]
