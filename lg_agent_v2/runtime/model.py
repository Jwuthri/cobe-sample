"""Model resolution — one place that decides which chat model the agents talk to.

``provider:model`` strings (e.g. ``"openai:gpt-5.4-mini"``) are handed to LangChain's
``init_chat_model``. The name is resolved from the env so the whole repo can be
pointed at a different model without touching code:

    LG_AGENT_V2_MODEL  →  AGENT_V2_OPENAI_MODEL  →  openai:gpt-5.4-mini

Each agent gets its OWN model instance (temperature differs per role), built by
:func:`build_model`. This is the LangChain analogue of Pydantic AI's ``Agent(model,
model_settings=...)`` — there the model is a shared string + per-agent settings; here
it is a per-agent ``BaseChatModel``.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model

_DEFAULT_MODEL = "openai:gpt-5.4-mini"


def model_name() -> str:
    name = os.environ.get("LG_AGENT_V2_MODEL") or os.environ.get("AGENT_V2_OPENAI_MODEL")
    if not name:
        return _DEFAULT_MODEL
    return name if ":" in name else f"openai:{name}"  # a bare id is assumed OpenAI


MODEL_NAME = model_name()


def build_model(temperature: float, **kwargs: Any):
    """Build the chat model the agents talk to, at the given temperature."""
    return init_chat_model(MODEL_NAME, temperature=temperature, **kwargs)


__all__ = ["MODEL_NAME", "model_name", "build_model"]
