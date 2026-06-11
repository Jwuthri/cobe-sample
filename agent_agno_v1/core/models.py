"""Model resolution — a ``ModelConfig`` becomes an Agno ``OpenAIChat``.

agent_v4_1 routed every model through LangChain's ``init_chat_model(
"openai:gpt-4.1-mini", temperature=...)``. The Agno equivalent is
``OpenAIChat(id="gpt-4.1-mini", temperature=...)`` (Chat Completions API — the
match for ``init_chat_model``; ``OpenAIResponses`` is the Responses API and the
library's ``model=None`` default, which we deliberately avoid).

Temperature is a *per-model-instance* setting in Agno (not per-agent), so each
agent/member gets its own model object.
"""

from __future__ import annotations

import os

from agno.models.openai import OpenAIChat

from agent_agno_v1.core.config import ModelConfig

_DEFAULT_MODEL_ID = "gpt-4.1-mini"


def default_model_id() -> str:
    """Resolve the default model id from the env chain.

    ``AGENT_AGNO_V1_MODEL`` → ``AGENT_V2_OPENAI_MODEL`` → ``gpt-4.1-mini``.
    A ``provider:model`` string has the provider prefix stripped (OpenAI only).
    """
    name = os.environ.get("AGENT_AGNO_V1_MODEL") or os.environ.get("AGENT_V2_OPENAI_MODEL")
    if not name:
        return _DEFAULT_MODEL_ID
    return _strip_provider(name)


def _strip_provider(name: str) -> str:
    """``"openai:gpt-4.1-mini"`` → ``"gpt-4.1-mini"``; a bare id passes through."""
    if ":" in name:
        provider, _, model = name.partition(":")
        if provider and provider != "openai":
            raise ValueError(
                f"agent_agno_v1 only supports the 'openai' provider; got {provider!r}. "
                "Use an OpenAI model id (e.g. 'gpt-4.1-mini')."
            )
        return model
    return name


def resolve_model(cfg: ModelConfig) -> OpenAIChat:
    """Build the Agno chat model described by ``cfg``."""
    model_id = _strip_provider(cfg.provider_model) if cfg.provider_model else default_model_id()
    kwargs: dict = {"id": model_id, "temperature": cfg.temperature}
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return OpenAIChat(**kwargs)


__all__ = ["default_model_id", "resolve_model"]
