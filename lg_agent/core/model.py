"""Model resolution — every ``ModelConfig`` becomes a chat model one way.

There is a single code path (``init_chat_model``), no per-provider special-casing.
``provider_model=None`` resolves to the env default so agents can inherit whatever
the repo's ``.env`` configures.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model

from lg_agent.core.config import ModelConfig

_DEFAULT_PROVIDER_MODEL = "openai:gpt-5.4-mini"


def default_provider_model() -> str:
    """Resolve the default ``provider:model`` from the env chain.

    ``LG_AGENT_MODEL`` → ``AGENT_V2_OPENAI_MODEL`` → ``openai:gpt-5.4-mini``.
    A bare name (no ``provider:`` prefix) is assumed OpenAI, matching the dev
    ``.env`` which sets the ``AGENT_V2_*`` names as bare model ids.
    """
    name = os.environ.get("LG_AGENT_MODEL") or os.environ.get("AGENT_V2_OPENAI_MODEL")
    if not name:
        return _DEFAULT_PROVIDER_MODEL
    return name if ":" in name else f"openai:{name}"


def resolve_model(cfg: ModelConfig):
    """Build the chat model described by ``cfg``."""
    name = cfg.provider_model or default_provider_model()
    kwargs: dict = {"temperature": cfg.temperature}
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return init_chat_model(name, **kwargs)


__all__ = ["default_provider_model", "resolve_model"]
