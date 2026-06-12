"""Model resolution — always through ``init_chat_model``.

A ``ModelConfig`` becomes a chat model with one code path (no ``ChatOpenAI``
special-casing). ``provider_model=None`` resolves to the env default so the
shopping sub-agents inherit whatever the repo's ``.env`` configures
(``AGENT_V2_OPENAI_MODEL``), exactly like v4.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model

from agent_v4_1.core.config import ModelConfig

_DEFAULT_PROVIDER_MODEL = "openai:gpt-5.4-mini"


def default_provider_model() -> str:
    """Resolve the default ``provider:model`` from the env chain.

    ``AGENT_V4_1_MODEL`` → ``AGENT_V2_OPENAI_MODEL`` → ``openai:gpt-4.1-mini``.
    A bare name (no ``provider:`` prefix) is assumed OpenAI, matching the dev
    ``.env`` which only sets the ``AGENT_V2_*`` names as bare model ids.
    """
    name = os.environ.get("AGENT_V4_1_MODEL") or os.environ.get("AGENT_V2_OPENAI_MODEL")
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
