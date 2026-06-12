"""Model resolution for the OpenAI Agents SDK.

A ``ModelConfig`` becomes a ``(model_name, ModelSettings)`` pair. The config keeps
agent_v4_1's ``"provider:model"`` convention; the SDK's model id is the bare name,
so we strip a leading ``openai:`` (and map ``provider/model`` style through for
litellm). ``provider_model=None`` resolves to the env default so the shopping
agents inherit whatever the repo's ``.env`` configures (``AGENT_V2_OPENAI_MODEL``).
"""

from __future__ import annotations

import os

from agents import ModelSettings

from openai_agent_v1.core.config import ModelConfig

_DEFAULT_PROVIDER_MODEL = "openai:gpt-5.4-mini"


def default_provider_model() -> str:
    """Resolve the default ``provider:model`` from the env chain.

    ``OPENAI_AGENT_V1_MODEL`` → ``AGENT_V4_1_MODEL`` → ``AGENT_V2_OPENAI_MODEL`` →
    ``openai:gpt-5.4-mini``. A bare name (no ``provider:`` prefix) is assumed
    OpenAI, matching the dev ``.env`` which sets the model id without a prefix.
    """
    name = (
        os.environ.get("OPENAI_AGENT_V1_MODEL")
        or os.environ.get("AGENT_V4_1_MODEL")
        or os.environ.get("AGENT_V2_OPENAI_MODEL")
    )
    if not name:
        return _DEFAULT_PROVIDER_MODEL
    return name if ":" in name else f"openai:{name}"


def resolve_model_name(provider_model: str | None) -> str:
    """Map a ``provider:model`` string onto the SDK's model id.

    ``openai:gpt-5.4-mini`` → ``gpt-5.4-mini`` (the SDK's default provider is
    OpenAI). A non-openai provider is rendered ``provider/model`` for litellm.
    """
    name = provider_model or default_provider_model()
    if ":" not in name:
        return name
    provider, model = name.split(":", 1)
    if provider == "openai":
        return model
    return f"{provider}/{model}"


def resolve_model_settings(cfg: ModelConfig) -> ModelSettings:
    """Build the ``ModelSettings`` described by ``cfg`` (temperature / max_tokens)."""
    kwargs: dict = {"temperature": cfg.temperature}
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return ModelSettings(**kwargs)


__all__ = ["default_provider_model", "resolve_model_name", "resolve_model_settings"]
