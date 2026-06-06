"""OpenAI client and chat-model factories.

Two thin helpers:
  - ``classifier_client()``  — bare OpenAI client for structured-output
    classification (used by the supervisor).
  - ``chat_model()``         — a langchain-openai ``ChatOpenAI`` instance
    suitable as the ``model`` argument to ``create_agent``.

The model name resolves ``AGENT_V4_OPENAI_MODEL`` → ``AGENT_V2_OPENAI_MODEL``
→ ``gpt-4.1-mini`` so v4 picks up the same model the rest of the repo is
configured with (the dev ``.env`` only defines the ``AGENT_V2_*`` names).
The key comes from ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from openai import OpenAI

_DEFAULT_MODEL = "gpt-4.1-mini"


def model_name() -> str:
    return (
        os.environ.get("AGENT_V4_OPENAI_MODEL")
        or os.environ.get("AGENT_V2_OPENAI_MODEL")
        or _DEFAULT_MODEL
    )


def writer_model_name() -> str:
    return (
        os.environ.get("AGENT_V4_WRITER_MODEL")
        or os.environ.get("AGENT_V2_WRITER_MODEL")
        or model_name()
    )


def classifier_client() -> OpenAI:
    """Bare OpenAI client; used by the supervisor for structured output."""
    return OpenAI()


def chat_model(*, temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    """langchain-openai ChatOpenAI; passed to ``create_agent`` as ``model``."""
    kwargs: dict = {"model": model_name(), "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)
