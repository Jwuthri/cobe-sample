"""OpenAI client and chat-model factories.

Two thin helpers:
  - ``classifier_client()``  — bare OpenAI client for structured-output
    classification (used by the supervisor).
  - ``chat_model()``         — a langchain-openai ``ChatOpenAI`` instance
    suitable as the ``model`` argument to ``create_agent``.

The model name is read from ``AGENT_V2_OPENAI_MODEL`` (default
``gpt-4.1-mini``). The key comes from ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from openai import OpenAI


def model_name() -> str:
    return os.environ.get("AGENT_V2_OPENAI_MODEL", "gpt-5.4-mini")


def classifier_client() -> OpenAI:
    """Bare OpenAI client; used by the supervisor for structured output."""
    return OpenAI()


def chat_model() -> ChatOpenAI:
    """langchain-openai ChatOpenAI; passed to ``create_agent`` as ``model``."""
    return ChatOpenAI(model=model_name(), temperature=0)
