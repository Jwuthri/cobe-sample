"""Agno model factories (replaces ``agent_v2.llm``).

The single OpenAI model used by every agent + the supervisor classifier
+ the writer. Model name from ``AGENT_V3_OPENAI_MODEL`` (falling back to
the legacy ``AGENT_V2_OPENAI_MODEL``), default ``gpt-4.1-mini``. Key from
``OPENAI_API_KEY``.

In agent_v2 there were two clients: a bare ``openai.OpenAI`` for the
classifier's structured output and a ``langchain_openai.ChatOpenAI`` for
the sub-agents. Agno unifies both — ``OpenAIChat`` powers the agents and,
via ``output_schema=``, the classifier's structured output too.
"""

from __future__ import annotations

import os

from agno.models.openai import OpenAIChat


def model_name() -> str:
    return (
        os.environ.get("AGENT_V3_OPENAI_MODEL")
        or os.environ.get("AGENT_V2_OPENAI_MODEL")
        or "gpt-4.1-mini"
    )


def writer_model_name() -> str:
    return (
        os.environ.get("AGENT_V3_WRITER_MODEL")
        or os.environ.get("AGENT_V2_WRITER_MODEL")
        or model_name()
    )


def chat_model(temperature: float = 0.0) -> OpenAIChat:
    """Deterministic model for sub-agents + the classifier."""
    return OpenAIChat(id=model_name(), temperature=temperature)


def writer_model(temperature: float = 0.3) -> OpenAIChat:
    """Slightly warmer model for the user-facing writer."""
    return OpenAIChat(id=writer_model_name(), temperature=temperature)
