"""The writer — the single customer-facing voice.

It has NO tools: it only composes prose, and is the terminal model call of a turn so
its tokens stream straight to the client. Its grounded input lives in :mod:`payload`
and its deterministic typed cards in :mod:`blocks`.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from lg_agent_v3.agents.writer.blocks import build_blocks
from lg_agent_v3.agents.writer.payload import WriterMode, build_writer_payload, pick_mode
from lg_agent_v3.agents.writer.prompt import WRITER_SYSTEM
from lg_agent_v3.runtime import ShoppingDeps, build_model, compile_guardrails


def build_writer(model: Any | None = None, guardrails: list | None = None):
    """Compile the no-tools writer (its tokens stream to the client).

    A touch of temperature for warmth; no tools, no context — pure composition. An
    ``on_output`` guardrail here forces the session into buffered mode (no token stream).
    """
    return create_agent(
        model=model or build_model(0.3),
        tools=[],
        system_prompt=WRITER_SYSTEM,
        context_schema=ShoppingDeps,
        middleware=compile_guardrails(guardrails, "writer"),
        name="writer",
    )


writer = build_writer()

__all__ = [
    "writer",
    "build_writer",
    "WRITER_SYSTEM",
    "WriterMode",
    "build_writer_payload",
    "build_blocks",
    "pick_mode",
]
