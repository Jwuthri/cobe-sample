"""The writer — the single customer-facing voice.

It has NO tools: it only composes prose, and is the terminal model call of a turn so
its tokens stream straight to the client. Its grounded input lives in :mod:`payload`
and its deterministic typed cards in :mod:`blocks`.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from lg_agent_v2.agents.writer.blocks import build_blocks
from lg_agent_v2.agents.writer.payload import WriterMode, build_writer_payload, pick_mode
from lg_agent_v2.agents.writer.prompt import WRITER_SYSTEM
from lg_agent_v2.runtime import build_model


def build_writer(model: Any | None = None):
    """Compile the no-tools writer (its tokens stream to the client).

    A touch of temperature for warmth; no tools, no context — pure composition.
    """
    return create_agent(
        model=model or build_model(0.3),
        tools=[],
        system_prompt=WRITER_SYSTEM,
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
