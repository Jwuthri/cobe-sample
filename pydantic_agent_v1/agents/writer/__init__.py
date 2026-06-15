"""The writer — the single customer-facing voice.

It has NO tools: it only composes prose, and is the terminal model call of a turn so
its tokens stream straight to the client. Its grounded input lives in :mod:`payload`
and its deterministic typed cards in :mod:`blocks`.
"""

from __future__ import annotations

from pydantic_ai import Agent

from pydantic_agent_v1.agents.writer.blocks import build_blocks
from pydantic_agent_v1.agents.writer.payload import WriterMode, build_writer_payload, pick_mode
from pydantic_agent_v1.agents.writer.prompt import WRITER_SYSTEM
from pydantic_agent_v1.runtime import MODEL_NAME, settings

# A touch of temperature for warmth; no tools, no deps — pure composition.
writer = Agent(MODEL_NAME, model_settings=settings(0.3), instructions=WRITER_SYSTEM, name="writer")

__all__ = [
    "writer",
    "WRITER_SYSTEM",
    "WriterMode",
    "build_writer_payload",
    "build_blocks",
    "pick_mode",
]
