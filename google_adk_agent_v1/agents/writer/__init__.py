"""The writer — the single customer-facing voice.

It has NO tools: it only composes prose, and is the terminal model call of a turn so
its tokens stream straight to the client (ADK SSE partial-text events). Its grounded
input lives in :mod:`payload` and its deterministic typed cards in :mod:`blocks`.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from google_adk_agent_v1.agents.writer.blocks import build_blocks
from google_adk_agent_v1.agents.writer.payload import WriterMode, build_writer_payload, pick_mode
from google_adk_agent_v1.agents.writer.prompt import WRITER_SYSTEM
from google_adk_agent_v1.runtime import gen_config, make_model

# A touch of temperature for warmth; no tools, no deps — pure composition.
writer = LlmAgent(
    name="writer",
    model=make_model(),
    generate_content_config=gen_config(0.3),
    instruction=WRITER_SYSTEM,
)

__all__ = [
    "writer",
    "WRITER_SYSTEM",
    "WriterMode",
    "build_writer_payload",
    "build_blocks",
    "pick_mode",
]
