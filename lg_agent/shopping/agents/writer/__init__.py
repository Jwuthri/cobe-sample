"""The writer — pre-defined agent #2 (the customer-facing voice).

Its abstractions are split into siblings: :mod:`prompt` (the voice), :mod:`payload`
(its grounded JSON input), and :mod:`blocks` (its typed structured output). It has
NO tools — it only composes prose, and is the terminal model call of a turn so its
tokens stream straight to the client.
"""

from __future__ import annotations

from typing import Any

from lg_agent.core.builder import build_agent
from lg_agent.shopping.agents.writer.blocks import build_blocks
from lg_agent.shopping.agents.writer.payload import build_writer_payload
from lg_agent.shopping.agents.writer.prompt import WRITER_SYSTEM

MODEL = "openai:gpt-5.4-mini"

CONFIG = {
    "name": "writer",
    "description": "Compose the single user-facing reply from verified step results + cart.",
    "system_prompt": WRITER_SYSTEM,
    "model": {"provider_model": MODEL, "temperature": 0.3},
    "tools": [],
}


def build_writer() -> Any:
    """Compile the no-tools writer (its tokens stream to the client)."""
    return build_agent(CONFIG)


__all__ = ["CONFIG", "WRITER_SYSTEM", "build_writer", "build_writer_payload", "build_blocks"]
