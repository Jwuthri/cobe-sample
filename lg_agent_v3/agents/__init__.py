"""The agents — five minds, each defined in one self-contained file.

  * :mod:`orchestrator` — the router; the only one that reads the conversation.
  * :mod:`product_rec`, :mod:`checkout`, :mod:`order_status` — the workers the
    orchestrator delegates to (agent-as-tool).
  * :mod:`writer` — the single customer-facing voice (streams its reply).

This module just collects them: the worker registry, the worker→block-kind map, and
re-exports of the orchestrator + writer the session drives.
"""

from __future__ import annotations

from lg_agent_v3.agents import checkout, order_status, product_rec
from lg_agent_v3.agents.orchestrator import (
    ROUTER_PROMPT,
    WORKERS,
    absorb_recalls,
    build_memo,
    build_orchestrator,
    orchestrator,
)
from lg_agent_v3.agents.writer import (
    WRITER_SYSTEM,
    build_blocks,
    build_writer,
    build_writer_payload,
    pick_mode,
    writer,
)

# worker name → the writer block kind it produces (consumed by build_blocks).
BLOCK_BY_SOP: dict[str, str | None] = {w.name: w.block for w in WORKERS}

__all__ = [
    "orchestrator",
    "build_orchestrator",
    "writer",
    "build_writer",
    "WORKERS",
    "BLOCK_BY_SOP",
    "ROUTER_PROMPT",
    "WRITER_SYSTEM",
    "build_memo",
    "absorb_recalls",
    "build_writer_payload",
    "build_blocks",
    "pick_mode",
]
