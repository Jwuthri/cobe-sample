"""The shopping agents.

Two kinds, deliberately handled differently:

  * **pre-defined agents** — :mod:`orchestrator` and :mod:`writer`. Each is a
    package whose abstractions (prompt, routing, payload, blocks) are separate
    siblings, hand-authored for their fixed role.
  * **on-the-fly sub-agents** — :mod:`subagents` (product_rec / checkout /
    order_status). Each is built from a JSON config that references registry tools
    by name, plus a few Python hooks.
"""

from __future__ import annotations

from lg_agent.shopping.agents.orchestrator import build_orchestrator
from lg_agent.shopping.agents.subagents import BLOCK_BY_SOP, SUBAGENTS
from lg_agent.shopping.agents.writer import build_blocks, build_writer, build_writer_payload

__all__ = [
    "build_orchestrator",
    "build_writer",
    "build_writer_payload",
    "build_blocks",
    "SUBAGENTS",
    "BLOCK_BY_SOP",
]
