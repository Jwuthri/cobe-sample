"""The on-the-fly sub-agents — built from JSON config + registry tools.

Each module here defines one :class:`~lg_agent.core.subagent.SubAgent` (config +
hooks). This package collects them and knows how to compile them into the
orchestrator's delegate tools.
"""

from __future__ import annotations

from typing import Any

from lg_agent.core.builder import build_agent
from lg_agent.core.subagent import SubAgent, build_delegate_tools
from lg_agent.shopping.agents.subagents import checkout, order_status, product_rec
from lg_agent.shopping.context import ShoppingContext

# The routable sub-agents, in routing-priority order.
SUBAGENTS: list[SubAgent] = [product_rec.SUBAGENT, checkout.SUBAGENT, order_status.SUBAGENT]

# sub-agent name -> the writer block kind it produces (consumed by build_blocks).
BLOCK_BY_SOP: dict[str, str | None] = {s.name: s.block for s in SUBAGENTS}


def build_delegates(store: Any | None = None) -> list[Any]:
    """Compile every sub-agent and wrap it as an orchestrator delegate tool."""
    return build_delegate_tools(
        SUBAGENTS,
        build_agent=build_agent,
        context_schema=ShoppingContext,
        store=store,
    )


__all__ = ["SUBAGENTS", "BLOCK_BY_SOP", "build_delegates"]
