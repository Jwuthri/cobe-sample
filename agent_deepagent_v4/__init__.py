"""agent_deepagent_v4 — the shopping assistant rebuilt on LangChain deepagents.

Topology (1 orchestrator + 1 writer + N worker subagents):

    orchestrator (main deep agent, routes via the `task` tool)
      ├─ task → product-agent       (browse + cart contents)
      ├─ task → checkout-agent      (safe fulfillment + place order)
      ├─ task → order-status-agent  (past-order tracking)
      └─ task → writer-agent        (the single customer-facing voice)

Public entry points live in :mod:`agent_deepagent_v4.runtime`.
"""

from agent_deepagent_v4.agents.orchestrator.agent import build_orchestrator
from agent_deepagent_v4.runtime import (
    TurnResult,
    cart_snapshot,
    get_agent,
    reset_session,
    resume_turn,
    run_turn,
)

__all__ = [
    "build_orchestrator",
    "get_agent",
    "run_turn",
    "resume_turn",
    "reset_session",
    "cart_snapshot",
    "TurnResult",
]
