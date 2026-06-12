"""Platform assembly — register leaf tools/middleware, build the agents.

The registry holds ONLY leaf tools (``search_products`` etc.) and middleware.
Sub-agents are NOT registry tools: they are declared in ``SUBAGENTS`` (with their
Python extractors), wrapped as delegate tools here, and handed to the orchestrator
via ``build_agent(..., delegates=...)`` — never registered globally. ``core`` never
imports this module — wiring flows shopping → core.

The orchestrator's ``empty_cart_guard`` middleware composes a ``tool_enabled``
gate into the checkout delegate (handled by the factory), so the checkout tool is
hidden while the cart is empty.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from openai_agent_v1.core.factory import build_agent
from openai_agent_v1.core.registry import register_tool
from openai_agent_v1.core.subagent import build_subagent_tools
from openai_agent_v1.shopping.agents import ORCHESTRATOR_AGENT, SUBAGENTS, WRITER_AGENT
from openai_agent_v1.shopping.context import ShoppingContext
from openai_agent_v1.shopping.domain.memory import build_store
from openai_agent_v1.shopping.middleware import register_shopping_middleware
from openai_agent_v1.shopping.tools import all_tools

# One long-term-memory store for the demo (swap for a DB-backed store in prod).
_STORE = build_store()
_registered = False
_DELEGATES: list[Any] | None = None


def register_shopping_platform(store: Any | None = None) -> None:
    """Register the leaf domain tools + shopping middleware (idempotent)."""
    global _registered
    if _registered:
        return
    for tool in all_tools():
        register_tool(tool, replace=True)
    register_shopping_middleware()
    _registered = True


def subagent_delegates(store: Any | None = None) -> list[Any]:
    """Build (once) the sub-agents wrapped as the orchestrator's delegate tools."""
    global _DELEGATES
    if _DELEGATES is None:
        register_shopping_platform(store=store)
        _DELEGATES = build_subagent_tools(
            SUBAGENTS,
            build_agent=build_agent,
            context=ShoppingContext,
            store=store or _STORE,
        )
    return _DELEGATES


def build_orchestrator(store: Any | None = None) -> Any:
    """Compile the router orchestrator, with the sub-agents wired in as delegates.

    Parallel tool calls are disabled so a compound message ("a green cap AND check
    ORD-7") routes one sub-agent per turn — keeping the live event bus ordered
    (router → … → step before the next sub-agent), matching v4_1's sequential
    routing.
    """
    orch = build_agent(
        ORCHESTRATOR_AGENT,
        context=ShoppingContext,
        store=store or _STORE,
        delegates=subagent_delegates(store),
    )
    orch.model_settings = dataclasses.replace(orch.model_settings, parallel_tool_calls=False)
    return orch


def build_writer() -> Any:
    """Compile the no-tools writer (its tokens stream to the client)."""
    return build_agent(WRITER_AGENT)


def store() -> Any:
    return _STORE


__all__ = [
    "register_shopping_platform",
    "subagent_delegates",
    "build_orchestrator",
    "build_writer",
    "store",
]
