"""Platform assembly — register leaf tools/middleware, build the agents.

The registry holds ONLY leaf tools (``search_products`` etc.) and middleware.
Sub-agents are NOT registry tools: they are declared in ``SUBAGENTS`` (with their
Python extractors), wrapped as delegate tools here, and handed to the orchestrator
via ``build_agent(..., delegates=...)`` — never registered globally, never listed
in a config's ``tools``. ``core`` never imports this module — wiring flows
shopping → core.
"""

from __future__ import annotations

from typing import Any

from agent_v4_1.core.factory import build_agent
from agent_v4_1.core.registry import register_tool
from agent_v4_1.core.subagent import build_subagent_tools
from agent_v4_1.shopping.agents import ORCHESTRATOR_AGENT, SUBAGENTS, WRITER_AGENT
from agent_v4_1.shopping.context import ShoppingContext
from agent_v4_1.shopping.domain.memory import build_store
from agent_v4_1.shopping.middleware import register_shopping_middleware
from agent_v4_1.shopping.tools import all_tools

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
            context_schema=ShoppingContext,
            store=store or _STORE,
        )
    return _DELEGATES


def build_orchestrator(store: Any | None = None) -> Any:
    """Compile the router orchestrator, with the sub-agents wired in as delegates."""
    return build_agent(
        ORCHESTRATOR_AGENT,
        context_schema=ShoppingContext,
        store=store or _STORE,
        delegates=subagent_delegates(store),
    )


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
