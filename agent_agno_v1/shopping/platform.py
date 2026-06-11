"""Platform assembly — register leaf tools/guardrails, build the per-session team.

The ``TOOLS`` registry holds ONLY leaf tools (``search_products`` etc.); the
members and the supervisor are built per session so each conversation gets its own
live ``CartService`` + memory store, shared into every tool by reference through
Agno ``dependencies``. ``core`` never imports this module — wiring flows
shopping → core.
"""

from __future__ import annotations

from typing import Any

from agno.db.in_memory import InMemoryDb

from agent_agno_v1.core.factory import build_agent, build_team
from agent_agno_v1.core.guardrails import register_builtin_guardrails
from agent_agno_v1.core.registry import register_tool
from agent_agno_v1.shopping.agents import MEMBER_CONFIGS, SUPERVISOR_TEAM
from agent_agno_v1.shopping.domain import CartService, MemoryStore
from agent_agno_v1.shopping.tools import all_tools

_registered = False


def register_shopping_platform() -> None:
    """Register the leaf domain tools + builtin guardrails (idempotent, global)."""
    global _registered
    if _registered:
        return
    for tool in all_tools():
        register_tool(tool, replace=True)
    register_builtin_guardrails()
    _registered = True


def build_supervisor(
    cart_service: CartService,
    store: MemoryStore,
    *,
    db: Any | None = None,
    session_state: dict[str, Any] | None = None,
):
    """Build the 3 members + the coordinate-mode supervisor for ONE session.

    The live ``cart_service`` + ``store`` ride ``dependencies`` (by reference) so
    every member tool mutates the same cart for the whole turn. The leader owns the
    ``db`` (history-in-context); members are stateless per delegation.
    """
    register_shopping_platform()
    deps = {"cart": cart_service, "store": store}
    members = [build_agent(cfg, dependencies=deps) for cfg in MEMBER_CONFIGS]
    team = build_team(
        SUPERVISOR_TEAM,
        members,
        dependencies=deps,
        db=db or InMemoryDb(),
        session_state=session_state or {},
    )
    return team


__all__ = ["register_shopping_platform", "build_supervisor"]
