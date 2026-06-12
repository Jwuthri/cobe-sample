"""Platform setup — populate the registries once, hold the shared memory store.

The registries are global singletons (see :mod:`lg_agent.core.registry`). This is
the single place that fills them for the shopping tenant: the built-in guardrails +
middleware, the leaf tools, and the two tenant middlewares. :func:`register_shopping`
is idempotent and runs automatically when ``lg_agent.shopping`` is imported.
"""

from __future__ import annotations

from typing import Any

from lg_agent.core import register_builtins, register_tool
from lg_agent.shopping.domain.memory import build_store
from lg_agent.shopping.middleware import register_shopping_middleware
from lg_agent.shopping.tools import all_tools

# One long-term-memory store for the demo (swap for a DB-backed store in prod).
_STORE = build_store()
_registered = False


def register_shopping() -> None:
    """Register the platform built-ins + the shopping leaf tools/middleware (idempotent)."""
    global _registered
    if _registered:
        return
    register_builtins()  # guardrail + middleware factories
    for tool in all_tools():
        register_tool(tool, replace=True)
    register_shopping_middleware()
    _registered = True


def store() -> Any:
    """The shared long-term-memory store (addresses / payment / order history)."""
    return _STORE


__all__ = ["register_shopping", "store"]
