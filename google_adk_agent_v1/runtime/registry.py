"""A tiny process-level registry for sharing the live ``ShoppingDeps`` by reference.

This is the one place where Google ADK's design diverges from Pydantic AI's. In
Pydantic AI the shared cart rode in ``ctx.deps`` — a real object handed to every
tool. ADK instead gives tools a *session* whose ``state`` is a JSON-ish dict, and
that state is **deep-copied** as it flows between the runner, the agents, and the
sub-agents (see ``InMemorySessionService._copy_session``). A live ``CartService``
put directly into state would therefore be *cloned*, and a mutation in one agent
would never reach the next.

So we keep the live object here, in memory, and put only a short **string key** in
session state (a string survives a deep-copy unharmed). Every tool resolves the key
back to the one shared :class:`~google_adk_agent_v1.runtime.deps.ShoppingDeps`:

    deps = registry.get(tool_context.state[RUNTIME_KEY])

One live cart is mutated in place across the orchestrator and every delegated worker
run — exactly the "shared deps, passed by reference" property of the Pydantic build,
reconstructed structurally for ADK. The session registers its deps for the duration
of a turn and unregisters them when the turn ends, so nothing leaks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google_adk_agent_v1.runtime.deps import ShoppingDeps

# The session-state key under which the registry key is stored.
RUNTIME_KEY = "runtime_key"

_REGISTRY: dict[str, "ShoppingDeps"] = {}


def register(key: str, deps: "ShoppingDeps") -> None:
    _REGISTRY[key] = deps


def get(key: str) -> "ShoppingDeps":
    try:
        return _REGISTRY[key]
    except KeyError:  # pragma: no cover - a misconfigured turn
        raise RuntimeError(
            f"no ShoppingDeps registered for key {key!r}; the turn did not register "
            "its deps, or the session-state key was lost in transit."
        ) from None


def unregister(key: str) -> None:
    _REGISTRY.pop(key, None)
