"""Per-turn dependencies — the Agno analogue of agent_v2's ``RuntimeContext``.

agent_v2 threaded a ``RuntimeContext(user_id, session_id, cart_service)``
into each sub-agent via ``agent.invoke(..., context=ctx)`` and tools read
it through ``ToolRuntime[RuntimeContext]``.

Agno's equivalent is the ``dependencies`` dict passed to
``workflow.run(..., dependencies=...)`` / ``agent.run(..., dependencies=...)``.
Agno injects a ``run_context: RunContext`` into tools and steps that ask
for it; tools then read ``run_context.dependencies["cart_service"]`` (the
live cart for this turn) and ``run_context.dependencies["store"]`` (the
long-term memory). ``user_id`` / ``session_id`` come from
``run_context.user_id`` / ``run_context.session_id`` (no longer carried in
the dependency object itself).
"""

from __future__ import annotations

from typing import Any

from agent_v3.checkout.service import CartService
from agent_v3.memory import Store

CART_SERVICE_KEY = "cart_service"
STORE_KEY = "store"
SKILLS_LOADED_KEY = "skills_loaded"


def build_dependencies(
    cart_service: CartService,
    store: Store,
    skills_loaded: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the per-turn dependency map handed to ``workflow.run``.

    ``skills_loaded`` should be the SAME list object as
    ``session_state["skills_loaded"]`` so the gating hook's appends are
    visible both inside the run and back in the persisted session state.
    """
    return {
        CART_SERVICE_KEY: cart_service,
        STORE_KEY: store,
        SKILLS_LOADED_KEY: skills_loaded if skills_loaded is not None else [],
    }


def get_cart_service(run_context: Any) -> CartService:
    """Read the live CartService out of a RunContext (raises if missing)."""
    return run_context.dependencies[CART_SERVICE_KEY]


def get_store(run_context: Any) -> Store | None:
    """Read the long-term store out of a RunContext, if present."""
    deps = getattr(run_context, "dependencies", None) or {}
    return deps.get(STORE_KEY)
