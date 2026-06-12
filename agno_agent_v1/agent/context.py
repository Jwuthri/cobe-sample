"""Per-turn runtime context + the structured result a sub-agent produces.

``ShoppingContext`` is the single shared state for one user turn. It rides Agno's
``dependencies`` dict (``dependencies={"ctx": ctx}``): the orchestrator passes it
into every sub-agent run, and every tool reads it back off
``run_context.dependencies["ctx"]``. Because the dict value is passed *by
reference*, one live ``CartService`` mutates in place and is visible end-to-end —
this is the Agno analogue of agent_v4_1's ``runtime.context``.

``StepResult`` is the bridge from a sub-agent's raw work to the deterministic
blocks + the writer: the orchestrator LLM only ever reads the terse ``summary``,
while the rich ``details`` feeds the (model-free) block builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agno_agent_v1.domain import CartService, MemoryStore, build_store


class StepResult(BaseModel):
    """One sub-agent's distilled outcome."""

    sop: str
    summary: str = ""
    asks: list[str] = Field(default_factory=list)
    next_sop: str | None = None
    cart_diff: dict | None = None
    details: dict | None = None
    # An LLM-ready snippet of facts this step surfaced that the orchestrator should
    # remember NEXT turn to resolve references ("the green one", "the second order").
    # Domain-rendered free text — the engine just carries it, agnostic to content.
    recall: str | None = None


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}


@dataclass
class ShoppingContext:
    """Static-per-turn config + the mutable accumulators (cart, results, events)."""

    user_id: str = "demo"
    session_id: str = "session"
    cart_service: CartService = field(default_factory=CartService)
    store: MemoryStore = field(default_factory=build_store)
    step_results: list[StepResult] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=_zero_usage)
    # Buffer the sub-agent wrappers fill while the orchestrator runs; the session
    # drains it into SSE events once the orchestrator turn completes.
    events: list[dict] = field(default_factory=list)
    debug: bool = False

    # ----- reference resolution (the orchestrator owns interpretation) -----
    def routing_context(self) -> dict[str, str]:
        """Live structured state the orchestrator can resolve references against.

        Returns ``{label: llm_ready_text}``. Domain-specific (the current cart);
        the session merges this with persisted per-step recalls into the routing
        memo. Empty cart → no block.
        """
        cart = self.cart_service.cart
        if not cart.items:
            return {}
        items = "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
        return {"cart": f"Current cart: {items}"}

    # ----- debug -----
    def debug_view(self) -> dict[str, Any]:
        cart = self.cart_service.cart
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "usage": dict(self.usage),
            "step_results": [r.model_dump(mode="json") for r in self.step_results],
            "cart": {
                "step": cart.step.value,
                "items": [
                    {"id": i.product_id, "name": i.name, "qty": i.quantity,
                     "unit_price": str(i.unit_price)}
                    for i in cart.items
                ],
                "subtotal": str(cart.subtotal),
                "confirmed": cart.confirmed,
                "blockers": [b.code for b in cart.blockers()],
            },
        }


__all__ = ["StepResult", "ShoppingContext"]
