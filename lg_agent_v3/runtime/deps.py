"""``ShoppingDeps`` — the shared state every agent and tool sees.

This is the LangChain ``context`` (``context_schema=ShoppingDeps``): one instance is
created per user turn and passed to ``orchestrator.ainvoke(..., context=deps)``.
LangChain forwards the SAME object to each tool's ``ToolRuntime`` and to every nested
sub-agent run (``worker.agent.ainvoke(..., context=deps)``), so one live
:class:`~lg_agent_v3.domain.CartService` is mutated in place across the whole turn —
there is no copy-back step. (It is the structural analogue of Pydantic AI's ``deps``.)

It also carries the turn's plumbing:

* ``emit`` — tools/nodes/guardrails raise a UI event via ``deps.emit(...)``, which
  forwards to LangGraph's ``get_stream_writer()``; the session's
  ``astream(subgraphs=True)`` loop receives it from any depth (no bus).
* ``steps`` — the :class:`StepResult` list workers append to as they run;
* ``routing_notes`` — the orchestrator's cross-turn memory (persisted by the session)
  used to resolve references like "the green one" without sub-agents seeing the chat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lg_agent_v3.domain import CartService, MemoryStore
from lg_agent_v3.runtime.step import StepResult


@dataclass
class ShoppingDeps:
    cart_service: CartService
    store: MemoryStore
    user_id: str = "demo"
    session_id: str = "sess-lgv3"
    # the live transcript (shared by reference with the session) — the writer payload +
    # snapshots read it; workers never see it (context isolation).
    transcript: list = field(default_factory=list)

    # turn plumbing
    steps: list[StepResult] = field(default_factory=list)
    routing_notes: dict[str, str] = field(default_factory=dict)
    debug: bool = True

    # guardrails: middleware append a GuardrailHit here when a rule fires; the session
    # reads orchestrator-level hits (→ refusal), run_subagent reads worker-level ones
    # (→ a flagged guardrail step). ``refusal`` is set when an orchestrator block routes
    # the turn straight to the writer.
    guardrail_hits: list = field(default_factory=list)
    refusal: str | None = None

    # ----- UI events -----
    def emit(self, event: dict) -> None:
        """Emit a UI event on the turn-graph's custom stream.

        Uses LangGraph's ``get_stream_writer()`` so an event raised anywhere — a node, a
        guardrail middleware, or a tool nested inside a sub-agent — reaches the session's
        ``astream(..., subgraphs=True)`` loop. A no-op outside a streaming context (e.g. a
        test that invokes an agent directly), so tools/guardrails stay callable offline.
        """
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:  # noqa: BLE001  (no active stream → drop the event)
            writer = None
        if writer is not None:
            writer(event)

    # ----- reference resolution surface -----
    def routing_context(self) -> dict[str, str]:
        """Live structured state the orchestrator resolves references + routing against.

        Two things: the current cart (to resolve "the green one") and, when a checkout
        is mid-flight, the exact step it is on — so a terse reply like "2h" or "cash"
        is recognized as checkout data instead of being mistaken for smalltalk.
        Returns ``{}`` when the cart is empty so the memo stays quiet.
        """
        from lg_agent_v3.domain import CheckoutStep

        cart = self.cart_service.cart
        if not cart.items:
            return {}
        items = "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
        out = {"cart": f"Current cart: {items}"}
        if not cart.confirmed and cart.step not in (CheckoutStep.COLLECTING_PRODUCTS, CheckoutStep.CONFIRMED):
            out["checkout"] = (
                f"A checkout is in progress for this cart (current step: {cart.step.value}). "
                "If the user's latest message provides what that step needs (a name, an "
                "address, a delivery option, a payment method, a promo code, or a "
                "yes/no), it is checkout data → route to checkout, not smalltalk."
            )
        return out

    # ----- debug -----
    def debug_view(self) -> dict[str, Any]:
        cart = self.cart_service.cart
        return {
            "user_id": self.user_id,
            "steps": [s.sop for s in self.steps],
            "cart": {
                "step": cart.step.value,
                "items": [
                    {"id": i.product_id, "name": i.name, "qty": i.quantity, "unit_price": str(i.unit_price)}
                    for i in cart.items
                ],
                "subtotal": str(cart.subtotal),
                "confirmed": cart.confirmed,
                "blockers": [b.code for b in cart.blockers()],
            },
        }
