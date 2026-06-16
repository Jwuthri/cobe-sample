"""``ShoppingContext`` — the shared state every agent and tool sees.

The OpenAI Agents SDK injects this into each tool as
``wrapper: RunContextWrapper[ShoppingContext]`` (then ``wrapper.context.cart_service``).
The SAME instance flows from the orchestrator into every delegated sub-agent run
(``as_tool()`` passes the parent's context wrapper through), so one live
:class:`~agent_openai_sdk_v1.domain.CartService` is mutated in place across the
whole turn — there is no copy-back step.

It also carries the turn's accumulated :class:`StepResult` list (workers append to
it via their ``custom_output_extractor``) and the orchestrator's cross-turn
``routing_notes`` — used to resolve references like "the green one" without
sub-agents ever seeing the chat. There is no event bus here: the SDK's
``Runner.run_streamed().stream_events()`` IS the bus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_openai_sdk_v1.domain import CartService, MemoryStore
from agent_openai_sdk_v1.runtime.step import StepResult


@dataclass
class ShoppingContext:
    cart_service: CartService
    store: MemoryStore
    user_id: str = "demo"

    # turn state
    steps: list[StepResult] = field(default_factory=list)
    routing_notes: dict[str, str] = field(default_factory=dict)
    debug: bool = True

    # The worker extractors push UI events for INNER tool calls onto this list
    # so the session can drain them inline with the orchestrator's outer stream
    # (without spinning up a background task or an asyncio.Queue). The events are
    # already in the wire vocabulary (see :mod:`runtime.events`).
    pending_events: list[dict] = field(default_factory=list)

    # ----- reference resolution surface -----
    def routing_context(self) -> dict[str, str]:
        """Live structured state the orchestrator resolves references + routing against.

        Two things: the current cart (to resolve "the green one") and, when a checkout
        is mid-flight, the exact step it is on — so a terse reply like "2h" or "cash"
        is recognized as checkout data instead of being mistaken for smalltalk.
        Returns ``{}`` when the cart is empty so the memo stays quiet.
        """
        from agent_openai_sdk_v1.domain import CheckoutStep

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
