"""``ShoppingContext`` — the per-turn context carrying the live cart.

Extends the generic :class:`lg_agent.core.context.TurnContext` with a
``cart_service`` handle. The same instance flows orchestrator → sub-agent tool →
sub-agent invoke, so every tool mutates one shared cart within a turn.

The tools bind to this class directly (they read ``runtime.context.cart_service``
and ``runtime.context.user_id``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lg_agent.core.context import TurnContext
from lg_agent.shopping.domain import CartService


@dataclass
class ShoppingContext(TurnContext):
    cart_service: CartService = field(default_factory=CartService)

    def debug_view(self) -> dict[str, Any]:
        view = super().debug_view()
        cart = self.cart_service.cart
        view["cart"] = {
            "step": cart.step.value,
            "items": [
                {"id": i.product_id, "name": i.name, "qty": i.quantity, "unit_price": str(i.unit_price)}
                for i in cart.items
            ],
            "subtotal": str(cart.subtotal),
            "confirmed": cart.confirmed,
            "blockers": [b.code for b in cart.blockers()],
        }
        return view

    def routing_context(self) -> dict[str, str]:
        """The live cart, as a block the orchestrator can resolve references against."""
        cart = self.cart_service.cart
        if not cart.items:
            return {}
        items = "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
        return {"cart": f"Current cart: {items}"}


__all__ = ["ShoppingContext"]
