"""``ShoppingContext`` — the per-turn context carrying the live cart.

Extends the generic :class:`agent_v4_1.core.context.TurnContext` with a
``cart_service`` handle. The same instance flows orchestrator → sub-agent tool →
sub-agent invoke, so every tool mutates one shared cart within a turn.

Field names are a superset of what the shopping tools read
(``runtime.context.cart_service`` / ``runtime.context.user_id``), so the tools
bind to this class directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_v4_1.core.context import TurnContext
from agent_v4_1.shopping.domain import CartService


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
