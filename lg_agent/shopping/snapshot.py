"""Build the ``AgentSnapshot`` the frontend renders (cart panel + transcript).

This is the exact JSON shape the web client's ``getState`` / ``state`` events
expect. Kept out of the session so the streaming pipeline reads cleanly; it is a
pure projection of (cart, messages) → dict.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from langchain_core.messages import BaseMessage

from lg_agent.shopping.domain import Cart


def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def _cart_view(cart: Cart) -> dict[str, Any]:
    return {
        "step": cart.step.value,
        "cart_id": cart.cart_id,
        "items": [
            {
                "id": i.product_id,
                "name": i.name,
                "qty": i.quantity,
                "unit_price": _d(i.unit_price),
                "line_total": _d(i.line_total),
                "tags": list(i.tags),
            }
            for i in cart.items
        ],
        "customer": cart.customer.model_dump(),
        "address": cart.address.model_dump(),
        "serviceable": cart.serviceable,
        "serviceable_options": list(cart.serviceable_options),
        "delivery_option": cart.delivery_option,
        "shipping": (
            {"cost": _d(cart.shipping.cost), "eta_hours": cart.shipping.eta_hours}
            if cart.shipping_is_fresh()
            else None
        ),
        "tax": (
            {"amount": _d(cart.tax.amount), "rate": _d(cart.tax.rate)}
            if cart.tax_is_fresh()
            else None
        ),
        "promo": (
            {"code": cart.promo.code, "discount": _d(cart.promo.discount)} if cart.promo else None
        ),
        "payment_method": cart.payment_method,
        "card_token_set": bool(cart.card_token),
        "subtotal": _d(cart.subtotal),
        "grand_total": _d(cart.grand_total) if cart.grand_total is not None else None,
        "blockers": [{"code": b.code, "message": b.message} for b in cart.blockers()],
        "ready_to_confirm": cart.ready_to_confirm(),
        "confirmed": cart.confirmed,
        "receipt_id": cart.receipt_id,
    }


def build_snapshot(
    *,
    user_id: str,
    session_id: str,
    cart: Cart,
    messages: list[BaseMessage],
    skills_loaded: list[str],
) -> dict[str, Any]:
    """Project the session state into the frontend's ``AgentSnapshot`` shape."""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "active_sop": None,
        "skills_loaded": list(skills_loaded),
        "cart": _cart_view(cart),
        "messages": [
            {
                "role": getattr(m, "type", "?"),
                "content": str(m.content),
                "blocks": (getattr(m, "additional_kwargs", {}) or {}).get("blocks", []),
            }
            for m in messages
        ],
        "iteration": 0,
        "done": True,
    }


__all__ = ["build_snapshot"]
