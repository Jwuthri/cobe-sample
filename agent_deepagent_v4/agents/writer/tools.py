"""Writer tool: read the authoritative cart state for composing the reply.

The writer is told the turn's *facts* (products found, order status, what
changed) in the task prompt the orchestrator hands it. For anything financial
or safety-related — totals, blockers, and especially whether the order is
actually placed — it reads the live cart here so those values are verbatim and
never invented.
"""

from __future__ import annotations

import json

from langchain.tools import ToolRuntime, tool

from agent_deepagent_v4.context import ShopContext

# Blockers the customer can actually act on (mirrors what the reply should ask for).
_USER_ACTIONABLE = {
    "empty_cart",
    "missing_identity",
    "missing_address",
    "not_serviceable",
    "missing_delivery_option",
    "unserviceable_delivery_option",
    "missing_payment",
    "missing_card_token",
    "invalid_promo",
}


@tool
def read_cart(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Return the authoritative cart snapshot as JSON.

    Use this for totals, what's in the cart, the next thing the customer must
    provide (``actionable_blockers``), and especially ``confirmed`` / ``receipt_id``
    — only treat the order as placed when ``confirmed`` is true.
    """
    c = runtime.context.cart_service.cart
    snapshot = {
        "step": c.step.value,
        "items": [
            {"id": i.product_id, "name": i.name, "qty": i.quantity, "unit_price": str(i.unit_price)}
            for i in c.items
        ],
        "subtotal": str(c.subtotal),
        "customer": c.customer.model_dump(),
        "address": c.address.model_dump(),
        "serviceable": c.serviceable,
        "serviceable_options": list(c.serviceable_options),
        "delivery_option": c.delivery_option,
        "shipping": (
            {"cost": str(c.shipping.cost), "eta_hours": c.shipping.eta_hours} if c.shipping_is_fresh() else None
        ),
        "tax": {"amount": str(c.tax.amount)} if c.tax_is_fresh() else None,
        "promo": {"code": c.promo.code, "discount": str(c.promo.discount)} if c.promo else None,
        "payment_method": c.payment_method,
        "card_token_set": bool(c.card_token),
        "grand_total": str(c.grand_total) if c.grand_total is not None else None,
        "actionable_blockers": [
            {"code": b.code, "message": b.message} for b in c.blockers() if b.code in _USER_ACTIONABLE
        ],
        "ready_to_confirm": c.ready_to_confirm(),
        "confirmed": c.confirmed,
        "receipt_id": c.receipt_id,
    }
    return json.dumps(snapshot, ensure_ascii=False, indent=2)


WRITER_TOOLS = [read_cart]
