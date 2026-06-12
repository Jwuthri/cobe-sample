"""The *skill* abstraction — stateful instruction injected at run time.

A :class:`Skill` is a named instruction provider: ``render(cart) -> str`` turns
the live domain state into a transient instruction block. Unlike a static prompt
(the agent's fixed voice) or a tool (an action), a skill is *derived from state*
and re-rendered on every run.

The one skill here is ``CHECKOUT_PROGRESS``: the deterministic "Checkout progress"
anchor. The cart is the source of truth, so we render its state explicitly instead
of making the model rediscover it from a growing thread; ``cart.step`` drives the
single NEXT STEP. The builder wires a declared skill into the agent as an Agno
callable-``instructions`` function, so Agno re-evaluates it against the live cart
each run (the agent_v4_1 ``cart_anchor`` middleware, ported to Agno).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agno_agent_v1.domain.cart import Cart

# step.value -> the one-line "what to do next" hint shown in the anchor
_NEXT_STEP_HINT = {
    "collecting_products": "items missing — this shouldn't happen mid-checkout.",
    "collecting_identity": "identity — capture the customer's name with set_customer.",
    "collecting_address": "address — capture the shipping address with set_address.",
    "awaiting_serviceability": "serviceability — call lookup_serviceability().",
    "collecting_delivery": "delivery — set_delivery_option the user chose, then quote_shipping() + compute_tax().",
    "collecting_payment": "payment — attach_payment with the user's method (card needs a token).",
    "awaiting_pricing": (
        "pricing — the cart changed, so the shipping quote and tax are stale. Recompute "
        "NOW yourself: call quote_shipping() then compute_tax(). Do NOT confirm yet — the "
        "refreshed total must be shown so the user can approve it."
    ),
    "ready_to_confirm": "ready — if the user's latest message is an explicit yes/confirm, call confirm_checkout(); otherwise do nothing.",
    "confirmed": "order already placed — do nothing.",
}


def checkout_anchor_text(cart: Cart) -> str:
    """Render the deterministic 'what's done / what's next' checkout block."""
    c = cart

    def mark(done: bool, value: str) -> str:
        return f"✓ {value}".rstrip() if done else "— not provided"

    name = f"{c.customer.first_name or ''} {c.customer.last_name or ''}".strip()
    identity = mark(bool(c.customer.first_name), name)
    address = mark(
        c.address.is_complete(),
        f"{c.address.street}, {c.address.city} {c.address.zip_code}",
    )
    if c.serviceable is True:
        serviceability = f"✓ ships here (options: {', '.join(c.serviceable_options)})"
    elif c.serviceable is False:
        serviceability = "✗ NOT serviceable — ask for a different address"
    else:
        serviceability = "— not checked"
    delivery = mark(bool(c.delivery_option), c.delivery_option or "")
    payment = mark(bool(c.payment_method), c.payment_method or "")
    if c.shipping_is_fresh() and c.tax_is_fresh():
        pricing = f"✓ shipping {c.shipping.cost} + tax {c.tax.amount} → total {c.grand_total}"
    elif c.delivery_option:
        pricing = "✗ STALE — cart changed; recompute with quote_shipping() then compute_tax()"
    else:
        pricing = "— not computed"

    return (
        "Checkout progress (authoritative — never redo a ✓ field):\n"
        f"  identity:       {identity}\n"
        f"  address:        {address}\n"
        f"  serviceability: {serviceability}\n"
        f"  delivery:       {delivery}\n"
        f"  payment:        {payment}\n"
        f"  pricing:        {pricing}\n"
        f"Resume from: {_NEXT_STEP_HINT.get(c.step.value, 'the next missing field.')}\n"
        "Advance using the user's latest message + automatic internal steps; stop "
        "at the first field that needs info the user hasn't given."
    )


@dataclass(frozen=True)
class Skill:
    """A named, state-derived instruction block injected on each run."""

    name: str
    description: str
    render: Callable[[Cart], str]


CHECKOUT_PROGRESS = Skill(
    name="checkout_progress",
    description="Injects the live 'Checkout progress' anchor (cart step + captured fields).",
    render=checkout_anchor_text,
)

__all__ = ["Skill", "CHECKOUT_PROGRESS", "checkout_anchor_text"]
