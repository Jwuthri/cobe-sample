"""Regression: editing the cart after pricing was computed must force a
re-pricing step, not leave checkout silently stuck at ready_to_confirm.

Reproduces the trace where a user lowered an item's quantity at
ready_to_confirm; that invalidated shipping + tax, ``confirm_checkout``
failed on stale blockers, and nothing ever recomputed — the writer then
had no grand_total and looped on "I'm missing the final checkout total".
"""

from __future__ import annotations

from agent_v4.checkout import CartService
from agent_v4.checkout.cart import CheckoutStep
from agent_v4.leaves import checkout_anchor


def _ready_cart() -> CartService:
    """A cart with every field collected and fresh shipping + tax."""
    s = CartService()
    s.add_item("P-1", quantity=2)
    s.set_customer("Julien", "Wuthrich")
    s.set_address("1717 Webster", "San Francisco", zip_code="9412")
    s.lookup_serviceability()
    s.set_delivery_option("standard")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")
    return s


def test_ready_cart_is_ready_to_confirm():
    s = _ready_cart()
    assert s.cart.step is CheckoutStep.READY_TO_CONFIRM
    assert s.cart.ready_to_confirm()
    assert s.cart.grand_total is not None


def test_quantity_edit_moves_step_to_awaiting_pricing():
    s = _ready_cart()
    total_before = s.cart.grand_total

    # User: "actually remove one t-shirt" — invalidates shipping + tax.
    s.set_quantity("P-1", 1)

    # The step machine must NOT report ready_to_confirm with stale pricing.
    assert s.cart.step is CheckoutStep.AWAITING_PRICING
    assert not s.cart.ready_to_confirm()
    assert s.cart.grand_total is None
    codes = {b.code for b in s.cart.blockers()}
    assert {"stale_shipping", "stale_tax"} <= codes

    # Recomputing pricing (what the anchor now instructs the subagent to do)
    # clears the blockers and yields a fresh, lower total.
    s.quote_shipping()
    s.compute_tax()
    assert s.cart.step is CheckoutStep.READY_TO_CONFIRM
    assert s.cart.ready_to_confirm()
    assert s.cart.grand_total is not None
    assert s.cart.grand_total < total_before


def test_anchor_instructs_recompute_when_pricing_is_stale():
    s = _ready_cart()
    s.set_quantity("P-1", 1)

    anchor = checkout_anchor(s.cart)
    # The model must be told to recompute itself, not to confirm or stall.
    assert "quote_shipping()" in anchor
    assert "compute_tax()" in anchor
    assert "STALE" in anchor
