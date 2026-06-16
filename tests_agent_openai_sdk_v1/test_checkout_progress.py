"""Checkout-progress block + asks-for-step tests.

These cover the dynamic instructions block injected into every checkout run — the
authoritative "what's done / what's next" view that lets the model trust the cart
instead of re-deriving state from the transcript.
"""

from __future__ import annotations

from agent_openai_sdk_v1.agents.checkout import asks_for_step, checkout_progress
from agent_openai_sdk_v1.domain import CartService, CheckoutStep


def test_progress_block_marks_done_fields():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 Market St", "San Francisco", "94105")
    s.lookup_serviceability()
    block = checkout_progress(s.cart)
    assert "✓ Ada Lovelace" in block
    assert "✓ 1 Market St, San Francisco 94105" in block
    assert "✓ ships here" in block
    assert "— not provided" in block  # delivery + payment still missing


def test_progress_block_flags_stale_pricing_after_quantity_change():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 Market St", "San Francisco", "94105")
    s.lookup_serviceability()
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")  # past payment, so AWAITING_PRICING kicks in on edit
    assert s.cart.shipping_is_fresh()
    s.set_quantity("P-2", 2)
    block = checkout_progress(s.cart)
    assert "STALE" in block
    assert "Resume from: pricing" in block


def test_asks_for_step_matches_cart_state():
    s = CartService()
    s.add_item("P-2", 1)
    assert asks_for_step(s.cart.step.value, s.cart) == ["first name", "last name"]
    s.set_customer("Ada", "Lovelace")
    assert "street" in asks_for_step(s.cart.step.value, s.cart)


def test_asks_for_unserviceable_address():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 Nowhere", "Nowhere", "00001")
    s.lookup_serviceability()
    # step machine routes back to address with a "different, serviceable" ask
    assert s.cart.step == CheckoutStep.COLLECTING_ADDRESS
    asks = asks_for_step(s.cart.step.value, s.cart)
    assert any("different" in a for a in asks)
