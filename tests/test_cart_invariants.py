"""Pure-state tests for the cart's step machine and blockers."""

from __future__ import annotations

from agent_v2.checkout import CartService
from agent_v2.checkout.cart import CheckoutStep


def test_empty_cart_is_collecting_products():
    s = CartService()
    assert s.cart.step is CheckoutStep.COLLECTING_PRODUCTS
    assert not s.cart.ready_to_confirm()
    assert any(b.code == "empty_cart" for b in s.cart.blockers())


def test_step_progresses_through_checkout():
    s = CartService()
    s.add_item("P-2")
    assert s.cart.step is CheckoutStep.COLLECTING_IDENTITY
    s.set_customer("Julien", "Doe")
    assert s.cart.step is CheckoutStep.COLLECTING_ADDRESS
    s.set_address("123 Market", "SF", "94110", state="CA")
    assert s.cart.step is CheckoutStep.AWAITING_SERVICEABILITY
    s.lookup_serviceability()
    assert s.cart.step is CheckoutStep.COLLECTING_DELIVERY
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    assert s.cart.step is CheckoutStep.COLLECTING_PAYMENT
    s.attach_payment("card", card_token="tok_42")
    assert s.cart.step is CheckoutStep.READY_TO_CONFIRM
    assert s.cart.ready_to_confirm()
    s.confirm()
    assert s.cart.step is CheckoutStep.CONFIRMED


def test_unserviceable_zip_blocks():
    s = CartService()
    s.add_item("P-1")
    s.set_customer("A", "B")
    s.set_address("x", "y", "99999")
    s.lookup_serviceability()
    assert s.cart.serviceable is False
    assert any(b.code == "not_serviceable" for b in s.cart.blockers())


def test_picking_unserviceable_delivery_option_errors():
    """Try to pick a delivery option that's NOT in serviceable_options."""
    import pytest
    from agent_v2.checkout.service import CartError

    s = CartService()
    s.add_item("P-1")
    s.set_customer("A", "B")
    s.set_address("x", "y", "75001")  # Paris zone — only next_day + standard
    s.lookup_serviceability()
    assert "2h" not in s.cart.serviceable_options
    with pytest.raises(CartError):
        s.set_delivery_option("2h")


def test_card_without_token_blocks():
    s = CartService()
    s.add_item("P-1")
    s.set_customer("A", "B")
    s.set_address("x", "y", "94110")
    s.lookup_serviceability()
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("card")  # no token
    assert any(b.code == "missing_card_token" for b in s.cart.blockers())
    # And confirm refuses.
    import pytest
    from agent_v2.checkout.service import CartError

    with pytest.raises(CartError):
        s.confirm()


def test_payment_switch_clears_token():
    s = CartService()
    s.add_item("P-1")
    s.attach_payment("card", card_token="tok_x")
    assert s.cart.card_token == "tok_x"
    s.attach_payment("cash")
    assert s.cart.card_token is None
