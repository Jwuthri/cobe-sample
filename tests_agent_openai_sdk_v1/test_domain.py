"""Domain tests — the cart step machine, blockers, and freshness invalidation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent_openai_sdk_v1.domain import CartService, CheckoutStep
from agent_openai_sdk_v1.domain.cart_service import CartError


def _walk_to(service: CartService, target: CheckoutStep) -> None:
    """Walk a cart forward through the step machine up to (but not past) ``target``."""
    if target == CheckoutStep.COLLECTING_PRODUCTS:
        return
    service.add_item("P-2", 1)  # Black Hoodie ($49.99)
    if target == CheckoutStep.COLLECTING_IDENTITY:
        return
    service.set_customer("Ada", "Lovelace")
    if target == CheckoutStep.COLLECTING_ADDRESS:
        return
    service.set_address("1 Market St", "San Francisco", "94105")
    if target == CheckoutStep.AWAITING_SERVICEABILITY:
        return
    service.lookup_serviceability()
    if target == CheckoutStep.COLLECTING_DELIVERY:
        return
    service.set_delivery_option("2h")
    if target == CheckoutStep.COLLECTING_PAYMENT:
        return
    service.quote_shipping()
    service.compute_tax()
    service.attach_payment("cash")
    if target == CheckoutStep.READY_TO_CONFIRM:
        return
    service.confirm()


def test_step_machine_walks_through_all_states():
    s = CartService()
    assert s.cart.step == CheckoutStep.COLLECTING_PRODUCTS
    s.add_item("P-2", 1)
    assert s.cart.step == CheckoutStep.COLLECTING_IDENTITY
    s.set_customer("Ada", "Lovelace")
    assert s.cart.step == CheckoutStep.COLLECTING_ADDRESS
    s.set_address("1 Market St", "San Francisco", "94105")
    assert s.cart.step == CheckoutStep.AWAITING_SERVICEABILITY
    s.lookup_serviceability()
    assert s.cart.step == CheckoutStep.COLLECTING_DELIVERY
    s.set_delivery_option("2h")
    assert s.cart.step == CheckoutStep.COLLECTING_PAYMENT
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")
    assert s.cart.step == CheckoutStep.READY_TO_CONFIRM
    s.confirm()
    assert s.cart.step == CheckoutStep.CONFIRMED
    assert s.cart.confirmed
    assert s.cart.receipt_id == "RCPT-9000"


def test_blockers_gate_confirmation():
    s = CartService()
    assert any(b.code == "empty_cart" for b in s.cart.blockers())
    with pytest.raises(CartError):
        s.confirm()
    s.add_item("P-2", 1)
    assert any(b.code == "missing_identity" for b in s.cart.blockers())


def test_quantity_change_invalidates_shipping_and_tax():
    s = CartService()
    _walk_to(s, CheckoutStep.READY_TO_CONFIRM)
    assert s.cart.shipping_is_fresh()
    assert s.cart.tax_is_fresh()

    s.set_quantity("P-2", 2)
    assert not s.cart.shipping_is_fresh(), "qty change should invalidate shipping"
    assert not s.cart.tax_is_fresh(), "qty change should invalidate tax"
    assert s.cart.step == CheckoutStep.AWAITING_PRICING


def test_address_change_invalidates_serviceability_and_quotes():
    s = CartService()
    _walk_to(s, CheckoutStep.COLLECTING_DELIVERY)
    s.set_address("99 Rivoli St", "Paris", "75001", country="FR")
    assert s.cart.serviceable is None
    assert s.cart.delivery_option is None
    assert s.cart.shipping is None
    assert s.cart.tax is None


def test_unserviceable_zip_routes_back_to_address():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    # 00001 has no serviceability row → not serviceable.
    s.set_address("1 Nowhere", "Nowhere", "00001")
    s.lookup_serviceability()
    assert s.cart.serviceable is False
    assert s.cart.step == CheckoutStep.COLLECTING_ADDRESS


def test_payment_card_needs_token():
    s = CartService()
    _walk_to(s, CheckoutStep.COLLECTING_PAYMENT)
    s.attach_payment("card")  # no token
    assert any(b.code == "missing_card_token" for b in s.cart.blockers())
    s.attach_payment("card", card_token="tok_test")
    s.quote_shipping()
    s.compute_tax()
    assert s.cart.ready_to_confirm()


def test_promo_applies_and_revalidates_on_item_change():
    s = CartService()
    s.add_item("P-3", 1)  # shoes, $89.00
    s.apply_promo("SHOES20")
    assert s.cart.promo is not None
    s.remove_item("P-3")
    # removing the only matching item invalidates the promo
    assert s.cart.promo is None


def test_set_customer_rejects_field_labels():
    s = CartService()
    with pytest.raises(CartError):
        s.set_customer("Shipping", "Address")


def test_grand_total_includes_shipping_and_tax():
    s = CartService()
    _walk_to(s, CheckoutStep.READY_TO_CONFIRM)
    # subtotal 49.99 + shipping 19.99 + tax (8.75% of 49.99 ≈ 4.37) = 74.35
    assert s.cart.grand_total == Decimal("49.99") + Decimal("19.99") + Decimal("4.37")
