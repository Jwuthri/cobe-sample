"""Domain smoke — invalidation chain, confirmation gate, deterministic ids."""

from __future__ import annotations

from decimal import Decimal

from agent_v4_1.shopping.domain import CartService


def test_deterministic_ids():
    cs = CartService()
    assert cs.cart.cart_id == "CART-1000"


def test_add_and_subtotal():
    cs = CartService()
    cs.add_item("P-2", 2)  # 49.99 each
    assert cs.cart.subtotal == Decimal("99.98")
    assert cs.cart.step.value == "collecting_identity"


def test_zip_change_invalidates_quotes_and_serviceability():
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("A", "B")
    cs.set_address("1 Market", "San Francisco", "94110")
    cs.lookup_serviceability()
    cs.set_delivery_option("standard")
    cs.quote_shipping()
    cs.compute_tax()
    assert cs.cart.shipping_is_fresh() and cs.cart.tax_is_fresh()
    # changing the zip wipes serviceability + quotes
    cs.set_address("9 Rue", "Paris", "75001", country="FR")
    assert cs.cart.serviceable is None
    assert cs.cart.shipping is None and cs.cart.tax is None
    assert cs.cart.delivery_option is None


def test_full_checkout_confirms_with_receipt():
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("Ada", "Lovelace")
    cs.set_address("1 Market", "San Francisco", "94110")
    cs.lookup_serviceability()
    cs.set_delivery_option("standard")
    cs.quote_shipping()
    cs.compute_tax()
    cs.attach_payment("cash")
    assert cs.cart.ready_to_confirm()
    msg = cs.confirm()
    assert cs.cart.confirmed and cs.cart.receipt_id == "RCPT-9000"
    assert "RCPT-9000" in msg


def test_confirm_blocked_when_incomplete():
    cs = CartService()
    cs.add_item("P-1")
    assert not cs.cart.ready_to_confirm()
    assert {b.code for b in cs.cart.blockers()} >= {"missing_identity", "missing_address"}
