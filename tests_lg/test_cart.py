"""Domain smoke — invalidation chain, confirmation gate, deterministic ids."""

from __future__ import annotations

from decimal import Decimal

import pytest

from lg_agent.shopping.domain import CartService
from lg_agent.shopping.domain.cart_service import CartError


def test_deterministic_ids():
    cs = CartService()
    assert cs.cart.cart_id == "CART-1000"


def test_set_customer_rejects_confabulated_names():
    cs = CartService()
    # field labels / addresses the LLM might mistakenly pass as a name
    for first, last in [("Shipping", "address"), ("shipping", "ADDRESS"), ("1717", "webst"), ("Delivery", "method")]:
        with pytest.raises(CartError):
            cs.set_customer(first, last)
    assert cs.cart.customer.first_name is None  # nothing was written

    # real names still go through (incl. hyphen/apostrophe)
    assert "Customer set" in cs.set_customer("Jean-Luc", "O'Brien")
    assert cs.cart.customer.first_name == "Jean-Luc"


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


def test_unserviceable_address_routes_back_to_collecting_address():
    # a complete-but-unserviceable address must NOT advance to delivery — it routes
    # back to collecting_address so the user can fix it (set_address stays reachable).
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("Ada", "Lovelace")
    cs.set_address("1 Main", "Nowhere", "99999")  # not in the serviceability table
    cs.lookup_serviceability()
    assert cs.cart.serviceable is False
    assert cs.cart.step.value == "collecting_address"  # NOT collecting_delivery
    # correcting to a serviceable zip re-opens serviceability, then advances
    cs.set_address("1 Main", "San Francisco", "94110")
    assert cs.cart.serviceable is None and cs.cart.step.value == "awaiting_serviceability"
    cs.lookup_serviceability()
    assert cs.cart.serviceable is True and cs.cart.step.value == "collecting_delivery"


def test_confirm_blocked_when_incomplete():
    cs = CartService()
    cs.add_item("P-1")
    assert not cs.cart.ready_to_confirm()
    assert {b.code for b in cs.cart.blockers()} >= {"missing_identity", "missing_address"}
