"""The mock e-commerce domain — cart flow, blockers, freshness, pricing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent_agno_v1.shopping.domain import CartError, CartService, build_store, get_order
from agent_agno_v1.shopping.domain.memory import recent_orders, remember_order


def _full_checkout() -> CartService:
    s = CartService()
    s.add_item("P-2")
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 Main", "San Francisco", "94110")
    s.lookup_serviceability()
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")
    return s


def test_happy_path_confirms():
    s = _full_checkout()
    assert s.cart.ready_to_confirm()
    msg = s.confirm()
    assert s.cart.confirmed and s.cart.receipt_id and "RCPT-" in msg


def test_empty_cart_blocks():
    s = CartService()
    codes = {b.code for b in s.cart.blockers()}
    assert "empty_cart" in codes
    assert s.cart.step.value == "collecting_products"


def test_cart_edit_invalidates_pricing():
    s = _full_checkout()
    assert s.cart.ready_to_confirm()
    s.add_item("P-1")  # mutate items → shipping/tax stale
    assert not s.cart.shipping_is_fresh()
    assert not s.cart.tax_is_fresh()
    assert s.cart.step.value == "awaiting_pricing"
    assert not s.cart.ready_to_confirm()


def test_address_change_invalidates_serviceability():
    s = _full_checkout()
    s.set_address("9 Other", "Paris", "75001", country="FR")
    assert s.cart.serviceable is None
    assert s.cart.delivery_option is None


def test_confirm_refuses_with_blockers():
    s = CartService()
    s.add_item("P-2")
    with pytest.raises(CartError):
        s.confirm()


def test_unknown_product_errors():
    s = CartService()
    with pytest.raises(CartError):
        s.add_item("P-999")


def test_serviceability_and_orders():
    assert get_order("ORD-7").status == "shipped"
    assert get_order("nope") is None


def test_memory_store_roundtrip():
    store = build_store()
    remember_order(store, "u1", {"receipt_id": "RCPT-1", "items": [], "total": "10"})
    orders = recent_orders(store, "u1")
    assert orders and orders[0]["receipt_id"] == "RCPT-1"


def test_promo_requires_qualifying_items():
    s = CartService()
    s.add_item("P-2")  # apparel, no shoes
    with pytest.raises(CartError):
        s.apply_promo("SHOES20")
    s.add_item("P-3")  # shoes
    assert "SHOES20" in s.apply_promo("SHOES20")
    assert s.cart.promo.discount == Decimal("17.80")
