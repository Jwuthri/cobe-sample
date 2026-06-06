"""Freshness / invalidation rules — the cart blows away derived quotes
when their inputs change."""

from __future__ import annotations

import pytest
from agent_v2.checkout import CartService
from agent_v2.checkout.service import CartError


def _ready_cart() -> CartService:
    s = CartService()
    s.add_item("P-3")  # sneakers, tag="shoes"
    s.set_customer("A", "B")
    s.set_address("x", "y", "94110")
    s.lookup_serviceability()
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("card", card_token="tok_z")
    return s


def test_adding_item_invalidates_shipping_and_tax():
    s = _ready_cart()
    assert s.cart.shipping_is_fresh()
    assert s.cart.tax_is_fresh()
    s.add_item("P-4")
    assert not s.cart.shipping_is_fresh()
    assert not s.cart.tax_is_fresh()
    assert any(b.code == "stale_shipping" for b in s.cart.blockers())
    assert any(b.code == "stale_tax" for b in s.cart.blockers())


def test_changing_zip_invalidates_serviceability_shipping_tax_and_clears_delivery():
    s = _ready_cart()
    assert s.cart.serviceable is True
    s.set_address("new st", "Paris", "75001")
    assert s.cart.serviceable is None  # cleared, must re-lookup
    assert not s.cart.shipping_is_fresh()
    assert not s.cart.tax_is_fresh()
    assert s.cart.delivery_option is None  # cleared since previous option may not apply


def test_changing_delivery_option_invalidates_shipping_only():
    s = _ready_cart()
    assert s.cart.shipping_is_fresh()
    s.set_delivery_option("4h")
    assert not s.cart.shipping_is_fresh()
    assert s.cart.tax_is_fresh()  # tax depends on subtotal + zip, not on delivery option


def test_promo_auto_invalidates_when_qualifying_item_removed():
    s = CartService()
    s.add_item("P-3")  # SHOES20-eligible
    s.add_item("P-1")
    s.set_customer("A", "B")
    s.set_address("x", "y", "94110")
    s.lookup_serviceability()
    s.apply_promo("SHOES20")
    assert s.cart.promo is not None
    s.remove_item("P-3")
    # Promo no longer applies — service should have cleared it.
    assert s.cart.promo is None


def test_apply_promo_without_qualifying_items_fails():
    s = CartService()
    s.add_item("P-1")  # no shoes
    with pytest.raises(CartError):
        s.apply_promo("SHOES20")
