"""Domain logic — the cart state machine, freshness, blockers, confirmation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from openai_agent_v1.shopping.domain import CartService
from openai_agent_v1.shopping.domain.cart_service import CartError


def _ready_cart() -> CartService:
    svc = CartService()
    svc.add_item("P-2", 1)
    svc.set_customer("Ada", "Lovelace")
    svc.set_address("1 Main", "San Francisco", "94110")
    svc.lookup_serviceability()
    svc.set_delivery_option("2h")
    svc.quote_shipping()
    svc.compute_tax()
    svc.attach_payment("cash")
    return svc


def test_step_progression():
    svc = CartService()
    assert svc.cart.step.value == "collecting_products"
    svc.add_item("P-2", 1)
    assert svc.cart.step.value == "collecting_identity"
    svc.set_customer("Ada", "Lovelace")
    assert svc.cart.step.value == "collecting_address"
    svc.set_address("1 Main", "San Francisco", "94110")
    assert svc.cart.step.value == "awaiting_serviceability"


def test_add_item_merges_quantity():
    svc = CartService()
    svc.add_item("P-2", 1)
    svc.add_item("P-2", 2)
    assert len(svc.cart.items) == 1
    assert svc.cart.items[0].quantity == 3


def test_unknown_product_rejected():
    with pytest.raises(CartError):
        CartService().add_item("P-999")


def test_ready_cart_confirms():
    svc = _ready_cart()
    assert svc.cart.ready_to_confirm() is True
    assert svc.cart.grand_total is not None
    result = svc.confirm()
    assert "RCPT-9000" in result
    assert svc.cart.confirmed is True
    assert svc.cart.step.value == "confirmed"


def test_cart_edit_invalidates_pricing():
    svc = _ready_cart()
    assert svc.cart.ready_to_confirm() is True
    # A quantity change makes shipping + tax stale → no longer ready, own step.
    svc.set_quantity("P-2", 3)
    assert svc.cart.shipping_is_fresh() is False
    assert svc.cart.tax_is_fresh() is False
    assert svc.cart.step.value == "awaiting_pricing"
    assert svc.cart.ready_to_confirm() is False


def test_unserviceable_zip_blocks():
    svc = CartService()
    svc.add_item("P-2", 1)
    svc.set_customer("Ada", "Lovelace")
    svc.set_address("1 Broadway", "Oakland", "94607")  # 946 not in serviceability table
    out = svc.lookup_serviceability()
    assert "not serviceable" in out.lower()
    assert svc.cart.serviceable is False
    codes = {b.code for b in svc.cart.blockers()}
    assert "not_serviceable" in codes


def test_confirm_blocked_when_incomplete():
    svc = CartService()
    svc.add_item("P-2", 1)
    with pytest.raises(CartError):
        svc.confirm()


def test_set_customer_rejects_address_as_name():
    svc = CartService()
    svc.add_item("P-2", 1)
    with pytest.raises(CartError):
        svc.set_customer("1717 Webster", "San Francisco")  # digits → not a name
    with pytest.raises(CartError):
        svc.set_customer("Shipping", "address")  # all label words → not a name


def test_card_requires_token():
    svc = _ready_cart()
    svc.attach_payment("card")  # no token
    codes = {b.code for b in svc.cart.blockers()}
    assert "missing_card_token" in codes
    assert svc.cart.ready_to_confirm() is False
    svc.attach_payment("card", card_token="tok_visa_4242")
    assert svc.cart.ready_to_confirm() is True


def test_promo_shoes_requires_shoes():
    svc = CartService()
    svc.add_item("P-2", 1)  # hoodie, not shoes
    with pytest.raises(CartError):
        svc.apply_promo("SHOES20")
    svc.add_item("P-3", 1)  # sneakers
    out = svc.apply_promo("SHOES20")
    assert "SHOES20" in out
    assert svc.cart.promo.discount == Decimal("17.80")  # 20% of 89.00
