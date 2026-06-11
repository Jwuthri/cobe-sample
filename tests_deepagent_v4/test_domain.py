"""Offline domain tests — no model calls. These pin the checkout SAFETY
invariants (the blocker gate, step progression, quote freshness) that the
agents rely on, independent of any LLM behavior."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent_deepagent_v4.domain import catalog
from agent_deepagent_v4.domain.cart import Cart, CheckoutStep
from agent_deepagent_v4.domain.pricing import quote_promo
from agent_deepagent_v4.domain.serviceability import lookup
from agent_deepagent_v4.domain.service import CartError, CartService


def _ready_cart() -> CartService:
    """A fully-prepared, serviceable, paid cart with fresh quotes."""
    svc = CartService(Cart())
    svc.add_item("P-2")  # Black Hoodie 49.99
    svc.set_customer("Ada", "Lovelace")
    svc.set_address("1 Market St", "San Francisco", "94105")
    svc.lookup_serviceability()
    svc.set_delivery_option("next_day")
    svc.quote_shipping()
    svc.compute_tax()
    svc.attach_payment("card", card_token="tok_42")
    return svc


# ---------- catalog / search ----------
def test_search_is_token_scoped_not_substring():
    # "ca" (e.g. from California) must NOT match "cap".
    assert catalog.search("ca") == [] or all("hat" not in p.tags for p in catalog.search("ca"))
    hoodies = catalog.search("hoodie")
    assert [p.id for p in hoodies] == ["P-2"]


def test_product_id_lookup_hyphen_insensitive():
    assert catalog.get("p2").id == "P-2"
    assert catalog.get("P-2").id == "P-2"


# ---------- step progression ----------
def test_step_progresses_in_order():
    svc = CartService(Cart())
    assert svc.cart.step == CheckoutStep.COLLECTING_PRODUCTS
    svc.add_item("P-2")
    assert svc.cart.step == CheckoutStep.COLLECTING_IDENTITY
    svc.set_customer("Ada", "Lovelace")
    assert svc.cart.step == CheckoutStep.COLLECTING_ADDRESS
    svc.set_address("1 Market St", "San Francisco", "94105")
    assert svc.cart.step == CheckoutStep.AWAITING_SERVICEABILITY
    svc.lookup_serviceability()
    assert svc.cart.step == CheckoutStep.COLLECTING_DELIVERY
    svc.set_delivery_option("next_day")
    svc.quote_shipping()
    svc.compute_tax()
    assert svc.cart.step == CheckoutStep.COLLECTING_PAYMENT
    svc.attach_payment("card", card_token="tok_42")
    assert svc.cart.step == CheckoutStep.READY_TO_CONFIRM


# ---------- the safety gate ----------
def test_confirm_refuses_with_blockers():
    svc = CartService(Cart())
    svc.add_item("P-2")  # everything else missing
    with pytest.raises(CartError):
        svc.confirm()
    assert svc.cart.confirmed is False
    assert svc.cart.receipt_id is None


def test_confirm_succeeds_when_ready_and_mints_receipt():
    svc = _ready_cart()
    assert svc.cart.ready_to_confirm() is True
    msg = svc.confirm()
    assert svc.cart.confirmed is True
    assert svc.cart.receipt_id and svc.cart.receipt_id.startswith("RCPT-")
    assert "RCPT-" in msg


def test_not_serviceable_blocks_confirmation():
    svc = CartService(Cart())
    svc.add_item("P-2")
    svc.set_customer("Ada", "Lovelace")
    svc.set_address("1 Nowhere Rd", "Atlantis", "99999")  # unserviceable prefix
    svc.lookup_serviceability()
    assert svc.cart.serviceable is False
    codes = {b.code for b in svc.cart.blockers()}
    assert "not_serviceable" in codes
    with pytest.raises(CartError):
        svc.confirm()


# ---------- quote freshness ----------
def test_changing_address_invalidates_quotes_and_blocks():
    svc = _ready_cart()
    assert svc.cart.ready_to_confirm()
    # Move to a different serviceable region → serviceability/shipping/tax stale.
    svc.set_address("100 Broadway", "New York", "10001")
    assert svc.cart.serviceable is None
    assert svc.cart.shipping_is_fresh() is False
    assert svc.cart.ready_to_confirm() is False


def test_adding_item_invalidates_shipping_and_tax():
    svc = _ready_cart()
    assert svc.cart.shipping_is_fresh() and svc.cart.tax_is_fresh()
    svc.add_item("P-1")
    assert svc.cart.shipping_is_fresh() is False
    assert svc.cart.tax_is_fresh() is False


# ---------- delivery option validity ----------
def test_unserviceable_delivery_option_rejected():
    svc = CartService(Cart())
    svc.add_item("P-2")
    svc.set_customer("Ada", "Lovelace")
    svc.set_address("10 Rue", "Paris", "75001")  # FR: only next_day, standard
    svc.lookup_serviceability()
    with pytest.raises(CartError):
        svc.set_delivery_option("2h")


# ---------- promos ----------
def test_promo_requires_qualifying_items():
    svc = CartService(Cart())
    svc.add_item("P-2")  # apparel, no shoes
    with pytest.raises(CartError):
        svc.apply_promo("SHOES20")
    svc.add_item("P-3")  # running shoes
    assert "SHOES20" in svc.apply_promo("SHOES20")


def test_quote_promo_unknown_code():
    import pytest as _pytest

    with _pytest.raises(KeyError):
        quote_promo("NOPE", [])


# ---------- serviceability table ----------
def test_serviceability_lookup():
    assert lookup("94105").city == "San Francisco"
    assert lookup("99999") is None


def test_free_standard_shipping_over_100():
    svc = CartService(Cart())
    svc.add_item("P-3", quantity=2)  # 178.00
    svc.set_customer("A", "B")
    svc.set_address("1 Market St", "San Francisco", "94105")
    svc.lookup_serviceability()
    svc.set_delivery_option("standard")
    svc.quote_shipping()
    assert svc.cart.shipping.cost == Decimal("0.00")
