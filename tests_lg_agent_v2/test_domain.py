"""The checkout state machine + invariants — the behavioral spec, tested directly.

These are pure-logic tests (no agents, no LLM). They are the contract the whole
assistant exists to satisfy: the step order, freshness/staleness, the blocker gate,
and the confabulated-name guard.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from lg_agent_v2.domain import CartError, CartService, CheckoutStep


def _ready_cart() -> CartService:
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 Market St", "San Francisco", "94105")
    s.lookup_serviceability()
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")
    return s


# --------------------------------------------------------------------------- #
# step progression
# --------------------------------------------------------------------------- #
def test_step_walks_the_full_flow():
    s = CartService()
    assert s.cart.step is CheckoutStep.COLLECTING_PRODUCTS
    s.add_item("P-2", 1)
    assert s.cart.step is CheckoutStep.COLLECTING_IDENTITY
    s.set_customer("Ada", "Lovelace")
    assert s.cart.step is CheckoutStep.COLLECTING_ADDRESS
    s.set_address("1 Market St", "San Francisco", "94105")
    assert s.cart.step is CheckoutStep.AWAITING_SERVICEABILITY
    s.lookup_serviceability()
    assert s.cart.step is CheckoutStep.COLLECTING_DELIVERY
    s.set_delivery_option("2h")
    assert s.cart.step is CheckoutStep.COLLECTING_PAYMENT  # pricing not yet computed
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")
    assert s.cart.step is CheckoutStep.READY_TO_CONFIRM


def test_confirm_sets_receipt_and_step():
    s = _ready_cart()
    assert s.cart.ready_to_confirm()
    msg = s.confirm()
    assert "RCPT-9000" in msg
    assert s.cart.confirmed is True
    assert s.cart.step is CheckoutStep.CONFIRMED
    assert s.cart.receipt_id == "RCPT-9000"


def test_grand_total_is_subtotal_plus_shipping_plus_tax():
    s = _ready_cart()
    # 49.99 + 19.99 shipping (941/2h) + tax(8.75% of 49.99 = 4.37)
    assert s.cart.grand_total == Decimal("49.99") + Decimal("19.99") + Decimal("4.37")


# --------------------------------------------------------------------------- #
# freshness / staleness
# --------------------------------------------------------------------------- #
def test_quantity_change_makes_pricing_stale_and_blocks_total():
    s = _ready_cart()
    assert s.cart.step is CheckoutStep.READY_TO_CONFIRM
    s.set_quantity("P-2", 3)  # cart edit invalidates shipping + tax
    assert not s.cart.shipping_is_fresh()
    assert not s.cart.tax_is_fresh()
    assert s.cart.grand_total is None
    assert s.cart.step is CheckoutStep.AWAITING_PRICING
    s.quote_shipping()
    s.compute_tax()
    assert s.cart.step is CheckoutStep.READY_TO_CONFIRM


def test_changing_zip_invalidates_serviceability_and_quotes():
    s = _ready_cart()
    s.set_address("9 Rue", "Paris", "75001")  # different zip
    assert s.cart.serviceable is None
    assert s.cart.delivery_option is None
    assert s.cart.shipping is None and s.cart.tax is None
    assert s.cart.step is CheckoutStep.AWAITING_SERVICEABILITY


# --------------------------------------------------------------------------- #
# the blocker gate
# --------------------------------------------------------------------------- #
def test_confirm_refuses_with_blockers():
    s = CartService()
    s.add_item("P-2", 1)
    with pytest.raises(CartError):
        s.confirm()
    assert s.cart.confirmed is False


def test_unserviceable_address_routes_back_to_address_step():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 X", "Nowhere", "00000")  # not in the serviceability table
    s.lookup_serviceability()
    assert s.cart.serviceable is False
    assert s.cart.step is CheckoutStep.COLLECTING_ADDRESS  # not a dead end
    assert any(b.code == "not_serviceable" for b in s.cart.blockers())


def test_delivery_option_must_be_serviceable():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_address("9 Rue", "Paris", "75001")
    s.lookup_serviceability()  # Paris → only next_day, standard
    with pytest.raises(CartError):
        s.set_delivery_option("2h")


def test_card_payment_needs_a_token():
    s = _ready_cart()
    s.attach_payment("card")  # no token
    assert s.cart.step is CheckoutStep.COLLECTING_PAYMENT
    assert any(b.code == "missing_card_token" for b in s.cart.blockers())
    s.attach_payment("card", card_token="tok_123")
    assert s.cart.ready_to_confirm()


# --------------------------------------------------------------------------- #
# promo
# --------------------------------------------------------------------------- #
def test_promo_applies_and_revalidates_on_item_removal():
    s = CartService()
    s.add_item("P-3", 1)  # shoes
    s.apply_promo("SHOES20")
    assert s.cart.promo is not None
    s.remove_item("P-3")  # promo no longer applies → auto-cleared
    assert s.cart.promo is None


def test_unknown_and_unqualified_promos_raise():
    s = CartService()
    s.add_item("P-2", 1)  # not shoes
    with pytest.raises(CartError):
        s.apply_promo("NOPE")
    with pytest.raises(CartError):
        s.apply_promo("SHOES20")


# --------------------------------------------------------------------------- #
# confabulated-name guard
# --------------------------------------------------------------------------- #
def test_set_customer_rejects_label_words_and_digits():
    s = CartService()
    s.add_item("P-2", 1)
    with pytest.raises(CartError):
        s.set_customer("Shipping", "address")  # all label words
    with pytest.raises(CartError):
        s.set_customer("1717", "Webster")  # digits → an address, not a name
    s.set_customer("Ada", "Lovelace")  # a real name is fine
    assert s.cart.customer.is_complete()
