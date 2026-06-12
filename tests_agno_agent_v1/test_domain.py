"""Pure-domain tests — the behavioral spec, no framework, no LLM.

These lock down cart math, the checkout step machine, the repricing-on-edit
invariant, and the confirmation gate — the rules every higher layer relies on.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agno_agent_v1.domain import CartService
from agno_agent_v1.domain.cart_service import CartError, _looks_like_name
from agno_agent_v1.domain.pricing import quote_promo, quote_shipping, quote_tax
from agno_agent_v1.domain.serviceability import lookup


def _ready_cart() -> CartService:
    s = CartService()
    s.add_item("P-1", 2)  # 2 × 19.99 = 39.98
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 Market St", "San Francisco", "94105")
    s.lookup_serviceability()
    s.set_delivery_option("standard")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")
    return s


def test_step_machine_advances_with_state():
    s = CartService()
    assert s.cart.step.value == "collecting_products"
    s.add_item("P-1")
    assert s.cart.step.value == "collecting_identity"
    s.set_customer("Ada", "Lovelace")
    assert s.cart.step.value == "collecting_address"
    s.set_address("1 Market St", "San Francisco", "94105")
    assert s.cart.step.value == "awaiting_serviceability"
    s.lookup_serviceability()
    assert s.cart.step.value == "collecting_delivery"
    s.set_delivery_option("standard")
    # delivery set but shipping/tax not yet quoted
    assert s.cart.step.value == "collecting_payment" or s.cart.step.value == "awaiting_pricing"


def test_subtotal_and_grand_total_math():
    s = _ready_cart()
    assert s.cart.subtotal == Decimal("39.98")
    # 94105 standard under $100 → 0.00 shipping is NOT free (free only >= 100)
    cost, eta = quote_shipping("94105", "standard", s.cart.subtotal)
    assert cost == Decimal("0.00")  # 941 standard tier is 0.00
    assert s.cart.ready_to_confirm()
    # grand_total = subtotal + shipping + tax
    rate, amount = quote_tax("94105", s.cart.subtotal)
    assert s.cart.grand_total == s.cart.subtotal + s.cart.shipping.cost + s.cart.tax.amount


def test_editing_items_invalidates_pricing():
    """The repricing-on-backtrack invariant."""
    s = _ready_cart()
    assert s.cart.ready_to_confirm()
    assert s.cart.grand_total is not None
    # backtrack: change quantity → shipping + tax go stale, total becomes None
    s.set_quantity("P-1", 5)
    assert s.cart.step.value == "awaiting_pricing"
    assert s.cart.grand_total is None
    codes = {b.code for b in s.cart.blockers()}
    assert "stale_shipping" in codes and "stale_tax" in codes
    # re-quote → ready again
    s.quote_shipping()
    s.compute_tax()
    assert s.cart.ready_to_confirm()


def test_confirm_gate_blocks_until_complete():
    s = CartService()
    s.add_item("P-1")
    with pytest.raises(CartError):
        s.confirm()  # blockers present
    s = _ready_cart()
    msg = s.confirm()
    assert s.cart.confirmed and s.cart.receipt_id and "RCPT-" in msg


def test_changing_zip_invalidates_serviceability_and_delivery():
    s = _ready_cart()
    assert s.cart.serviceable is True
    s.set_address("9 Rue de Rivoli", "Paris", "75001")
    assert s.cart.serviceable is None  # invalidated
    assert s.cart.delivery_option is None
    s.lookup_serviceability()
    assert set(s.cart.serviceable_options) == {"next_day", "standard"}
    with pytest.raises(CartError):
        s.set_delivery_option("2h")  # not available in Paris


def test_name_guard_rejects_labels_and_digits():
    assert _looks_like_name("Ada")
    assert not _looks_like_name("Shipping address")
    assert not _looks_like_name("1717 Webster")
    s = CartService()
    s.add_item("P-1")
    with pytest.raises(CartError):
        s.set_customer("Shipping", "address")


def test_promo_rules():
    s = CartService()
    s.add_item("P-1")  # not a shoe
    with pytest.raises(CartError):
        s.apply_promo("SHOES20")  # no qualifying items
    s.add_item("P-3")  # sneakers (shoes)
    msg = s.apply_promo("SHOES20")
    assert "SHOES20" in msg and s.cart.promo.code == "SHOES20"
    # removing the shoe revalidates promo away
    s.remove_item("P-3")
    assert s.cart.promo is None


def test_serviceability_table():
    assert lookup("94105").city == "San Francisco"
    assert lookup("75001").country == "FR"
    assert lookup("99999") is None
