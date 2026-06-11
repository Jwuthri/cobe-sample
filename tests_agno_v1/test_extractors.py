"""Tool-event distillation → StepResults + the checkout anchor."""

from __future__ import annotations

from agent_agno_v1.core.context import ToolEvent
from agent_agno_v1.shopping.domain import CartService
from agent_agno_v1.shopping.extractors import (
    cart_quantities,
    checkout_anchor_text,
    extract_checkout,
    extract_order,
    extract_order_status,
    extract_product_rec,
    extract_products,
    extract_serviceability,
)

HOODIE = "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"


def _ev(sop, name, args=None, result=""):
    return ToolEvent(sop=sop, name=name, args=args or {}, result=result)


def test_extract_products_parses_lines():
    events = [_ev("product_rec", "search_products", {"query": "h"}, result=HOODIE)]
    products = extract_products(events)
    assert products == [
        {"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["apparel", "hoodie", "black"]}
    ]


def test_extract_serviceability_and_order():
    serv = extract_serviceability([_ev("product_rec", "check_serviceability", result="Yes, we ship to 94110.")])
    assert serv == {"raw": "Yes, we ship to 94110."}
    order = extract_order([_ev("order_status", "get_order_status", result="Order ORD-7 is shipped")])
    assert order["raw"].startswith("Order ORD-7")
    assert extract_order([_ev("order_status", "get_order_status", result="unknown order: X")]) is None


def test_product_rec_added_sets_next_checkout():
    cs = CartService()
    before = cart_quantities(cs.cart)
    cs.add_item("P-2")
    events = [_ev("product_rec", "add_item", {"product_id": "P-2"}, result="Added")]
    sr = extract_product_rec(cs.cart, events, before)
    assert sr.next_sop == "checkout"
    assert sr.details["added"] == ["P-2"]


def test_product_rec_search_only():
    cs = CartService()
    before = cart_quantities(cs.cart)
    events = [_ev("product_rec", "search_products", {"query": "h"}, result=HOODIE)]
    sr = extract_product_rec(cs.cart, events, before)
    assert sr.next_sop is None
    assert sr.details["products"][0]["id"] == "P-2"
    assert sr.asks  # prompts to pick a product


def test_product_rec_removed():
    cs = CartService()
    cs.add_item("P-2")
    before = cart_quantities(cs.cart)
    cs.remove_item("P-2")
    sr = extract_product_rec(cs.cart, [_ev("product_rec", "remove_item", {"product_id": "P-2"})], before)
    assert "P-2" in sr.details["cart_edit"]["removed"]


def test_checkout_asks_reflect_step():
    cs = CartService()
    cs.add_item("P-2")  # next step: identity
    sr = extract_checkout(cs.cart, [], None)
    assert sr.sop == "checkout"
    assert "first name" in sr.asks


def test_order_status_extractor():
    sr = extract_order_status(None, [_ev("order_status", "get_order_status", result="Order ORD-7 is shipped")], None)
    assert sr.details["raw"].startswith("Order ORD-7")
    assert sr.asks == []


def test_checkout_anchor_shows_progress():
    cs = CartService()
    cs.add_item("P-2")
    cs.set_customer("Ada", "Lovelace")
    anchor = checkout_anchor_text(cs.cart)
    assert "Checkout progress" in anchor
    assert "✓ Ada Lovelace" in anchor
    assert "Resume from" in anchor
