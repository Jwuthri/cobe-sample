"""Extractor + anchor domain logic (no LLM)."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from agent_v4_1.shopping.context import ShoppingContext
from agent_v4_1.shopping.domain import CartService
from agent_v4_1.shopping.extractors import (
    asks_for_step,
    cart_quantities,
    checkout_anchor_text,
    extract_product_rec,
    extract_products,
)


def test_product_line_regex_parsing():
    msgs = [
        ToolMessage(
            content="P-2: Black Hoodie — $49.99 [apparel, hoodie, black]",
            tool_call_id="t",
            name="search_products",
        )
    ]
    products = extract_products(msgs)
    assert products == [
        {"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["apparel", "hoodie", "black"]}
    ]


def test_extract_product_rec_classifies_add():
    cs = CartService()
    ctx = ShoppingContext(cart_service=cs)
    before = cart_quantities(ctx)  # {} (empty)
    cs.add_item("P-1")
    sr = extract_product_rec(ctx, [], before)
    assert sr.sop == "product_rec"
    assert sr.details["added"] == ["P-1"]
    assert sr.next_sop == "checkout"


def test_extract_product_rec_classifies_remove():
    cs = CartService()
    cs.add_item("P-1")
    cs.add_item("P-2")
    ctx = ShoppingContext(cart_service=cs)
    before = cart_quantities(ctx)  # {P-1:1, P-2:1}
    cs.remove_item("P-1")
    sr = extract_product_rec(ctx, [], before)
    assert "removed" in sr.summary
    assert sr.details["cart_edit"]["removed"] == ["P-1"]


def test_asks_for_step():
    cs = CartService()
    assert asks_for_step("collecting_identity", cs.cart) == ["first name", "last name"]
    assert asks_for_step("collecting_payment", cs.cart)[0].startswith("payment method")
    assert asks_for_step("ready_to_confirm", cs.cart) == []


def test_checkout_anchor_renders_progress():
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("Ada", "Lovelace")
    text = checkout_anchor_text(cs.cart)
    assert "✓ Ada Lovelace" in text  # identity captured
    assert "address:        — not provided" in text
    assert "Resume from:" in text


def test_checkout_anchor_flags_unserviceable():
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("A", "B")
    cs.set_address("1 Main", "Nowhere", "99999")
    cs.lookup_serviceability()  # 99999 not in table → not serviceable
    text = checkout_anchor_text(cs.cart)
    assert "NOT serviceable" in text
