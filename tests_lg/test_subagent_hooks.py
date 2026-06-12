"""Sub-agent domain hooks: tool-result parsing, classification, the progress block."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from lg_agent.shopping.agents.subagents import checkout, product_rec
from lg_agent.shopping.context import ShoppingContext
from lg_agent.shopping.domain import CartService


def test_product_line_regex_parsing():
    msgs = [
        ToolMessage(
            content="P-2: Black Hoodie — $49.99 [apparel, hoodie, black]",
            tool_call_id="t",
            name="search_products",
        )
    ]
    products = product_rec._parse_products(msgs)
    assert products == [
        {"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["apparel", "hoodie", "black"]}
    ]


def test_extract_product_rec_classifies_add():
    cs = CartService()
    ctx = ShoppingContext(cart_service=cs)
    before = product_rec.snapshot(ctx)  # {} (empty)
    cs.add_item("P-1")
    sr = product_rec.extract(ctx, [], before)
    assert sr.sop == "product_rec"
    assert sr.details["added"] == ["P-1"]
    assert sr.next_sop == "checkout"


def test_extract_product_rec_classifies_remove():
    cs = CartService()
    cs.add_item("P-1")
    cs.add_item("P-2")
    ctx = ShoppingContext(cart_service=cs)
    before = product_rec.snapshot(ctx)  # {P-1:1, P-2:1}
    cs.remove_item("P-1")
    sr = product_rec.extract(ctx, [], before)
    assert "removed" in sr.summary
    assert sr.details["cart_edit"]["removed"] == ["P-1"]


def test_asks_for_step():
    cs = CartService()
    assert checkout.asks_for_step("collecting_identity", cs.cart) == ["first name", "last name"]
    assert checkout.asks_for_step("collecting_payment", cs.cart)[0].startswith("payment method")
    assert checkout.asks_for_step("ready_to_confirm", cs.cart) == []


def test_checkout_progress_renders():
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("Ada", "Lovelace")
    text = checkout.checkout_progress(cs.cart)
    assert "✓ Ada Lovelace" in text  # identity captured
    assert "address:        — not provided" in text
    # the block names the skill the current step needs (detail loads on demand)
    assert "load_skill('collect_address')" in text


def test_checkout_progress_flags_unserviceable():
    cs = CartService()
    cs.add_item("P-1")
    cs.set_customer("A", "B")
    cs.set_address("1 Main", "Nowhere", "99999")
    cs.lookup_serviceability()  # 99999 not in table → not serviceable
    text = checkout.checkout_progress(cs.cart)
    assert "NOT serviceable" in text
