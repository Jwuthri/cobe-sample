"""Extractors distill a sub-agent's Msg list + cart into a StepResult."""

from __future__ import annotations

from openai_agent_v1.core.messages import ai, tool_msg
from openai_agent_v1.shopping.domain import CartService
from openai_agent_v1.shopping.extractors import (
    extract_order,
    extract_product_rec,
    extract_products,
    extract_serviceability,
)


def _ctx():
    from openai_agent_v1.shopping.context import ShoppingContext

    return ShoppingContext(cart_service=CartService())


def test_extract_products_parses_lines():
    msgs = [
        ai("", [{"name": "search_products", "args": {}}]),
        tool_msg(
            "search_products",
            "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]\n"
            "P-4: Baseball Cap (Green) — $14.50 [apparel, hat, green]",
        ),
    ]
    products = extract_products(msgs)
    assert [p["id"] for p in products] == ["P-2", "P-4"]
    assert products[0]["price"] == "49.99"
    assert "hoodie" in products[0]["tags"]


def test_extract_serviceability():
    msgs = [tool_msg("check_serviceability", "Yes, we ship to zip 94110 (San Francisco, US).")]
    serv = extract_serviceability(msgs)
    assert serv and "94110" in serv["raw"]


def test_extract_order_ignores_unknown():
    assert extract_order([tool_msg("get_order_status", "unknown order: ORD-X")]) is None
    found = extract_order([tool_msg("get_order_status", "Order ORD-7 is shipped, items=['P-1']")])
    assert found and "ORD-7" in found["raw"]


def test_extract_product_rec_add_signals_checkout():
    ctx = _ctx()
    ctx.cart_service.add_item("P-4", 1)
    msgs = [
        ai("", [{"name": "add_item", "args": {"product_id": "P-4"}}]),
        tool_msg("add_item", "Added 1 × Baseball Cap (Green) (cart now $14.50)."),
    ]
    sr = extract_product_rec(ctx, msgs, before={})
    assert sr.sop == "product_rec"
    assert sr.next_sop == "checkout"
    assert sr.details["added"] == ["P-4"]


def test_extract_product_rec_remove():
    ctx = _ctx()  # cart already empty after a remove
    sr = extract_product_rec(ctx, [tool_msg("remove_item", "Removed P-4 (cart now $0.00).")], before={"P-4": 1})
    assert "removed" in sr.summary
    assert sr.details["cart_edit"]["removed"] == ["P-4"]


def test_extract_product_rec_recall_for_shown_products():
    ctx = _ctx()
    msgs = [tool_msg("search_products", "P-4: Baseball Cap (Green) — $14.50 [apparel, hat, green]")]
    sr = extract_product_rec(ctx, msgs, before={})
    assert sr.recall and "P-4" in sr.recall
