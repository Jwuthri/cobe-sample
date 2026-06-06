"""Regression tests: wrappers MUST pass structured data to the writer.

The bug being prevented: product_rec saying 'I presented options'
without actually including the products. The writer then either
hallucinates or asks the user to 'rerun with the product data'.
"""

from __future__ import annotations

from agent_v2.graph import _extract_order_from_messages, _extract_products_from_messages
from langchain_core.messages import AIMessage, ToolMessage


def _tool_msg(name: str, content: str) -> ToolMessage:
    return ToolMessage(name=name, content=content, tool_call_id="t-1")


def test_extract_products_from_search_results():
    msgs = [
        _tool_msg(
            "search_products",
            "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]\n"
            "P-4: Baseball Cap (Green) — $14.50 [apparel, hat, green]",
        ),
        AIMessage(content="here you go"),
    ]
    products = _extract_products_from_messages(msgs)
    assert len(products) == 2
    assert products[0] == {
        "id": "P-2",
        "name": "Black Hoodie",
        "price": "49.99",
        "tags": ["apparel", "hoodie", "black"],
    }
    assert products[1]["name"] == "Baseball Cap (Green)"
    assert "green" in products[1]["tags"]


def test_extract_products_dedups_across_tool_calls():
    msgs = [
        _tool_msg("search_products", "P-2: Black Hoodie — $49.99 [apparel, hoodie]"),
        _tool_msg("get_product", "P-2: Black Hoodie — $49.99 [apparel, hoodie]"),
    ]
    products = _extract_products_from_messages(msgs)
    assert len(products) == 1


def test_extract_products_ignores_unknown_tools():
    msgs = [_tool_msg("compute_tax", "P-2: Black Hoodie — $49.99 [apparel]")]
    assert _extract_products_from_messages(msgs) == []


def test_extract_products_ignores_no_results_message():
    msgs = [_tool_msg("search_products", "No products match 'xyz'.")]
    assert _extract_products_from_messages(msgs) == []


def test_extract_order_keeps_raw_text():
    msgs = [
        _tool_msg(
            "get_order_status",
            "Order ORD-7 is shipped, items=['P-1', 'P-4'], tracking: https://track.example/ORD-7",
        )
    ]
    out = _extract_order_from_messages(msgs)
    assert out is not None
    assert "ORD-7" in out["raw"]
    assert "shipped" in out["raw"]


def test_extract_order_skips_not_found():
    msgs = [_tool_msg("get_order_status", "unknown order: ORD-99")]
    assert _extract_order_from_messages(msgs) is None
