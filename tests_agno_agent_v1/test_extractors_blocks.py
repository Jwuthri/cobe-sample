"""Extractors + deterministic blocks — the grounding/hallucination-firewall layer."""

from __future__ import annotations

from agno_agent_v1.agent.agents import BLOCK_BY_SOP
from agno_agent_v1.agent.blocks import build_blocks
from agno_agent_v1.agent.context import ShoppingContext, StepResult
from agno_agent_v1.agent.extractors import (
    cart_quantities,
    extract_checkout,
    extract_order_status,
    extract_product_rec,
)
from agno_agent_v1.domain import CartService
from tests_agno_agent_v1.conftest import FakeToolCall

_SEARCH_RESULT = (
    "P-4: Baseball Cap (Green) — $14.50 [apparel, hat, green]\n"
    "P-5: Baseball Cap (Red) — $14.50 [apparel, hat, red]"
)


def _ctx() -> ShoppingContext:
    return ShoppingContext(cart_service=CartService())


def test_extract_product_rec_search():
    ctx = _ctx()
    before = cart_quantities(ctx)
    calls = [FakeToolCall("search_products", _SEARCH_RESULT)]
    sr = extract_product_rec(ctx, calls, before)
    assert sr.sop == "product_rec"
    assert sr.details["products"][0]["id"] == "P-4"
    assert len(sr.details["products"]) == 2
    # recall snippet carries the shown ids for next-turn reference resolution
    assert "P-4" in sr.recall and "P-5" in sr.recall


def test_extract_product_rec_add_sets_next_checkout():
    ctx = _ctx()
    before = cart_quantities(ctx)
    ctx.cart_service.add_item("P-4")  # mutate as the tool would
    calls = [FakeToolCall("add_item", "Added 1 × Baseball Cap (Green)")]
    sr = extract_product_rec(ctx, calls, before)
    assert sr.details["added"] == ["P-4"]
    assert sr.next_sop == "checkout"


def test_extract_product_rec_remove_is_cart_edit():
    ctx = _ctx()
    ctx.cart_service.add_item("P-4")
    before = cart_quantities(ctx)  # {P-4: 1}
    ctx.cart_service.remove_item("P-4")
    sr = extract_product_rec(ctx, [FakeToolCall("remove_item", "Removed P-4")], before)
    assert "P-4" in sr.details["cart_edit"]["removed"]
    assert sr.details["cart_edit"]["items"] == []


def test_extract_checkout_reports_step_and_asks():
    ctx = _ctx()
    ctx.cart_service.add_item("P-1")
    sr = extract_checkout(ctx, [], None)
    assert sr.sop == "checkout"
    assert "first name" in sr.asks  # collecting_identity


def test_extract_order_status():
    ctx = _ctx()
    calls = [FakeToolCall("get_order_status", "Order ORD-7 is shipped, items=['P-1', 'P-4'], tracking: https://x")]
    sr = extract_order_status(ctx, calls, None)
    assert sr.details["raw"].startswith("Order ORD-7")
    assert "ORD-7" in sr.recall


def test_build_blocks_product_and_checkout():
    cart = CartService().cart
    # a product_reco block from a search step
    search_sr = StepResult(
        sop="product_rec",
        details={"products": [{"id": "P-4", "name": "Baseball Cap (Green)", "price": "14.50", "tags": ["green"]}]},
    )
    blocks = build_blocks([search_sr], cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "product_reco"
    assert blocks[0]["products"][0]["id"] == "P-4"

    # a checkout block reflects the live cart deterministically
    cs = CartService()
    cs.add_item("P-1", 2)
    co_sr = StepResult(sop="checkout", asks=["payment method"])
    blocks = build_blocks([co_sr], cs.cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "checkout"
    assert blocks[0]["items"][0]["id"] == "P-1"
    assert blocks[0]["subtotal"] == "39.98"


def test_order_status_block_resolves_order():
    cart = CartService().cart
    sr = StepResult(sop="order_status", details={"raw": "Order ORD-7 is shipped"})
    blocks = build_blocks([sr], cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "order_status"
    assert blocks[0]["order"]["id"] == "ORD-7"
