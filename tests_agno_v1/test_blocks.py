"""Deterministic block assembly (the hallucination firewall)."""

from __future__ import annotations

from agent_agno_v1.core.step_result import StepResult
from agent_agno_v1.shopping.agents import BLOCK_BY_SOP
from agent_agno_v1.shopping.blocks import build_blocks
from agent_agno_v1.shopping.domain import CartService


def test_product_block_from_step():
    sr = StepResult(
        sop="product_rec",
        details={
            "products": [{"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["apparel"]}],
            "added": ["P-2"],
        },
    )
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "product_reco"
    assert blocks[0]["products"][0]["id"] == "P-2"
    assert blocks[0]["added_ids"] == ["P-2"]


def test_order_status_block_resolves_known_order():
    sr = StepResult(sop="order_status", details={"raw": "Order ORD-7 is shipped"})
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "order_status"
    assert blocks[0]["order"]["id"] == "ORD-7"


def test_checkout_block_from_cart():
    cs = CartService()
    cs.add_item("P-2")
    sr = StepResult(sop="checkout", asks=["first name", "last name"])
    blocks = build_blocks([sr], cs.cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "checkout"
    assert blocks[0]["items"][0]["id"] == "P-2"
    assert blocks[0]["confirmed"] is False
    assert blocks[0]["asks"] == ["first name", "last name"]


def test_conversational_turn_has_no_blocks():
    assert build_blocks([], CartService().cart, BLOCK_BY_SOP) == []


def test_single_checkout_block_even_with_multiple_steps():
    cs = CartService()
    cs.add_item("P-2")
    steps = [StepResult(sop="checkout"), StepResult(sop="checkout")]
    blocks = build_blocks(steps, cs.cart, BLOCK_BY_SOP)
    assert sum(1 for b in blocks if b["kind"] == "checkout") == 1
