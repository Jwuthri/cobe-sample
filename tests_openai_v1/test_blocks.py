"""Deterministic block assembly (the hallucination firewall)."""

from __future__ import annotations

from openai_agent_v1.core.step_result import StepResult
from openai_agent_v1.shopping.agents import BLOCK_BY_SOP
from openai_agent_v1.shopping.blocks import build_blocks
from openai_agent_v1.shopping.domain import CartService


def test_product_reco_block_from_step():
    sr = StepResult(
        sop="product_rec",
        summary="catalog returned 1",
        details={"products": [{"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["hoodie"]}]},
    )
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "product_reco"
    assert blocks[0]["products"][0]["id"] == "P-2"


def test_checkout_block_carries_cart_facts():
    svc = CartService()
    svc.add_item("P-2", 1)
    sr = StepResult(sop="checkout", summary="at identity", asks=["first name", "last name"])
    blocks = build_blocks([sr], svc.cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "checkout"
    assert blocks[0]["items"][0]["id"] == "P-2"
    assert blocks[0]["subtotal"] == "49.99"
    assert blocks[0]["asks"] == ["first name", "last name"]


def test_smalltalk_yields_no_blocks():
    assert build_blocks([], CartService().cart, BLOCK_BY_SOP) == []


def test_order_status_block_resolves_order():
    sr = StepResult(sop="order_status", summary="looked up", details={"raw": "Order ORD-7 is shipped"})
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "order_status"
    assert blocks[0]["order"]["id"] == "ORD-7"


def test_only_one_checkout_block():
    svc = CartService()
    svc.add_item("P-2", 1)
    srs = [StepResult(sop="checkout", summary="a"), StepResult(sop="checkout", summary="b")]
    blocks = build_blocks(srs, svc.cart, BLOCK_BY_SOP)
    assert sum(1 for b in blocks if b["kind"] == "checkout") == 1
