"""Deterministic block assembly — ids/prices verbatim, never invented."""

from __future__ import annotations

from agent_v4_1.core.step_result import StepResult
from agent_v4_1.shopping.agents import BLOCK_BY_SOP
from agent_v4_1.shopping.blocks import build_blocks
from agent_v4_1.shopping.domain import CartService


def test_product_reco_block_from_details():
    sr = StepResult(
        sop="product_rec",
        summary="added",
        details={
            "added": ["P-1"],
            "products": [{"id": "P-1", "name": "Tee", "price": "19.99", "tags": ["apparel"]}],
        },
    )
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert blocks == [
        {
            "kind": "product_reco",
            "products": [{"id": "P-1", "name": "Tee", "price": "19.99", "tags": ["apparel"]}],
            "added_ids": ["P-1"],
            "serviceability": None,
        }
    ]


def test_order_status_block_resolves_order():
    sr = StepResult(
        sop="order_status",
        summary="looked up",
        details={"raw": "Order ORD-7 is shipped, items=['P-1', 'P-4']"},
    )
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "order_status"
    assert blocks[0]["order"]["id"] == "ORD-7"
    assert blocks[0]["order"]["status"] == "shipped"


def test_checkout_block_from_cart():
    cs = CartService()
    cs.add_item("P-2", 1)
    sr = StepResult(sop="checkout", summary="checkout", asks=["payment method"])
    blocks = build_blocks([sr], cs.cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "checkout"
    assert blocks[0]["items"][0]["id"] == "P-2"
    assert blocks[0]["asks"] == ["payment method"]
    assert blocks[0]["confirmed"] is False


def test_smalltalk_yields_no_blocks():
    assert build_blocks([], CartService().cart, BLOCK_BY_SOP) == []


def test_multi_block_turn():
    srs = [
        StepResult(
            sop="product_rec",
            details={"products": [{"id": "P-1", "name": "Tee", "price": "19.99", "tags": []}]},
        ),
        StepResult(sop="order_status", details={"raw": "Order ORD-9 is delivered"}),
    ]
    blocks = build_blocks(srs, CartService().cart, BLOCK_BY_SOP)
    assert [b["kind"] for b in blocks] == ["product_reco", "order_status"]


def test_checkout_block_deduped():
    cs = CartService()
    cs.add_item("P-1")
    srs = [StepResult(sop="checkout"), StepResult(sop="checkout")]
    blocks = build_blocks(srs, cs.cart, BLOCK_BY_SOP)
    assert sum(1 for b in blocks if b["kind"] == "checkout") == 1
