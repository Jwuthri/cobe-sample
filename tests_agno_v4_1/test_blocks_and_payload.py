"""Deterministic blocks + writer payload shaping (no LLM)."""

from __future__ import annotations

from agent_agno_v4_1 import tools
from agent_agno_v4_1.extractors import BLOCK_BY_SOP
from agent_agno_v4_1.writer_payload import build_writer_payload
from agent_v4_1.core.step_result import StepResult
from agent_v4_1.shopping.blocks import build_blocks


def _sr(**kw) -> StepResult:
    return StepResult(**kw)


def test_blocks_product_reco_from_catalog(ctx):
    sr = _sr(
        sop="product_rec",
        summary="catalog returned 1",
        details={"products": [{"id": "P-1", "name": "Tee", "price": "19.99", "tags": ["x"]}]},
    )
    blocks = build_blocks([sr], ctx.cart_service.cart, BLOCK_BY_SOP)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "product_reco"
    assert blocks[0]["products"][0]["id"] == "P-1"


def test_blocks_order_status_resolves_order(ctx):
    sr = _sr(
        sop="order_status",
        summary="looked up order status",
        details={"raw": "Order ORD-7 is shipped"},
    )
    blocks = build_blocks([sr], ctx.cart_service.cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "order_status"
    assert blocks[0]["order"]["id"] == "ORD-7"


def test_blocks_checkout_summary(run_context, ctx):
    rc = run_context()
    tools.add_item("P-1", rc, quantity=2)
    sr = _sr(sop="checkout", summary="...", asks=["first name", "last name"])
    blocks = build_blocks([sr], ctx.cart_service.cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "checkout"
    assert blocks[0]["subtotal"] == "39.98"
    assert blocks[0]["asks"] == ["first name", "last name"]


def test_blocks_of_different_types_in_one_turn(ctx):
    # a compound turn: product list + order status -> two blocks, in order.
    srs = [
        _sr(sop="order_status", summary="x", details={"raw": "Order ORD-7 is shipped"}),
        _sr(sop="product_rec", summary="y",
            details={"products": [{"id": "P-2", "name": "Hoodie", "price": "49.99", "tags": []}]}),
    ]
    kinds = [b["kind"] for b in build_blocks(srs, ctx.cart_service.cart, BLOCK_BY_SOP)]
    assert kinds == ["order_status", "product_reco"]


def test_writer_payload_mode_checkout(run_context, ctx):
    rc = run_context()
    tools.add_item("P-1", rc, quantity=2)
    sr = _sr(sop="checkout", summary="...", asks=["first name"])
    payload, mode = build_writer_payload(
        [{"role": "user", "content": "checkout"}], [sr], ctx.cart_service.cart
    )
    assert mode == "checkout"
    assert '"step": "collecting_identity"' in payload
    assert '"subtotal": "39.98"' in payload


def test_writer_payload_mode_smalltalk():
    payload, mode = build_writer_payload([{"role": "user", "content": "hi"}], [], None)
    assert mode == "smalltalk"
    assert '"cart"' not in payload
