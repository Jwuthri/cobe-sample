"""The deterministic block builder — the hallucination firewall.

Blocks are assembled from step results + the live cart, never written by the model,
so ids/prices/totals are always grounded.
"""

from __future__ import annotations

from pydantic_agent_v1.agents import BLOCK_BY_SOP, build_blocks
from pydantic_agent_v1.domain import CartService
from pydantic_agent_v1.runtime import StepResult


def test_product_reco_block_from_step_details():
    sr = StepResult(
        sop="product_rec",
        summary="catalog returned 1",
        details={"products": [{"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["apparel"]}]},
    )
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "product_reco"
    assert blocks[0]["products"][0]["id"] == "P-2"


def test_order_status_block_resolves_known_order():
    sr = StepResult(sop="order_status", summary="ok", details={"raw": "Order ORD-7 is shipped"})
    blocks = build_blocks([sr], CartService().cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "order_status"
    assert blocks[0]["order"]["id"] == "ORD-7"


def test_checkout_block_uses_live_cart_not_model_text():
    s = CartService()
    s.add_item("P-2", 1)
    sr = StepResult(sop="checkout", summary="checkout", asks=["first name", "last name"])
    blocks = build_blocks([sr], s.cart, BLOCK_BY_SOP)
    assert blocks[0]["kind"] == "checkout"
    assert blocks[0]["items"][0]["id"] == "P-2"
    assert blocks[0]["subtotal"] == "49.99"
    assert blocks[0]["confirmed"] is False
    assert blocks[0]["asks"] == ["first name", "last name"]


def test_conversational_turn_yields_no_blocks():
    assert build_blocks([], CartService().cart, BLOCK_BY_SOP) == []
