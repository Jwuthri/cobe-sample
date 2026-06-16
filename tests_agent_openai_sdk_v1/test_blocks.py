"""Writer-block builder tests.

The deterministic typed blocks are the "hallucination firewall" — the LLM streams
prose, but the structured cards are built from verified step results + the live
cart so an id or a total can never be invented.
"""

from __future__ import annotations

from agent_openai_sdk_v1.agents import BLOCK_BY_SOP, build_blocks
from agent_openai_sdk_v1.agents.names import CHECKOUT, ORDER_STATUS, PRODUCT_REC
from agent_openai_sdk_v1.domain import CartService
from agent_openai_sdk_v1.runtime import StepResult


def test_product_reco_block_renders_products_and_added_ids():
    cart = CartService().cart
    sr = StepResult(
        sop=PRODUCT_REC,
        summary="added P-2",
        details={
            "added": ["P-2"],
            "products": [
                {"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["apparel", "hoodie"]}
            ],
        },
    )
    blocks = build_blocks([sr], cart, BLOCK_BY_SOP)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["kind"] == "product_reco"
    assert b["added_ids"] == ["P-2"]
    assert b["products"][0]["id"] == "P-2"


def test_checkout_block_reflects_live_cart():
    s = CartService()
    s.add_item("P-2", 1)
    sr = StepResult(sop=CHECKOUT, summary="captured identity", asks=["street", "city"])
    blocks = build_blocks([sr], s.cart, BLOCK_BY_SOP)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["kind"] == "checkout"
    assert b["items"][0]["id"] == "P-2"
    assert b["subtotal"] == "49.99"
    assert b["asks"] == ["street", "city"]


def test_order_status_block_renders_raw_lookup():
    cart = CartService().cart
    sr = StepResult(
        sop=ORDER_STATUS, summary="looked up", details={"raw": "Order ORD-7 is shipped"}
    )
    blocks = build_blocks([sr], cart, BLOCK_BY_SOP)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "order_status"
    assert blocks[0]["raw"] == "Order ORD-7 is shipped"


def test_no_blocks_for_pure_smalltalk():
    cart = CartService().cart
    assert build_blocks([], cart, BLOCK_BY_SOP) == []
