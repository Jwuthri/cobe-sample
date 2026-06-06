"""build_blocks — deterministic typed blocks from step results + cart.

The writer's LLM only writes the prose `message`; the structured blocks are
assembled from what the leaves already produced, so ids/prices are verbatim.
"""

from __future__ import annotations

from agent_v4 import ids
from agent_v4.checkout import CartService
from agent_v4.state import AgentState
from agent_v4.step_result import StepResult
from agent_v4.writer import build_blocks


def _state(**kw) -> AgentState:
    return AgentState(user_id="u", session_id="s", **kw)


def test_no_step_results_yields_no_blocks():
    assert build_blocks(_state()) == []


def test_product_reco_block_from_products():
    s = _state(
        step_results=[
            StepResult(
                sop=ids.PRODUCT_REC,
                details={"products": [{"id": "P-2", "name": "Black Hoodie", "price": "49.99", "tags": ["apparel"]}]},
            )
        ]
    )
    blocks = build_blocks(s)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["kind"] == "product_reco"
    assert b["products"][0]["id"] == "P-2"
    assert b["products"][0]["price"] == "49.99"


def test_product_reco_block_carries_added_and_serviceability():
    s = _state(
        step_results=[
            StepResult(
                sop=ids.PRODUCT_REC,
                details={"added": ["P-3"], "serviceability": {"raw": "Yes, we ship to 94110."}},
            )
        ]
    )
    b = build_blocks(s)[0]
    assert b["added_ids"] == ["P-3"]
    assert "94110" in b["serviceability"]


def test_product_reco_with_no_data_yields_no_block():
    # product_rec ran but matched nothing → prose only, no structured block.
    s = _state(step_results=[StepResult(sop=ids.PRODUCT_REC, summary="no match", asks=["clarify"])])
    assert build_blocks(s) == []


def test_order_status_block_resolves_structured_order():
    s = _state(
        step_results=[
            StepResult(
                sop=ids.ORDER_STATUS,
                details={"raw": "Order ORD-7 is shipped, items=[P-1, P-4], tracking: https://track.example/ORD-7"},
            )
        ]
    )
    b = build_blocks(s)[0]
    assert b["kind"] == "order_status"
    assert b["order"]["id"] == "ORD-7"
    assert b["order"]["status"] == "shipped"
    assert b["raw"].startswith("Order ORD-7")


def test_order_status_block_falls_back_to_raw_when_unresolvable():
    s = _state(
        step_results=[StepResult(sop=ids.ORDER_STATUS, details={"raw": "RCPT-9000: $59.98 (2026-06-01)"})]
    )
    b = build_blocks(s)[0]
    assert b["kind"] == "order_status"
    assert b["order"] is None
    assert "RCPT-9000" in b["raw"]


def test_checkout_block_from_cart():
    svc = CartService()
    svc.add_item("P-1")
    s = _state(
        cart=svc.cart,
        step_results=[StepResult(sop=ids.CHECKOUT, asks=["first name"], details={"step": "collecting_identity"})],
    )
    b = build_blocks(s)[0]
    assert b["kind"] == "checkout"
    assert b["items"][0]["id"] == "P-1"
    assert b["asks"] == ["first name"]
    assert b["confirmed"] is False


def test_compound_turn_yields_blocks_in_order():
    s = _state(
        step_results=[
            StepResult(sop=ids.PRODUCT_REC, details={"products": [{"id": "P-2", "name": "Hoodie", "price": "49.99", "tags": []}]}),
            StepResult(sop=ids.ORDER_STATUS, details={"raw": "Order ORD-9 is delivered, items=[P-2]"}),
        ]
    )
    kinds = [b["kind"] for b in build_blocks(s)]
    assert kinds == ["product_reco", "order_status"]


def test_checkout_block_built_once_even_if_checkout_runs_twice():
    svc = CartService()
    svc.add_item("P-1")
    s = _state(
        cart=svc.cart,
        step_results=[
            StepResult(sop=ids.CHECKOUT, details={"step": "collecting_identity"}),
            StepResult(sop=ids.CHECKOUT, details={"step": "collecting_address"}),
        ],
    )
    checkout_blocks = [b for b in build_blocks(s) if b["kind"] == "checkout"]
    assert len(checkout_blocks) == 1
