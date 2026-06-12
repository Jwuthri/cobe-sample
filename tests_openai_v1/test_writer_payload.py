"""The writer payload: mode selection + grounded cart facts."""

from __future__ import annotations

import json

from openai_agent_v1.core.messages import ai, human
from openai_agent_v1.core.step_result import StepResult
from openai_agent_v1.shopping.domain import CartService
from openai_agent_v1.shopping.writer_payload import build_writer_payload, pick_mode


def test_pick_mode():
    assert pick_mode([]) == "smalltalk"
    assert pick_mode([StepResult(sop="product_rec")]) == "info"
    assert pick_mode([StepResult(sop="order_status")]) == "info"
    assert pick_mode([StepResult(sop="checkout")]) == "checkout"
    # checkout wins over info if both ran
    assert pick_mode([StepResult(sop="product_rec"), StepResult(sop="checkout")]) == "checkout"


def test_payload_excludes_recall_and_includes_history():
    msgs = [human("hi"), ai("hello"), human("show hoodies")]
    sr = StepResult(sop="product_rec", summary="s", recall="SECRET RECALL")
    pj, mode = build_writer_payload(msgs, [sr], CartService().cart)
    payload = json.loads(pj)
    assert mode == "info"
    assert payload["user_message"] == "show hoodies"
    assert "SECRET RECALL" not in pj  # recall is excluded from the writer
    assert payload["recent_conversation"][0] == {"role": "user", "content": "hi"}


def test_checkout_payload_has_cart_and_blockers():
    svc = CartService()
    svc.add_item("P-2", 1)
    pj, mode = build_writer_payload([human("checkout")], [StepResult(sop="checkout")], svc.cart)
    payload = json.loads(pj)
    assert mode == "checkout"
    assert payload["cart"]["items"][0]["id"] == "P-2"
    codes = {b["code"] for b in payload["cart"]["blockers"]}
    assert "missing_identity" in codes  # user-actionable blocker surfaced


def test_info_payload_omits_empty_cart():
    pj, _ = build_writer_payload([human("hi")], [StepResult(sop="product_rec")], CartService().cart)
    assert "cart" not in json.loads(pj)
