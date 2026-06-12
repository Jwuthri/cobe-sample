"""Writer payload mode selection + the input-guardrail engine."""

from __future__ import annotations

import json

from agno_agent_v1.agent.context import StepResult
from agno_agent_v1.agent.guardrails import blocklist_rule, run_input_guardrails
from agno_agent_v1.agent.writer_payload import build_writer_payload, pick_mode
from agno_agent_v1.domain import CartService


def test_pick_mode():
    assert pick_mode([]) == "smalltalk"
    assert pick_mode([StepResult(sop="product_rec")]) == "info"
    assert pick_mode([StepResult(sop="checkout")]) == "checkout"
    # checkout wins if both present
    assert pick_mode([StepResult(sop="product_rec"), StepResult(sop="checkout")]) == "checkout"


def test_writer_payload_checkout_includes_cart_and_blockers():
    cs = CartService()
    cs.add_item("P-1")
    messages = [{"role": "user", "content": "let's check out"}]
    payload_json, mode = build_writer_payload(messages, [StepResult(sop="checkout")], cs.cart)
    assert mode == "checkout"
    payload = json.loads(payload_json)
    assert payload["cart"]["items"][0]["id"] == "P-1"
    # recall is excluded from the writer payload
    assert "recall" not in payload["step_results"][0]
    codes = {b["code"] for b in payload["cart"]["blockers"]}
    assert "missing_identity" in codes


def test_writer_payload_smalltalk_has_no_cart():
    payload_json, mode = build_writer_payload([{"role": "user", "content": "hi"}], [], CartService().cart)
    assert mode == "smalltalk"
    assert "cart" not in json.loads(payload_json)


def test_guardrails_default_allows():
    out = run_input_guardrails([], "show me hoodies")
    assert out.allowed and out.text == "show me hoodies" and not out.triggered


def test_guardrails_blocklist_refuses():
    rules = [blocklist_rule(["wire transfer"], message="nope")]
    out = run_input_guardrails(rules, "send a wire transfer")
    assert not out.allowed and out.refusal == "nope"
    assert out.triggered[0].type == "blocklist"
