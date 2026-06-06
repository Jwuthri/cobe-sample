"""Writer modes + payload shape (ports tests/test_writer.py to session_state)."""

from __future__ import annotations

import json

from agent_v3.checkout import CartService
from agent_v3.sop_names import SOPName
from agent_v3.state import fresh_state, load_cart
from agent_v3.step_result import StepResult
from agent_v3.writer import _build_writer_payload, _pick_mode, generate_draft


def _ss(step_results=None, messages=None):
    s = fresh_state(user_id="u", session_id="s")
    if messages is not None:
        s["messages"] = messages
    s["step_results"] = [sr.model_dump(mode="json") for sr in (step_results or [])]
    return s


def test_mode_smalltalk_when_no_step_results():
    assert _pick_mode(_ss(messages=[{"role": "human", "content": "hi"}])) == "smalltalk"


def test_mode_checkout_when_checkout_step_present():
    s = _ss(step_results=[StepResult(sop=SOPName.CHECKOUT, summary="captured identity")])
    assert _pick_mode(s) == "checkout"


def test_mode_info_when_only_product_rec_ran():
    s = _ss(step_results=[StepResult(sop=SOPName.PRODUCT_REC, summary="shown 3")])
    assert _pick_mode(s) == "info"


def test_smalltalk_payload_has_no_cart():
    svc = CartService()
    s = _ss(messages=[{"role": "human", "content": "hey how are you"}])
    payload_str, mode = _build_writer_payload(s, svc.cart)
    assert mode == "smalltalk"
    payload = json.loads(payload_str)
    assert "cart" not in payload
    assert payload["step_results"] == []
    assert payload["user_message"] == "hey how are you"


def test_info_payload_only_includes_cart_when_items_exist():
    svc = CartService()
    s = _ss(step_results=[StepResult(sop=SOPName.PRODUCT_REC, summary="shown 3 caps")])
    payload = json.loads(_build_writer_payload(s, svc.cart)[0])
    assert "cart" not in payload
    svc.add_item("P-2")
    payload = json.loads(_build_writer_payload(s, svc.cart)[0])
    assert payload["cart"]["items"][0]["id"] == "P-2"
    assert "blockers" not in payload["cart"]


def test_checkout_payload_filters_internal_blockers():
    svc = CartService()
    svc.add_item("P-3")
    svc.set_customer("Julien", "Doe")
    svc.set_address("123 Market", "SF", "94110", state="CA")
    s = _ss(step_results=[StepResult(sop=SOPName.CHECKOUT, summary="address captured")])
    payload = json.loads(_build_writer_payload(s, svc.cart)[0])
    codes = {b["code"] for b in payload["cart"]["blockers"]}
    assert "missing_serviceability" not in codes
    assert "stale_shipping" not in codes
    assert "stale_tax" not in codes


def test_checkout_payload_surfaces_actionable_blockers():
    svc = CartService()
    s = _ss(step_results=[StepResult(sop=SOPName.CHECKOUT, summary="started checkout")])
    payload = json.loads(_build_writer_payload(s, svc.cart)[0])
    assert "empty_cart" in {b["code"] for b in payload["cart"]["blockers"]}


def test_payload_surfaces_ready_to_confirm():
    svc = CartService()
    svc.add_item("P-1")
    svc.set_customer("J", "D")
    svc.set_address("1 Market", "SF", "94110", state="CA")
    svc.lookup_serviceability()
    svc.set_delivery_option("2h")
    svc.quote_shipping()
    svc.compute_tax()
    svc.attach_payment("card", card_token="tok_x")
    s = _ss(step_results=[StepResult(sop=SOPName.CHECKOUT, summary="ready", asks=[])])
    payload = json.loads(_build_writer_payload(s, svc.cart)[0])
    assert payload["cart"]["ready_to_confirm"] is True
    assert payload["cart"]["confirmed"] is False


def test_generate_draft_uses_writer_agent(monkeypatch):
    from types import SimpleNamespace

    class FakeWriter:
        def run(self, input=None, **kw):
            return SimpleNamespace(content="Got it. Now I need your shipping address.")

    monkeypatch.setattr("agent_v3.writer._WRITER", FakeWriter())
    s = _ss(step_results=[StepResult(sop=SOPName.CHECKOUT, summary="need address", asks=["street"])])
    draft = generate_draft(s, load_cart(s))
    assert "shipping address" in draft
