"""Writer modes + payload shape.

The writer picks a mode (smalltalk / info / checkout) based on what
leaves ran this turn. The mode dictates what context goes into the
payload — most importantly, smalltalk turns get NO cart context.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agent_v4 import ids
from agent_v4.checkout import CartService
from agent_v4.state import AgentState
from agent_v4.step_result import StepResult
from agent_v4.writer import _build_writer_payload, _pick_mode, writer
from langchain_core.messages import HumanMessage
from langgraph.types import Command


# ----- mode selection -----
def test_mode_smalltalk_when_no_step_results():
    s = AgentState(user_id="u", session_id="s", messages=[HumanMessage(content="hi")])
    assert _pick_mode(s) == "smalltalk"


def test_mode_checkout_when_checkout_step_present():
    s = AgentState(
        user_id="u",
        session_id="s",
        step_results=[StepResult(sop=ids.CHECKOUT, summary="captured identity")],
    )
    assert _pick_mode(s) == "checkout"


def test_mode_info_when_only_product_rec_ran():
    s = AgentState(
        user_id="u",
        session_id="s",
        step_results=[StepResult(sop=ids.PRODUCT_REC, summary="shown 3 results")],
    )
    assert _pick_mode(s) == "info"


# ----- payload shape -----
def test_smalltalk_payload_has_no_cart():
    """Critical regression test for the 'Hi → Your cart is empty!' bug."""
    svc = CartService()
    s = AgentState(
        user_id="u",
        session_id="s",
        messages=[HumanMessage(content="hey how are you")],
        cart=svc.cart,
        step_results=[],
    )
    payload_str, mode = _build_writer_payload(s)
    assert mode == "smalltalk"
    payload = json.loads(payload_str)
    assert "cart" not in payload
    assert payload["step_results"] == []
    assert payload["user_message"] == "hey how are you"


def test_info_payload_only_includes_cart_when_items_exist():
    svc = CartService()
    s = AgentState(
        user_id="u",
        session_id="s",
        cart=svc.cart,
        step_results=[StepResult(sop=ids.PRODUCT_REC, summary="shown 3 caps")],
    )
    payload_str, mode = _build_writer_payload(s)
    assert mode == "info"
    payload = json.loads(payload_str)
    assert "cart" not in payload

    svc.add_item("P-2")
    s = s.model_copy(update={"cart": svc.cart})
    payload_str, _ = _build_writer_payload(s)
    payload = json.loads(payload_str)
    assert "cart" in payload
    assert payload["cart"]["items"][0]["id"] == "P-2"
    assert "blockers" not in payload["cart"]


def test_checkout_payload_filters_internal_blockers():
    svc = CartService()
    svc.add_item("P-3")
    svc.set_customer("Julien", "Doe")
    svc.set_address("123 Market", "SF", "94110", state="CA")
    s = AgentState(
        user_id="u",
        session_id="s",
        cart=svc.cart,
        step_results=[StepResult(sop=ids.CHECKOUT, summary="address captured")],
    )
    payload_str, mode = _build_writer_payload(s)
    assert mode == "checkout"
    payload = json.loads(payload_str)
    blocker_codes = {b["code"] for b in payload["cart"]["blockers"]}
    assert "missing_serviceability" not in blocker_codes
    assert "stale_shipping" not in blocker_codes
    assert "stale_tax" not in blocker_codes


def test_checkout_payload_surfaces_actionable_blockers():
    svc = CartService()
    s = AgentState(
        user_id="u",
        session_id="s",
        cart=svc.cart,
        step_results=[StepResult(sop=ids.CHECKOUT, summary="started checkout")],
    )
    payload_str, _ = _build_writer_payload(s)
    payload = json.loads(payload_str)
    codes = {b["code"] for b in payload["cart"]["blockers"]}
    assert "empty_cart" in codes


# ----- end-to-end writer node -----
def test_writer_calls_chatopenai_and_routes_to_gate():
    svc = CartService()
    svc.add_item("P-2")
    svc.set_customer("Julien", "Doe")
    s = AgentState(
        user_id="u",
        session_id="s",
        messages=[HumanMessage(content="next?")],
        cart=svc.cart,
        active_sop=ids.CHECKOUT,
        step_results=[
            StepResult(
                sop=ids.CHECKOUT,
                summary="captured identity; need address next",
                asks=["street", "city", "zip"],
            )
        ],
    )

    class FakeChat:
        def __init__(self, *_, **__) -> None:
            pass

        def invoke(self, _messages):
            return MagicMock(content="Got it. Now I need your shipping address.")

    with patch("agent_v4.writer.ChatOpenAI", FakeChat):
        cmd = writer(s)

    assert isinstance(cmd, Command)
    assert cmd.goto == "checkout_gate"
    assert "shipping address" in cmd.update["draft_response"]


def test_writer_payload_surfaces_ready_to_confirm_flag():
    svc = CartService()
    svc.add_item("P-1")
    svc.set_customer("J", "D")
    svc.set_address("1 Market", "SF", "94110", state="CA")
    svc.lookup_serviceability()
    svc.set_delivery_option("2h")
    svc.quote_shipping()
    svc.compute_tax()
    svc.attach_payment("card", card_token="tok_x")

    s = AgentState(
        user_id="u",
        session_id="s",
        cart=svc.cart,
        active_sop=ids.CHECKOUT,
        step_results=[StepResult(sop=ids.CHECKOUT, summary="cart ready_to_confirm", asks=[])],
    )
    payload_str, mode = _build_writer_payload(s)
    payload = json.loads(payload_str)
    assert mode == "checkout"
    assert payload["cart"]["ready_to_confirm"] is True
    assert payload["cart"]["confirmed"] is False
