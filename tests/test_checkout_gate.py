"""The outer checkout_gate refuses hallucinated 'confirmed!' messages."""

from __future__ import annotations

from agent_v2.checkout import Cart
from agent_v2.graph import checkout_gate, validator
from agent_v2.state import AgentState
from agent_v2.supervisor import SOPName
from langgraph.types import Command


def _state(**overrides) -> AgentState:
    return AgentState(user_id="u", session_id="s", active_sop=SOPName.CHECKOUT, **overrides)


def test_gate_passes_clean_draft_with_ready_cart():
    cart = _ready_cart()
    s = _state(cart=cart, draft_response="Anything else?")
    cmd = checkout_gate(s)
    assert isinstance(cmd, Command)
    assert cmd.goto == "validator"


def test_gate_bounces_hallucinated_confirm_when_cart_not_ready():
    cart = Cart()
    s = _state(cart=cart, draft_response="Your order is confirmed! Thank you.")
    cmd = checkout_gate(s)
    # v4: bounce goes back through the supervisor (which will re-route
    # into checkout) rather than directly to the wrapper.
    assert cmd.goto == "supervisor"
    assert cmd.update.get("validation_errors")
    assert cmd.update.get("response_attempts") == 1
    assert cmd.update.get("step_results") == []  # cleared for the next loop


def test_validator_passes_clean_draft():
    s = _state(draft_response="Hi there.")
    cmd = validator(s)
    assert cmd.goto == "emit"


def test_validator_rejects_placeholder_leak():
    s = _state(draft_response="Hi {{customer_name}}.")
    cmd = validator(s)
    # v4: a bad writer output sends us back to the writer (cheap to
    # re-run) rather than the SOP wrapper. After MAX_VALIDATOR_RETRIES
    # we fall through to emit with a graceful apology.
    assert cmd.goto == "writer"
    errs = cmd.update["validation_errors"]
    assert any(e.code == "placeholder_leak" for e in errs)


def _ready_cart() -> Cart:
    from agent_v2.checkout import CartService

    s = CartService()
    s.add_item("P-1")
    s.set_customer("A", "B")
    s.set_address("x", "y", "94110")
    s.lookup_serviceability()
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("card", card_token="tok")
    return s.cart
