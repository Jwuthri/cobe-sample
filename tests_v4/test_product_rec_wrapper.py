"""product_rec wrapper — cart-growth detection + history passing.

v4 builds the wrapper from a factory that closes over the compiled leaf
agent, so we test it by passing a stub agent directly (no singleton to
patch, no OpenAI). The stub supports both ``.stream`` (the primary path)
and ``.invoke`` (the fallback).
"""

from __future__ import annotations

from agent_v4 import ids
from agent_v4.checkout import CartService
from agent_v4.leaves import make_product_rec_wrapper
from agent_v4.state import AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _state_with_msgs(*msgs) -> AgentState:
    return AgentState(user_id="u", session_id="s", messages=list(msgs))


class StubAgent:
    """Stand-in for a compiled product_rec leaf."""

    def __init__(self, on_cart=None, messages=None, capture: dict | None = None):
        self.on_cart = on_cart
        self.messages = messages if messages is not None else [AIMessage(content="ok")]
        self.capture = capture

    def stream(self, payload, config=None, context=None, stream_mode=None):
        if self.capture is not None:
            self.capture["messages"] = payload["messages"]
        if self.on_cart is not None and context is not None:
            self.on_cart(context.cart_service)
        yield ("values", {"messages": self.messages})

    def invoke(self, payload, config=None, context=None):
        if self.capture is not None:
            self.capture["messages"] = payload["messages"]
        if self.on_cart is not None and context is not None:
            self.on_cart(context.cart_service)
        return {"messages": self.messages}


def test_wrapper_detects_cart_growth_and_hints_checkout():
    state = _state_with_msgs(HumanMessage(content="add P-3"))
    wrapper = make_product_rec_wrapper(StubAgent(on_cart=lambda s: s.add_item("P-3")))
    cmd = wrapper(state)

    assert cmd.goto == "supervisor"
    sr = cmd.update["step_results"][0]
    assert sr.sop == ids.PRODUCT_REC
    assert sr.next_sop == ids.CHECKOUT
    assert "P-3" in sr.summary
    assert sr.details["added"] == ["P-3"]
    assert cmd.update["cart"].items
    assert cmd.update["cart"].items[0].product_id == "P-3"


def test_wrapper_does_not_hint_checkout_when_no_cart_growth():
    state = _state_with_msgs(HumanMessage(content="show me shoes"))
    wrapper = make_product_rec_wrapper(StubAgent(on_cart=None))
    cmd = wrapper(state)
    sr = cmd.update["step_results"][0]
    assert sr.next_sop is None


def test_wrapper_passes_recent_history_not_just_last_message():
    state = _state_with_msgs(
        HumanMessage(content="show me shoes"),
        AIMessage(content="Found P-3: Running Sneakers — $89.00."),
        HumanMessage(content="add them"),
    )
    capture: dict = {}
    wrapper = make_product_rec_wrapper(StubAgent(capture=capture))
    wrapper(state)

    msg_contents = [str(m.content) for m in capture["messages"]]
    assert "show me shoes" in msg_contents
    assert "Found P-3: Running Sneakers — $89.00." in msg_contents
    assert "add them" in msg_contents


def test_wrapper_handles_multiple_items_added_in_one_turn():
    state = _state_with_msgs(HumanMessage(content="add p3 and p4"))

    def _add_two(service):
        service.add_item("P-3")
        service.add_item("P-4")

    wrapper = make_product_rec_wrapper(StubAgent(on_cart=_add_two))
    cmd = wrapper(state)
    sr = cmd.update["step_results"][0]
    assert sr.next_sop == ids.CHECKOUT
    assert set(sr.details["added"]) == {"P-3", "P-4"}


def test_wrapper_starts_from_existing_cart_items():
    svc = CartService()
    svc.add_item("P-1")  # pre-existing
    state = _state_with_msgs(HumanMessage(content="now add P-2"))
    state = state.model_copy(update={"cart": svc.cart})

    wrapper = make_product_rec_wrapper(StubAgent(on_cart=lambda s: s.add_item("P-2")))
    cmd = wrapper(state)
    sr = cmd.update["step_results"][0]
    assert sr.details["added"] == ["P-2"]
