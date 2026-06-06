"""product_rec wrapper — cart-growth detection + history passing.

These tests use a stub product_rec subagent so we don't need OpenAI.
We're testing the wrapper's behavior around the subagent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_v2.checkout import CartService
from agent_v2.state import AgentState
from agent_v2.supervisor import SOPName
from langchain_core.messages import AIMessage, HumanMessage


def _state_with_msgs(*msgs) -> AgentState:
    return AgentState(user_id="u", session_id="s", messages=list(msgs))


def _make_stub_agent(side_effect_on_cart=None):
    """Return a stand-in for _PRODUCT_REC_AGENT.

    If side_effect_on_cart is provided, it's called with the cart_service
    inside the simulated invoke (mimicking the subagent calling add_item).
    """

    class Stub:
        def stream(self, payload, config=None, context=None, stream_mode=None):
            if side_effect_on_cart is not None:
                side_effect_on_cart(context.cart_service)
            yield ("values", {"messages": [AIMessage(content="ok")]})

        def invoke(self, payload, context=None, **_kw):
            if side_effect_on_cart is not None:
                side_effect_on_cart(context.cart_service)
            return {"messages": [AIMessage(content="ok")]}

    return Stub()


def test_wrapper_detects_cart_growth_and_hints_checkout():
    """When the subagent calls add_item, the wrapper must set next_sop=CHECKOUT."""
    import sys

    import agent_v2.graph  # noqa: F401 — ensure submodule is in sys.modules

    graph_mod = sys.modules["agent_v2.graph"]

    state = _state_with_msgs(HumanMessage(content="add P-3"))

    def _add_p3(service):
        service.add_item("P-3")

    with patch.object(graph_mod, "_PRODUCT_REC_AGENT", _make_stub_agent(_add_p3)):
        cmd = graph_mod.product_rec_wrapper(state)

    assert cmd.goto == "supervisor"
    sr = cmd.update["step_results"][0]
    assert sr.sop == SOPName.PRODUCT_REC
    assert sr.next_sop == SOPName.CHECKOUT
    assert "P-3" in sr.summary
    assert sr.details["added"] == ["P-3"]
    # The mutated cart is propagated outward.
    assert cmd.update["cart"].items
    assert cmd.update["cart"].items[0].product_id == "P-3"


def test_wrapper_does_not_hint_checkout_when_no_cart_growth():
    """If the subagent just searched but didn't add anything, stay in product_rec."""
    import sys

    import agent_v2.graph  # noqa: F401 — ensure submodule is in sys.modules

    graph_mod = sys.modules["agent_v2.graph"]

    state = _state_with_msgs(HumanMessage(content="show me shoes"))
    with patch.object(graph_mod, "_PRODUCT_REC_AGENT", _make_stub_agent(None)):
        cmd = graph_mod.product_rec_wrapper(state)

    sr = cmd.update["step_results"][0]
    assert sr.next_sop is None


def test_wrapper_passes_recent_history_not_just_last_message():
    """The subagent must receive multi-turn history so it can resolve 'them'."""
    import sys

    import agent_v2.graph  # noqa: F401 — ensure submodule is in sys.modules

    graph_mod = sys.modules["agent_v2.graph"]

    state = _state_with_msgs(
        HumanMessage(content="show me shoes"),
        AIMessage(content="Found P-3: Running Sneakers — $89.00."),
        HumanMessage(content="add them"),
    )

    captured: dict = {}

    class CapturingStub:
        def stream(self, payload, config=None, context=None, stream_mode=None):
            captured["messages"] = payload["messages"]
            yield ("values", {"messages": [AIMessage(content="done")]})

        def invoke(self, payload, context=None, **_kw):
            captured["messages"] = payload["messages"]
            return {"messages": [AIMessage(content="done")]}

    with patch.object(graph_mod, "_PRODUCT_REC_AGENT", CapturingStub()):
        graph_mod.product_rec_wrapper(state)

    # All 3 turns reach the subagent.
    msg_contents = [str(m.content) for m in captured["messages"]]
    assert "show me shoes" in msg_contents
    assert "Found P-3: Running Sneakers — $89.00." in msg_contents
    assert "add them" in msg_contents


def test_wrapper_handles_multiple_items_added_in_one_turn():
    """If the subagent adds P-3 AND P-4 in the same turn, both show up."""
    import sys

    import agent_v2.graph  # noqa: F401 — ensure submodule is in sys.modules

    graph_mod = sys.modules["agent_v2.graph"]

    state = _state_with_msgs(HumanMessage(content="add p3 and p4"))

    def _add_two(service):
        service.add_item("P-3")
        service.add_item("P-4")

    with patch.object(graph_mod, "_PRODUCT_REC_AGENT", _make_stub_agent(_add_two)):
        cmd = graph_mod.product_rec_wrapper(state)

    sr = cmd.update["step_results"][0]
    assert sr.next_sop == SOPName.CHECKOUT
    assert set(sr.details["added"]) == {"P-3", "P-4"}


def test_wrapper_starts_from_existing_cart_items():
    """A cart with pre-existing items should detect ONLY the newly added ones."""
    import sys

    import agent_v2.graph  # noqa: F401 — ensure submodule is in sys.modules

    graph_mod = sys.modules["agent_v2.graph"]

    svc = CartService()
    svc.add_item("P-1")  # pre-existing
    state = _state_with_msgs(HumanMessage(content="now add P-2"))
    state = state.model_copy(update={"cart": svc.cart})

    def _add_p2(service):
        service.add_item("P-2")

    with patch.object(graph_mod, "_PRODUCT_REC_AGENT", _make_stub_agent(_add_p2)):
        cmd = graph_mod.product_rec_wrapper(state)

    sr = cmd.update["step_results"][0]
    # Only the newly-added item is in `added`.
    assert sr.details["added"] == ["P-2"]
