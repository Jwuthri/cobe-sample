"""End-to-end workflow integration with stubbed agents (no LLM).

Validates the supervisor Loop + Router, the SOP wrappers, StepResult
hand-off, cart mutation via dependencies, the compose/gate/validator
chain, and stream_turn's UI events.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent_v3.workflow as wf
from agent_v3.sop_names import SOPName, SupervisorDecision
from agent_v3.state import fresh_state, load_cart


def _tool(name, result):
    return SimpleNamespace(tool_name=name, tool_args={}, result=result)


def _stub(behavior):
    class Stub:
        def run(self, input=None, session_state=None, dependencies=None, **kw):
            tools = behavior(input, (dependencies or {}).get("cart_service")) or []
            return SimpleNamespace(tools=tools, content="", messages=[])

    return Stub()


@pytest.fixture
def stub_agents(monkeypatch):
    monkeypatch.setattr(
        wf,
        "_PRODUCT_REC_AGENT",
        _stub(lambda i, cs: (cs.add_item("P-1", 1), [_tool("add_item", "added P-1")])[1]),
    )
    monkeypatch.setattr(wf, "_CHECKOUT_AGENT", _stub(lambda i, cs: []))
    monkeypatch.setattr(
        wf,
        "_ORDER_STATUS_AGENT",
        _stub(lambda i, cs: [_tool("get_order_status", "Order ORD-7 is shipped, items=['P-1']")]),
    )
    monkeypatch.setattr(wf, "generate_draft", lambda ss, cart, c=None: "stub-draft")


def test_add_item_routes_product_rec_then_checkout(stub_agents, monkeypatch):
    monkeypatch.setattr(
        "agent_v3.supervisor.classify_with_history",
        lambda ss: SupervisorDecision(done=False, next_sop=SOPName.PRODUCT_REC),
    )
    ss = fresh_state(user_id="u", session_id="s1")
    wf.run_turn(ss, "add P-1 to my cart")
    assert [r["sop"] for r in ss["step_results"]] == ["product_rec", "checkout"]
    assert len(load_cart(ss).items) == 1
    assert ss["active_sop"] == "checkout"
    assert ss["messages"][-1] == {"role": "ai", "content": "stub-draft"}


def test_smalltalk_runs_no_sop(stub_agents):
    ss = fresh_state(user_id="u", session_id="s2")
    wf.run_turn(ss, "hi")
    assert ss["step_results"] == []
    assert ss["messages"][-1]["role"] == "ai"


def test_stream_turn_emits_ui_events(stub_agents, monkeypatch):
    monkeypatch.setattr(
        "agent_v3.supervisor.classify_with_history",
        lambda ss: SupervisorDecision(done=False, next_sop=SOPName.ORDER_STATUS),
    )
    ss = fresh_state(user_id="u", session_id="s3")
    events = list(wf.stream_turn(ss, "where is ORD-7?"))
    types = [e["type"] for e in events]
    for needed in ("router", "agent", "step", "writer", "bot"):
        assert needed in types, f"missing {needed} in {types}"
    # loop terminates cleanly: order_status + finalize == 2 router events
    assert types.count("router") == 2
    assert [e for e in events if e["type"] == "bot"][-1]["content"] == "stub-draft"


def test_gate_blocks_false_confirmation(stub_agents, monkeypatch):
    """If the draft claims 'confirmed' but the cart isn't ready, the gate
    forces a corrected regeneration (safety property preserved)."""
    monkeypatch.setattr(
        "agent_v3.supervisor.classify_with_history",
        lambda ss: SupervisorDecision(done=False, next_sop=SOPName.PRODUCT_REC),
    )
    drafts = iter(["Your order is confirmed!", "You still need to add an address."])
    monkeypatch.setattr(wf, "generate_draft", lambda ss, cart, c=None: next(drafts))
    ss = fresh_state(user_id="u", session_id="s4")
    events = list(wf.stream_turn(ss, "add P-1"))
    assert any(e["type"] == "gate" and e["rejected"] for e in events)
    # final bot message is the corrected one (not the false confirmation)
    assert ss["messages"][-1]["content"] == "You still need to add an address."
