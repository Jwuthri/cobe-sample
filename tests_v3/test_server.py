"""The v3 FastAPI server preserves the v2 SSE contract (stubbed agents)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import agent_v3.workflow as wf
from agent_v3.sop_names import SOPName, SupervisorDecision


def _tool(name, result):
    return SimpleNamespace(tool_name=name, tool_args={}, result=result)


@pytest.fixture
def client(monkeypatch):
    class Stub:
        def __init__(self, beh):
            self.beh = beh

        def run(self, input=None, session_state=None, dependencies=None, **kw):
            return SimpleNamespace(
                tools=self.beh(input, (dependencies or {}).get("cart_service")) or [],
                content="",
                messages=[],
            )

    monkeypatch.setattr(
        wf, "_PRODUCT_REC_AGENT", Stub(lambda i, cs: (cs.add_item("P-1", 1), [_tool("add_item", "added")])[1])
    )
    monkeypatch.setattr(wf, "_CHECKOUT_AGENT", Stub(lambda i, cs: []))
    monkeypatch.setattr(wf, "generate_draft", lambda ss, cart, c=None: "Added P-1. What's your name?")
    monkeypatch.setattr(
        "agent_v3.supervisor.classify_with_history",
        lambda ss: SupervisorDecision(done=False, next_sop=SOPName.PRODUCT_REC),
    )
    import server.main_v3 as srv

    return TestClient(srv.app)


def test_session_state_and_turn_contract(client):
    sid = client.post("/api/session").json()["session_id"]

    snap = client.get(f"/api/state/{sid}").json()
    assert {"user_id", "session_id", "active_sop", "skills_loaded", "cart", "messages", "iteration", "done"} <= set(snap)
    assert {"step", "items", "blockers", "ready_to_confirm", "subtotal", "grand_total"} <= set(snap["cart"])

    with client.stream("POST", f"/api/turn/{sid}", json={"message": "add P-1 to my cart"}) as r:
        assert r.status_code == 200
        frames = [json.loads(line[5:].strip()) for line in r.iter_lines() if line and line.startswith("data:")]

    types = [f["type"] for f in frames]
    for needed in ("user", "state", "router", "agent", "step", "writer", "bot", "end"):
        assert needed in types, f"missing {needed} in {types}"

    final = [f for f in frames if f["type"] == "state"][-1]["snapshot"]
    assert len(final["cart"]["items"]) == 1
    assert final["cart"]["step"] == "collecting_identity"
    assert [f for f in frames if f["type"] == "bot"][-1]["content"] == "Added P-1. What's your name?"
