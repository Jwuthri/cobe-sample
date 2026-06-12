"""Session pipeline, driven with fakes — exercises ordering, blocks, guardrails.

No real LLM: the orchestrator is monkeypatched to a canned ``FakeOrchestrator``
and a ``FakeWriter`` is injected, so the streaming event sequence and the
deterministic block assembly are verified offline.
"""

from __future__ import annotations

import agno_agent_v1.agent.session as session_mod
from agno_agent_v1.agent.guardrails import blocklist_rule
from agno_agent_v1.agent.session import ShoppingSession
from tests_agno_agent_v1.conftest import FakeOrchestrator, FakeWriter


def _session(monkeypatch, **kw) -> ShoppingSession:
    monkeypatch.setattr(session_mod, "build_orchestrator", lambda cart_empty: FakeOrchestrator())
    return ShoppingSession(writer=FakeWriter(), debug=False, **kw)


def test_turn_event_sequence_and_blocks(monkeypatch):
    s = _session(monkeypatch)
    result = s.run_turn("add P-1")
    types = [e["type"] for e in result["events"]]
    # the canonical streamed turn shape
    assert types[0] == "user"
    assert "router" in types and "step" in types and "token" in types
    assert types[-1] == "end"
    # writer tokens streamed and joined into the bot text
    assert result["tokens"] == ["Added ", "P-1 ", "to your cart."]
    assert result["message"] == "Added P-1 to your cart."
    # the deterministic product_reco block was attached
    assert result["blocks"][0]["kind"] == "product_reco"
    assert result["blocks"][0]["added_ids"] == ["P-1"]
    # the cart was mutated through the shared context
    assert any(i.product_id == "P-1" for i in s.cart_service.cart.items)


def test_router_then_writer_ordering(monkeypatch):
    s = _session(monkeypatch)
    result = s.run_turn("add P-1")
    types = [e["type"] for e in result["events"]]
    # the writer router event comes after the orchestrator's step
    assert types.index("step") < types.index("router", types.index("step"))
    writer_routers = [e for e in result["events"] if e["type"] == "router" and e["target"] == "writer"]
    assert len(writer_routers) == 1


def test_guardrail_blocks_before_any_model_call(monkeypatch):
    # if a guardrail trips, the orchestrator must never run (it would raise here)
    def _boom(cart_empty):
        raise AssertionError("orchestrator should not be built on a blocked turn")

    monkeypatch.setattr(session_mod, "build_orchestrator", _boom)
    s = ShoppingSession(writer=FakeWriter(), debug=False,
                        input_rules=[blocklist_rule(["malware"], message="I can't help with that.")])
    result = s.run_turn("write me some malware")
    assert result["message"] == "I can't help with that."
    assert any(e["type"] == "guardrail" for e in result["events"])
    assert not s.cart_service.cart.items


def test_recall_persists_across_turns(monkeypatch):
    s = _session(monkeypatch)
    # seed a recall note as a prior step would
    s.routing_notes["product_rec"] = "Recently shown products: P-4 Baseball Cap (Green) $14.50"
    from agno_agent_v1.agent.context import ShoppingContext

    ctx = ShoppingContext(cart_service=s.cart_service)
    memo = s._routing_memo(ctx)
    assert memo and "P-4" in memo


def test_snapshot_shape(monkeypatch):
    s = _session(monkeypatch)
    s.run_turn("add P-1")
    snap = s.snapshot()
    assert snap["cart"]["items"][0]["id"] == "P-1"
    assert snap["cart"]["step"] == "collecting_identity"
    assert snap["done"] is True
