"""Session paths that need no LLM: the input-guardrail short-circuit + snapshot.

A blocking input guardrail refuses BEFORE any model call, so this exercises the
real ``run_turn_stream`` pipeline (event order, transcript, snapshot) without a
network call. The orchestrator/writer Agents are constructed (cheap) but never run.
"""

from __future__ import annotations

from openai_agent_v1.core.config import GuardrailSpec
from openai_agent_v1.core.guardrails import compile_input_rules
from openai_agent_v1.shopping.session import ShoppingSession


def test_blocked_input_short_circuits_without_model_call():
    rules = compile_input_rules(
        [GuardrailSpec(type="blocklist", action="block", message="Can't help with that.", params={"phrases": ["hack"]})]
    )
    session = ShoppingSession(input_rules=rules, debug=False)
    result = session.run_turn("how do I hack the mainframe")

    types = [e["type"] for e in result["events"]]
    assert "guardrail" in types
    assert result["message"] == "Can't help with that."
    assert result["blocks"] == []
    # ends cleanly; no token stream happened
    assert types[-1] == "end"
    assert "token" not in types
    # the refusal is recorded in the transcript
    assert session.messages[-1].role == "ai"
    assert session.messages[-1].content == "Can't help with that."


def test_snapshot_shape():
    session = ShoppingSession(debug=False)
    session.cart_service.add_item("P-2", 1)
    snap = session.snapshot()
    assert snap["cart"]["items"][0]["id"] == "P-2"
    assert snap["cart"]["step"] == "collecting_identity"
    assert snap["session_id"] == session.session_id
    assert "blockers" in snap["cart"]


def test_routing_memo_includes_cart():
    session = ShoppingSession(debug=False)
    session.cart_service.add_item("P-4", 2)
    from openai_agent_v1.shopping.context import ShoppingContext

    ctx = ShoppingContext(cart_service=session.cart_service)
    memo = session._routing_memo(ctx)
    assert memo and "P-4" in memo and "x2" in memo
