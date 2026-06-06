"""Supervisor classifier uses recent conversation history.

We don't hit OpenAI in tests — we verify the *prompt construction*
includes the right context (history + step_results + cart_step).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_v2.state import AgentState
from agent_v2.step_result import StepResult
from agent_v2.supervisor import (
    SOPName,
    SupervisorDecision,
    _format_history,
    _format_step_results,
    classify_with_history,
)
from langchain_core.messages import AIMessage, HumanMessage


def test_format_history_renders_roles():
    msgs = [
        HumanMessage(content="i'm looking for a hat"),
        AIMessage(content="here are 3 caps"),
        HumanMessage(content="P-4 please"),
    ]
    out = _format_history(msgs)
    assert "USER: i'm looking for a hat" in out
    assert "ASSISTANT: here are 3 caps" in out
    assert "USER: P-4 please" in out


def test_format_step_results_includes_next_sop_hint():
    rs = [
        StepResult(sop=SOPName.PRODUCT_REC, summary="added P-4 to cart", next_sop=SOPName.CHECKOUT),
        StepResult(sop=SOPName.CHECKOUT, summary="asked for identity", asks=["first/last name"]),
    ]
    out = _format_step_results(rs)
    assert "next_sop=checkout" in out
    assert "asks=" in out


def test_classify_payload_includes_history_and_cart_step():
    """The classifier's user payload contains conversation history,
    cart step, and step_results — so the LLM has everything it needs."""
    state = AgentState(
        user_id="u",
        session_id="s",
        messages=[
            HumanMessage(content="what is my zip?"),
            AIMessage(content="please provide your zip"),
            HumanMessage(content="94110"),
        ],
        step_results=[
            StepResult(sop=SOPName.CHECKOUT, summary="asked for zip", asks=["zip"]),
        ],
    )

    captured: dict = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def parse(model, messages, response_format):
                    captured["messages"] = messages
                    return MagicMock(
                        choices=[
                            MagicMock(
                                message=MagicMock(
                                    parsed=SupervisorDecision(
                                        done=False,
                                        next_sop=SOPName.CHECKOUT,
                                        reason="mid-flow",
                                    )
                                )
                            )
                        ]
                    )

    with patch("agent_v2.supervisor.classifier_client", return_value=FakeClient()):
        result = classify_with_history(state)

    assert result.next_sop == SOPName.CHECKOUT
    user_text = captured["messages"][1]["content"]
    assert "USER: 94110" in user_text  # latest turn included
    assert "USER: what is my zip?" in user_text  # prior turn included
    assert "Cart step:" in user_text
    assert "Step results this turn:" in user_text
    assert "asked for zip" in user_text


def test_classifier_falls_back_to_product_rec_on_no_choice():
    """Defensive: when classifier returns done=False but no next_sop,
    the supervisor still picks something valid (PRODUCT_REC)."""
    state = AgentState(
        user_id="u",
        session_id="s",
        messages=[HumanMessage(content="???")],
    )

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def parse(model, messages, response_format):
                    return MagicMock(
                        choices=[
                            MagicMock(
                                message=MagicMock(
                                    parsed=SupervisorDecision(done=False, next_sop=None)
                                )
                            )
                        ]
                    )

    with patch("agent_v2.supervisor.classifier_client", return_value=FakeClient()):
        result = classify_with_history(state)
    assert result.next_sop == SOPName.PRODUCT_REC
