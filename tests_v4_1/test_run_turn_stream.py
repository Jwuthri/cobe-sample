"""End-to-end run_turn_stream: event order, token=bot text, blocked-input path."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from agent_v4_1.core.config import GuardrailSpec
from agent_v4_1.core.guardrails import compile_input_rules
from agent_v4_1.core.middleware import log_tool_calls
from agent_v4_1.core.step_result import StepResult
from agent_v4_1.shopping.context import ShoppingContext
from agent_v4_1.shopping.session import ShoppingSession


def _stub_product_rec():
    @tool("product_rec")
    def _t(query: str, runtime: ToolRuntime[ShoppingContext] = None) -> str:
        """stub"""
        runtime.context.step_results.append(
            StepResult(
                sop="product_rec",
                summary="added P-1 to cart",
                details={
                    "added": ["P-1"],
                    "products": [{"id": "P-1", "name": "Tee", "price": "19.99", "tags": ["x"]}],
                },
                next_sop="checkout",
            )
        )
        return "added P-1 to cart"

    return _t


def _happy_session():
    from tests_v4_1.conftest import ToolCallingFake

    orchestrator = create_agent(
        model=ToolCallingFake(
            messages=iter(
                [
                    AIMessage(content="", tool_calls=[{"name": "product_rec", "args": {"query": "tee"}, "id": "1"}]),
                    AIMessage(content="DONE"),
                ]
            )
        ),
        tools=[_stub_product_rec()],
        system_prompt="route",
        context_schema=ShoppingContext,
        middleware=[log_tool_calls("orch"), ToolCallLimitMiddleware(run_limit=4, exit_behavior="end")],
    )
    writer = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="Added the tee for you.")])),
        tools=[],
        system_prompt="w",
    )
    return ShoppingSession(orchestrator=orchestrator, writer=writer)


def test_happy_path_event_order_and_blocks():
    session = _happy_session()
    res = session.run_turn("add a tee")
    types = [e["type"] for e in res["events"]]

    # exact skeleton (collapse the run of tokens to one marker for comparison)
    collapsed = []
    for t in types:
        if t == "token" and collapsed and collapsed[-1] == "token":
            continue
        collapsed.append(t)
    assert collapsed == [
        "user",
        "state",
        "router",
        "agent",
        "step",
        "state",
        "router",
        "token",
        "writer",
        "bot",
        "state",
        "end",
    ]

    assert "".join(res["tokens"]) == res["message"] == "Added the tee for you."
    # deterministic block built from step_results.details (verbatim id/price)
    assert res["blocks"] == [
        {
            "kind": "product_reco",
            "products": [{"id": "P-1", "name": "Tee", "price": "19.99", "tags": ["x"]}],
            "added_ids": ["P-1"],
            "serviceability": None,
        }
    ]


def test_blocked_input_short_circuits_before_orchestrator():
    class _ExplodingOrchestrator:
        def astream(self, *a, **k):
            raise AssertionError("orchestrator must not run on blocked input")

    writer = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="unused")])),
        tools=[],
        system_prompt="w",
    )
    session = ShoppingSession(
        orchestrator=_ExplodingOrchestrator(),
        writer=writer,
        input_rules=compile_input_rules(
            [GuardrailSpec(type="blocklist", message="No legal advice.", params={"phrases": ["sue"]})]
        ),
    )
    res = session.run_turn("can I sue them?")
    types = [e["type"] for e in res["events"]]
    assert types == ["user", "state", "guardrail", "bot", "state", "end"]
    assert res["message"] == "No legal advice."
    assert all(e["type"] != "error" for e in res["events"])  # orchestrator never ran
