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

    # exact skeleton (drop additive 'trace' rows; collapse the run of tokens)
    collapsed = []
    for t in types:
        if t == "trace":
            continue
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


def test_emits_orchestrator_and_writer_trace_frames():
    session = _happy_session()  # debug defaults True
    res = session.run_turn("add a tee")
    traces = [e for e in res["events"] if e["type"] == "trace"]
    phases = [t["phase"] for t in traces]

    assert "orchestrator_input" in phases and "writer_payload" in phases
    oi = next(t for t in traces if t["phase"] == "orchestrator_input")
    assert oi["data"]["conversation_seen"][-1]["content"] == "add a tee"
    assert oi["data"]["delegates"]  # the routable sub-agents are listed
    wp = next(t for t in traces if t["phase"] == "writer_payload")
    assert wp["data"]["mode"] in {"smalltalk", "info", "checkout"}
    assert "payload" in wp["data"] and "system_prompt" in wp["data"]


def test_debug_false_emits_no_trace_frames():
    session = _happy_session()
    session.debug = False
    res = session.run_turn("add a tee")
    assert [e for e in res["events"] if e["type"] == "trace"] == []


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


def test_user_event_carries_incrementing_turn_number():
    # blocked input short-circuits before the orchestrator, so we can run many turns
    class _NoOrchestrator:
        def astream(self, *a, **k):
            raise AssertionError("orchestrator must not run on blocked input")

    writer = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="unused")])),
        tools=[],
        system_prompt="w",
    )
    session = ShoppingSession(
        orchestrator=_NoOrchestrator(),
        writer=writer,
        input_rules=compile_input_rules(
            [GuardrailSpec(type="blocklist", message="blocked", params={"phrases": ["block"]})]
        ),
    )
    t1 = next(e for e in session.run_turn("block one")["events"] if e["type"] == "user")["turn"]
    t2 = next(e for e in session.run_turn("block two")["events"] if e["type"] == "user")["turn"]
    assert (t1, t2) == (1, 2)


def test_routing_memo_merges_live_state_and_persisted_recalls():
    # agnostic: live state comes from ctx.routing_context(), recall from routing_notes
    from agent_v4_1.shopping.domain import CartService

    cs = CartService()
    cs.add_item("P-2")
    session = ShoppingSession(
        orchestrator=object(),  # skip __post_init__ build (no real models)
        writer=object(),
        cart_service=cs,
        routing_notes={"product_rec": "Recently shown products: P-4 Green Cap $14.50"},
    )
    memo = session._routing_memo(ShoppingContext(cart_service=cs))
    assert memo is not None
    assert "P-2" in memo  # live cart, via ctx.routing_context()
    assert "P-4" in memo  # persisted per-step recall (engine never parsed it)


def test_absorb_recalls_persists_step_recall_by_sop():
    # the session is agnostic — it keys opaque recall text by sop, never inspecting it
    session = ShoppingSession(orchestrator=object(), writer=object())
    ctx = ShoppingContext()
    ctx.step_results.append(StepResult(sop="order_status", recall="Recently looked up order: ORD-9 delivered"))
    session._absorb_recalls(ctx)
    assert session.routing_notes == {"order_status": "Recently looked up order: ORD-9 delivered"}
