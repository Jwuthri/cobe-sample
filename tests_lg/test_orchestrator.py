"""Orchestrator routing + guards, driven by a scripted tool-calling fake."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from lg_agent.core.step import StepResult
from lg_agent.shopping.context import ShoppingContext
from lg_agent.shopping.domain import CartService
from lg_agent.shopping.middleware import EmptyCartGuardMiddleware


def _stub(name):
    @tool(name)
    def _t(query: str, runtime: ToolRuntime[ShoppingContext] = None) -> str:
        """stub subagent"""
        runtime.context.step_results.append(StepResult(sop=name, summary=f"{name} ran"))
        return f"{name} ran"

    return _t


def _orchestrator(scripted, tools, run_limit=4):
    from tests_lg.conftest import ToolCallingFake

    return create_agent(
        model=ToolCallingFake(messages=iter(scripted)),
        tools=tools,
        system_prompt="route",
        context_schema=ShoppingContext,
        middleware=[ToolCallLimitMiddleware(run_limit=run_limit, exit_behavior="end")],
    )


def test_routes_to_single_subagent():
    agent = _orchestrator(
        [
            AIMessage(content="", tool_calls=[{"name": "product_rec", "args": {"query": "x"}, "id": "1"}]),
            AIMessage(content="DONE"),
        ],
        [_stub("product_rec")],
    )
    ctx = ShoppingContext(cart_service=CartService())
    agent.invoke({"messages": [("user", "find a tee")]}, context=ctx)
    assert [sr.sop for sr in ctx.step_results] == ["product_rec"]


def test_multi_intent_two_tool_calls():
    agent = _orchestrator(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "product_rec", "args": {"query": "cap"}, "id": "1"},
                    {"name": "order_status", "args": {"query": "ORD-7"}, "id": "2"},
                ],
            ),
            AIMessage(content="DONE"),
        ],
        [_stub("product_rec"), _stub("order_status")],
    )
    ctx = ShoppingContext(cart_service=CartService())
    agent.invoke({"messages": [("user", "cap + ORD-7")]}, context=ctx)
    assert sorted(sr.sop for sr in ctx.step_results) == ["order_status", "product_rec"]


def test_tool_call_limit_caps_the_loop():
    # model always calls a tool, never says DONE → limit must stop it.
    loop = [
        AIMessage(content="", tool_calls=[{"name": "product_rec", "args": {"query": "x"}, "id": str(i)}])
        for i in range(10)
    ]
    agent = _orchestrator(loop, [_stub("product_rec")], run_limit=2)
    ctx = ShoppingContext(cart_service=CartService())
    agent.invoke({"messages": [("user", "loop")]}, context=ctx)
    assert len(ctx.step_results) <= 2  # capped, did not run away


def test_empty_cart_guard_strips_checkout_tool():
    guard = EmptyCartGuardMiddleware()

    class _Req:
        def __init__(self, cart_service, tools):
            self.runtime = type("R", (), {"context": type("C", (), {"cart_service": cart_service})()})()
            self.tools = tools
            self._overrides = {}

        def override(self, **kw):
            self.tools = kw.get("tools", self.tools)
            return self

    checkout_tool = type("T", (), {"name": "checkout"})()
    other = type("T", (), {"name": "product_rec"})()

    empty = _Req(CartService(), [checkout_tool, other])
    kept = guard._apply(empty)
    assert [t.name for t in kept.tools] == ["product_rec"]  # checkout removed

    cs = CartService()
    cs.add_item("P-1")
    non_empty = _Req(cs, [checkout_tool, other])
    kept2 = guard._apply(non_empty)
    assert sorted(t.name for t in kept2.tools) == ["checkout", "product_rec"]  # kept
