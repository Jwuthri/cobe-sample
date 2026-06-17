"""Shared fixtures for tests_lg_agent_v3. No test makes a real LLM call.

The session tests drive the LangChain agents with fake chat models (scripted
``AIMessage`` sequences) injected at build time, so the whole streaming pipeline runs
deterministically and offline — the analogue of ``pydantic_agent_v1``'s
``FunctionModel`` + ``agent.override``.
"""

from __future__ import annotations

import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Deterministic CART-1000 / RCPT-9000 ids across tests."""
    monkeypatch.setattr("lg_agent_v3.domain.cart_service._CART_COUNTER", itertools.count(1000))
    monkeypatch.setattr("lg_agent_v3.domain.cart_service._RECEIPT_COUNTER", itertools.count(9000))
    yield


# --------------------------------------------------------------------------- #
# fake chat models (scripted AIMessage sequences)
# --------------------------------------------------------------------------- #
class ToolCallingFake(GenericFakeChatModel):
    """A streamable fake that also accepts ``bind_tools`` (the base raises on it).

    Script it with ``messages=iter([...])`` of AIMessages: a tool-call message drives
    the agent loop, a plain-text message ends it. ``bind_tools`` is a no-op (the script
    decides the calls) so it works for both the tool-using orchestrator/workers and the
    no-tools streaming writer.

    The turn-graph streams every nested model (``stream_mode="messages"``), so ``_stream``
    yields the scripted message as ONE chunk — the base's word-chunking emits nothing for
    an empty-content tool-call message ("No generations found in stream"), which the real
    streaming pipeline would never hit.
    """

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = next(self.messages)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_calls=getattr(message, "tool_calls", []) or [],
                id=getattr(message, "id", None),
            )
        )


def _tool_call(name: str, args: dict, idx: int) -> dict:
    return {"name": name, "args": args, "id": f"call_{idx}"}


def call_then_done(*tool_calls):
    """A model that emits the given tool calls on the first step, then 'DONE'.

    Each ``tool_call`` is ``(name, args_dict)``. After the tools return, a plain-text
    'DONE' ends the agent run (mirrors the real worker contract).
    """
    calls = [_tool_call(n, a, i) for i, (n, a) in enumerate(tool_calls)]
    return ToolCallingFake(
        messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="DONE")])
    )


def sequence(*tool_calls):
    """A model that emits ONE tool call per step (in order), then 'DONE'.

    Use when later calls depend on earlier ones (e.g. set_address then
    lookup_serviceability) so they run sequentially against the shared cart.
    """
    msgs = [AIMessage(content="", tool_calls=[_tool_call(n, a, i)]) for i, (n, a) in enumerate(tool_calls)]
    msgs.append(AIMessage(content="DONE"))
    return ToolCallingFake(messages=iter(msgs))


def say(text: str):
    """A streamable model that always replies with ``text`` (for the writer or DONE)."""
    return ToolCallingFake(messages=iter([AIMessage(content=text)]))


# --------------------------------------------------------------------------- #
# session factory — wires fake models into freshly-built agents
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_session():
    """Build a ShoppingSession whose agents are driven by the given fake models.

    Usage::

        s = make_session(orchestrator=call_then_done(...), product_rec=call_then_done(...),
                         writer=say("..."))
        out = s.run_turn("...")
    """
    from lg_agent_v3 import ShoppingSession
    from lg_agent_v3.agents import build_writer
    from lg_agent_v3.agents import checkout as co
    from lg_agent_v3.agents import order_status as os_
    from lg_agent_v3.agents import product_rec as pr
    from lg_agent_v3.agents import writer as default_writer
    from lg_agent_v3.agents.orchestrator import build_orchestrator
    from lg_agent_v3.agents.orchestrator import orchestrator as default_orch

    def _make(
        *,
        orchestrator=None,
        product_rec=None,
        checkout=None,
        order_status=None,
        writer=None,
        orchestrator_guardrails=None,
        product_rec_guardrails=None,
        checkout_guardrails=None,
        order_status_guardrails=None,
        writer_guardrails=None,
        **session_kwargs,
    ):
        worker_agents = {}
        if product_rec is not None:
            worker_agents["product_rec"] = pr.build(product_rec, guardrails=product_rec_guardrails)
        if checkout is not None:
            worker_agents["checkout"] = co.build(checkout, guardrails=checkout_guardrails)
        if order_status is not None:
            worker_agents["order_status"] = os_.build(order_status, guardrails=order_status_guardrails)

        if orchestrator is not None or worker_agents or orchestrator_guardrails:
            orch = build_orchestrator(model=orchestrator, worker_agents=worker_agents, guardrails=orchestrator_guardrails)
        else:
            orch = default_orch
        wr = build_writer(writer, guardrails=writer_guardrails) if writer is not None else default_writer

        return ShoppingSession(orchestrator_agent=orch, writer_agent=wr, **session_kwargs)

    return _make
