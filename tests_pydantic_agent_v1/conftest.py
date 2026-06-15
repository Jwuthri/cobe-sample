"""Shared fixtures for tests_pydantic_agent_v1. No test makes a real LLM call.

The session tests drive the agents with Pydantic AI's ``FunctionModel`` (a scripted
fake model) via ``agent.override(...)``, so the whole streaming pipeline runs
deterministically and offline.
"""

from __future__ import annotations

import contextlib
import itertools

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

# the package re-exports orchestrator/writer as the *Agent* objects, and
# product_rec/checkout/order_status as their *modules* (use .agent).
from pydantic_agent_v1.agents import checkout, order_status, product_rec
from pydantic_agent_v1.agents import orchestrator as orchestrator_agent
from pydantic_agent_v1.agents import writer as writer_agent


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Deterministic CART-1000 / RCPT-9000 ids across tests."""
    monkeypatch.setattr("pydantic_agent_v1.domain.cart_service._CART_COUNTER", itertools.count(1000))
    monkeypatch.setattr("pydantic_agent_v1.domain.cart_service._RECEIPT_COUNTER", itertools.count(9000))
    yield


# --------------------------------------------------------------------------- #
# scripted-model helpers (build FunctionModel callables)
# --------------------------------------------------------------------------- #
def _awaiting_tool_result(messages) -> bool:
    """True if the most recent message carries a tool result (i.e. we already acted)."""
    last = messages[-1]
    return isinstance(last, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in last.parts)


def call_then_done(*tool_calls):
    """A model that emits the given tool calls on the first step, then 'DONE'.

    Each ``tool_call`` is ``(name, args_dict)``. After the tools return, emit a
    plain-text 'DONE' so the agent run completes (mirrors the real worker contract).
    """

    def fn(messages, info):
        if _awaiting_tool_result(messages):
            return ModelResponse(parts=[TextPart("DONE")])
        return ModelResponse(parts=[ToolCallPart(tool_name=n, args=a) for n, a in tool_calls])

    return fn


def sequence(*tool_calls):
    """A model that emits ONE tool call per step (in order), then 'DONE'.

    Use when later calls depend on earlier ones (e.g. set_address then
    lookup_serviceability) so they run sequentially against the shared cart.
    """

    def fn(messages, info):
        done = sum(
            1
            for m in messages
            if isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts)
        )
        if done < len(tool_calls):
            name, args = tool_calls[done]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        return ModelResponse(parts=[TextPart("DONE")])

    return fn


def say(text: str):
    """A streamable model that always replies with ``text``.

    Returns a real ``FunctionModel`` (with a ``stream_function``) so it works for both
    the non-streamed agents (``.run``) and the streamed writer (``.run_stream`` →
    ``stream_text(delta=True)``). The text is emitted in two chunks to exercise token
    streaming.
    """
    from pydantic_ai.models.function import FunctionModel

    def fn(messages, info):
        return ModelResponse(parts=[TextPart(text)])

    async def stream_fn(messages, info):
        mid = max(1, len(text) // 2)
        yield text[:mid]
        yield text[mid:]

    return FunctionModel(function=fn, stream_function=stream_fn)


@pytest.fixture
def override():
    """Context manager: override any subset of agents with scripted FunctionModels.

    Usage::

        with override(orchestrator=fm1, product_rec=fm2, writer=fm3):
            session.run_turn("...")
    """
    from pydantic_ai.models import Model
    from pydantic_ai.models.function import FunctionModel

    agents = {
        "orchestrator": orchestrator_agent,
        "product_rec": product_rec.agent,
        "checkout": checkout.agent,
        "order_status": order_status.agent,
        "writer": writer_agent,
    }

    @contextlib.contextmanager
    def _override(**fns):
        with contextlib.ExitStack() as stack:
            for name, fn in fns.items():
                model = fn if isinstance(fn, Model) else FunctionModel(fn)
                stack.enter_context(agents[name].override(model=model))
            yield

    return _override
