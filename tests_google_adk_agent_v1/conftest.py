"""Shared fixtures for tests_google_adk_agent_v1. No test makes a real LLM call.

The session tests drive the agents with a scripted :class:`ScriptedModel` (a fake
:class:`~google.adk.models.BaseLlm`) swapped in via the ``override`` fixture, so the
whole streaming pipeline runs deterministically and offline.

A scripted model decides each model turn from the request: it inspects how many tool
results are already in ``llm_request.contents`` to know which step it is on (the ADK
analogue of Pydantic AI's ``FunctionModel`` counting ToolReturnParts).
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Any

import pytest
from google.adk.models import BaseLlm, LlmResponse
from google.genai import types

# The package re-exports orchestrator/writer as the *Agent* objects, and
# product_rec/checkout/order_status as their *modules* (use .agent).
from google_adk_agent_v1.agents import checkout, order_status, product_rec
from google_adk_agent_v1.agents import orchestrator as orchestrator_agent
from google_adk_agent_v1.agents import writer as writer_agent


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Deterministic CART-1000 / RCPT-9000 ids across tests."""
    monkeypatch.setattr("google_adk_agent_v1.domain.cart_service._CART_COUNTER", itertools.count(1000))
    monkeypatch.setattr("google_adk_agent_v1.domain.cart_service._RECEIPT_COUNTER", itertools.count(9000))
    yield


# --------------------------------------------------------------------------- #
# the scripted fake model
# --------------------------------------------------------------------------- #
def _count_tool_results(llm_request) -> int:
    """How many tool results are already in the request (i.e. how many steps we've run)."""
    return sum(
        1
        for c in llm_request.contents
        for p in (c.parts or [])
        if getattr(p, "function_response", None) is not None
    )


class ScriptedModel(BaseLlm):
    """A fake ``BaseLlm`` that yields scripted parts.

    ``decide(llm_request)`` returns either ``("calls", [(name, args), ...])`` to emit
    tool calls, or ``("text", str)`` to emit a final text turn. Text turns support SSE
    streaming (two partial chunks + a final aggregated event) so the streamed writer
    works through ``run_stream``.
    """

    decide: Any

    def __init__(self, decide):
        super().__init__(model="scripted", decide=decide)

    async def generate_content_async(self, llm_request, stream: bool = False):
        kind, payload = self.decide(llm_request)
        if kind == "calls":
            parts = [
                types.Part(function_call=types.FunctionCall(name=name, args=dict(args)))
                for name, args in payload
            ]
            yield LlmResponse(content=types.Content(role="model", parts=parts))
            return
        text = payload or ""
        if stream and text:
            mid = max(1, len(text) // 2)
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=text[:mid])]), partial=True
            )
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=text[mid:])]), partial=True
            )
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]), partial=False)


# --------------------------------------------------------------------------- #
# scripted-model helpers (mirror the pydantic conftest's call_then_done / sequence / say)
# --------------------------------------------------------------------------- #
def call_then_done(*tool_calls):
    """A model that emits the given tool calls on the first step, then 'DONE'.

    Each ``tool_call`` is ``(name, args_dict)``. After the tools return, emit 'DONE'
    so the agent run completes (mirrors the real worker contract).
    """

    def decide(llm_request):
        if _count_tool_results(llm_request) > 0:
            return ("text", "DONE")
        return ("calls", list(tool_calls))

    return ScriptedModel(decide)


def sequence(*tool_calls):
    """A model that emits ONE tool call per step (in order), then 'DONE'.

    Use when later calls depend on earlier ones (e.g. set_address then
    lookup_serviceability) so they run sequentially against the shared cart.
    """

    def decide(llm_request):
        done = _count_tool_results(llm_request)
        if done < len(tool_calls):
            return ("calls", [tool_calls[done]])
        return ("text", "DONE")

    return ScriptedModel(decide)


def say(text: str):
    """A streamable model that always replies with ``text`` (two chunks under SSE)."""

    def decide(llm_request):
        return ("text", text)

    return ScriptedModel(decide)


@pytest.fixture
def override():
    """Context manager: override any subset of agents with scripted models.

    Usage::

        with override(orchestrator=m1, product_rec=m2, writer=m3):
            session.run_turn("...")
    """
    agents = {
        "orchestrator": orchestrator_agent,
        "product_rec": product_rec.agent,
        "checkout": checkout.agent,
        "order_status": order_status.agent,
        "writer": writer_agent,
    }

    @contextlib.contextmanager
    def _override(**models):
        saved = {name: agents[name].model for name in models}
        try:
            for name, model in models.items():
                agents[name].model = model
            yield
        finally:
            for name, model in saved.items():
                agents[name].model = model

    return _override
