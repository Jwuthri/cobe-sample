"""Shared fixtures for tests_agno_v4_1. No test makes a real LLM call.

The Agno layer is exercised with hand-rolled fakes: a ``RunContext`` carrying the
live ShoppingContext, ``SimpleNamespace`` tool-executions, and a fake team/writer
for the session pipeline. The deterministic layers (tools, extractors, blocks,
writer payload, the reused cart state machine) need no model at all.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

import pytest
from agno.run import RunContext
from agno.run.agent import RunContentEvent

from agent_agno_v4_1.context import ShoppingContext


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Deterministic CART-1000 / RCPT-9000 ids across tests."""
    monkeypatch.setattr(
        "agent_v4_1.shopping.domain.cart_service._CART_COUNTER", itertools.count(1000)
    )
    monkeypatch.setattr(
        "agent_v4_1.shopping.domain.cart_service._RECEIPT_COUNTER", itertools.count(9000)
    )
    yield


@pytest.fixture
def ctx() -> ShoppingContext:
    return ShoppingContext()


@pytest.fixture
def run_context(ctx):
    """A RunContext carrying the ShoppingContext via dependencies (tool-side)."""

    def _make(c: ShoppingContext | None = None) -> RunContext:
        return RunContext(
            run_id="t", session_id="t", dependencies=(c or ctx).as_dependencies()
        )

    return _make


def tool_exec(name: str, result: str, args: dict | None = None) -> SimpleNamespace:
    """Stand-in for an Agno ToolExecution (only tool_name/tool_args/result are read)."""
    return SimpleNamespace(tool_name=name, tool_args=args or {}, result=result)


def member_response(name: str, tools: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(agent_name=name, tools=tools)


class FakeTeam:
    """A team whose ``arun`` replays a scripted member turn (no LLM).

    ``script`` is a callable ``(ctx) -> list[member_response]`` that may mutate the
    cart via the real domain (simulating what the live members would do).
    """

    def __init__(self, script):
        self._script = script

    async def arun(self, _input: str, dependencies: dict[str, Any], **_kw):
        ctx = dependencies["ctx"]
        members = self._script(ctx)
        return SimpleNamespace(content="(leader synthesis — discarded)", member_responses=members)


class FakeWriter:
    """A writer whose ``arun(stream=True)`` yields scripted content deltas."""

    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    def arun(self, _input: str, stream: bool = False, **_kw):
        deltas = self._deltas

        async def _gen():
            for d in deltas:
                yield RunContentEvent(content=d)

        return _gen()
