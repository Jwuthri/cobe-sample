"""Shared fixtures for tests_v4_1. No test makes a real LLM call."""

from __future__ import annotations

import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from agent_v4_1.shopping.context import ShoppingContext
from agent_v4_1.shopping.domain import CartService


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
def fresh_ctx():
    """Factory for a fresh ShoppingContext with an empty cart."""

    def _make(**kwargs) -> ShoppingContext:
        return ShoppingContext(cart_service=CartService(), **kwargs)

    return _make


class ToolCallingFake(GenericFakeChatModel):
    """A streamable fake that also accepts ``bind_tools`` (which the base raises on).

    Script it with ``messages=iter([...])`` of AIMessages; tool calls drive the
    agent loop, a plain-text AIMessage ends it.
    """

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def tool_calling_fake():
    return ToolCallingFake
