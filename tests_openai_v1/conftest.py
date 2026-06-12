"""Shared fixtures for tests_openai_v1. No test here makes a real LLM call."""

from __future__ import annotations

import itertools

import pytest

from openai_agent_v1.shopping.context import ShoppingContext
from openai_agent_v1.shopping.domain import CartService


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Deterministic CART-1000 / RCPT-9000 ids across tests."""
    monkeypatch.setattr(
        "openai_agent_v1.shopping.domain.cart_service._CART_COUNTER", itertools.count(1000)
    )
    monkeypatch.setattr(
        "openai_agent_v1.shopping.domain.cart_service._RECEIPT_COUNTER", itertools.count(9000)
    )
    yield


@pytest.fixture
def fresh_ctx():
    """Factory for a fresh ShoppingContext with an empty cart."""

    def _make(**kwargs) -> ShoppingContext:
        return ShoppingContext(cart_service=CartService(), **kwargs)

    return _make
