"""Shared fixtures for tests_agent_openai_sdk_v1.

The OpenAI Agents SDK does not ship a stub model, so the end-to-end
``ShoppingSession`` pipeline is exercised live (it shares its prompts and tools
with the offline-tested logic). These tests cover the pure pieces — domain,
blocks, checkout progress, guardrails — that don't need a model.
"""

from __future__ import annotations

import itertools

import pytest


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Deterministic CART-1000 / RCPT-9000 ids across tests."""
    monkeypatch.setattr(
        "agent_openai_sdk_v1.domain.cart_service._CART_COUNTER", itertools.count(1000)
    )
    monkeypatch.setattr(
        "agent_openai_sdk_v1.domain.cart_service._RECEIPT_COUNTER", itertools.count(9000)
    )
    yield
