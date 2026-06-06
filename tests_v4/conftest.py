"""Test fixtures for the agent_v4 (declarative builder) suite.

No OpenAI is needed: we exercise the declarative builder, the leaf
registry, the cart service + constrained tools, the supervisor routing
math (classifier mocked), the writer payload, and the gate directly.
"""

from __future__ import annotations

import itertools

import pytest


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Make cart_id and receipt_id deterministic across tests."""
    monkeypatch.setattr("agent_v4.checkout.service._CART_COUNTER", itertools.count(1000))
    monkeypatch.setattr("agent_v4.checkout.service._RECEIPT_COUNTER", itertools.count(9000))
    yield
