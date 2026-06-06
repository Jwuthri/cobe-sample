"""Test fixtures for the agent_v3 (Agno) suite.

No OpenAI is needed: we exercise the cart service, the gating hook, the
supervisor selector, the writer payload, and the workflow with stubbed
agents. The classifier / writer / SOP agents are patched where needed.
"""

from __future__ import annotations

import itertools

import pytest


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Make cart_id and receipt_id deterministic across tests."""
    monkeypatch.setattr("agent_v3.checkout.service._CART_COUNTER", itertools.count(1000))
    monkeypatch.setattr("agent_v3.checkout.service._RECEIPT_COUNTER", itertools.count(9000))
    yield
