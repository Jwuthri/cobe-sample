"""Test fixtures.

The graph-level tests don't need OpenAI: we never invoke the outer graph
end-to-end in tests. We invoke the cart service, the constrained tools,
and the gate directly. The only place that would hit OpenAI is the
supervisor classifier, and we don't exercise that here.
"""

from __future__ import annotations

import itertools

import pytest
from agent_v2.checkout.service import _CART_COUNTER, _RECEIPT_COUNTER


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    """Make cart_id and receipt_id deterministic across tests."""
    monkeypatch.setattr("agent_v2.checkout.service._CART_COUNTER", itertools.count(1000))
    monkeypatch.setattr("agent_v2.checkout.service._RECEIPT_COUNTER", itertools.count(9000))
    yield
