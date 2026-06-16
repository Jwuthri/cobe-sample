"""The deterministic checkout progress anchor (injected each checkout run)."""

from __future__ import annotations

from google_adk_agent_v1.agents.checkout import asks_for_step, checkout_progress
from google_adk_agent_v1.domain import CartService


def test_progress_marks_done_and_next_step():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    block = checkout_progress(s.cart)
    assert "✓ Ada Lovelace" in block
    assert "address" in block.lower()
    # the "Resume from" line points at the current step (collecting_address)
    assert "Resume from: address" in block


def test_asks_for_step_reflects_unserviceable_address():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 X", "Nowhere", "00000")
    s.lookup_serviceability()  # not serviceable
    asks = asks_for_step(s.cart.step.value, s.cart)
    assert asks and "different" in asks[0]
