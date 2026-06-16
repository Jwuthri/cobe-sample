"""Snapshot tests — the frontend's ``AgentSnapshot`` shape from cart + transcript.

This is a pure projection (no model), so a regression in the wire shape (which
would break the web UI) shows up here.
"""

from __future__ import annotations

from agent_openai_sdk_v1.domain import CartService
from agent_openai_sdk_v1.snapshot import build_snapshot


def test_empty_cart_snapshot_minimal_shape():
    snap = build_snapshot(
        user_id="u",
        session_id="s",
        cart=CartService().cart,
        transcript=[],
    )
    assert snap["user_id"] == "u"
    assert snap["session_id"] == "s"
    assert snap["cart"]["step"] == "collecting_products"
    assert snap["cart"]["items"] == []
    assert snap["cart"]["confirmed"] is False
    assert snap["messages"] == []


def test_snapshot_projects_transcript_roles_to_frontend_vocab():
    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!", "blocks": []},
    ]
    snap = build_snapshot(
        user_id="u", session_id="s", cart=CartService().cart, transcript=transcript
    )
    roles = [m["role"] for m in snap["messages"]]
    assert roles == ["human", "ai"]


def test_snapshot_after_walking_to_ready():
    s = CartService()
    s.add_item("P-2", 1)
    s.set_customer("Ada", "Lovelace")
    s.set_address("1 Market St", "San Francisco", "94105")
    s.lookup_serviceability()
    s.set_delivery_option("2h")
    s.quote_shipping()
    s.compute_tax()
    s.attach_payment("cash")
    snap = build_snapshot(user_id="u", session_id="s", cart=s.cart, transcript=[])
    assert snap["cart"]["step"] == "ready_to_confirm"
    assert snap["cart"]["ready_to_confirm"] is True
    assert snap["cart"]["grand_total"] == "74.35"
    assert snap["cart"]["serviceable"] is True
    # blockers list is empty when ready
    assert snap["cart"]["blockers"] == []
