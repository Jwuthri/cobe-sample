"""Long-term memory store (ports tests/test_memory_persistence.py)."""

from __future__ import annotations

from agent_v3.memory import (
    build_store,
    recent_orders,
    remember_address,
    remember_order,
    remember_payment,
    saved_addresses,
    saved_payment,
)


def test_orders_roundtrip_recent_first():
    store = build_store()
    remember_order(store, "u", {"receipt_id": "RCPT-1", "items": [], "total": "5.00"})
    remember_order(store, "u", {"receipt_id": "RCPT-2", "items": [], "total": "6.00"})
    orders = recent_orders(store, "u")
    assert [o["receipt_id"] for o in orders] == ["RCPT-2", "RCPT-1"]


def test_address_dedup_by_street_zip():
    store = build_store()
    addr = {"street": "1 Market", "zip_code": "94110", "city": "SF"}
    remember_address(store, "u", addr)
    remember_address(store, "u", dict(addr))  # duplicate
    assert len(saved_addresses(store, "u")) == 1


def test_payment_stores_last4_only():
    store = build_store()
    remember_payment(store, "u", "card", card_token="tok_111122223333")
    p = saved_payment(store, "u")
    assert p["method"] == "card"
    assert p["card_last4"] == "3333"


def test_scoped_per_user():
    store = build_store()
    remember_order(store, "alice", {"receipt_id": "RCPT-A", "items": [], "total": "1"})
    assert recent_orders(store, "bob") == []
