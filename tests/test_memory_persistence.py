"""Long-term memory layer: write address/payment/order; read back."""

from __future__ import annotations

from agent_v2.memory import (
    build_store,
    recent_orders,
    remember_address,
    remember_order,
    remember_payment,
    saved_addresses,
    saved_payment,
)


def test_saved_addresses_round_trip():
    store = build_store()
    remember_address(store, "u1", {"street": "1 Market", "zip_code": "94110"})
    remember_address(store, "u1", {"street": "2 Mission", "zip_code": "94110"})
    assert len(saved_addresses(store, "u1")) == 2


def test_saved_addresses_dedup_by_street_and_zip():
    store = build_store()
    remember_address(store, "u1", {"street": "1 Market", "zip_code": "94110"})
    remember_address(store, "u1", {"street": "1 Market", "zip_code": "94110"})
    assert len(saved_addresses(store, "u1")) == 1


def test_payment_round_trip_masks_card():
    store = build_store()
    remember_payment(store, "u1", "card", card_token="fixture-card-00001234")
    info = saved_payment(store, "u1")
    assert info["method"] == "card"
    assert info["card_last4"] == "1234"


def test_orders_history_in_reverse_chronological_order():
    store = build_store()
    remember_order(store, "u1", {"receipt_id": "RCPT-1", "items": [], "total": "10"})
    remember_order(store, "u1", {"receipt_id": "RCPT-2", "items": [], "total": "20"})
    out = recent_orders(store, "u1", limit=10)
    assert [o["receipt_id"] for o in out] == ["RCPT-2", "RCPT-1"]


def test_orders_separated_by_user():
    store = build_store()
    remember_order(store, "u1", {"receipt_id": "RCPT-1", "items": [], "total": "10"})
    remember_order(store, "u2", {"receipt_id": "RCPT-2", "items": [], "total": "20"})
    assert len(recent_orders(store, "u1")) == 1
    assert len(recent_orders(store, "u2")) == 1
