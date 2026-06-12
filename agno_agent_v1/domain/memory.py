"""Long-term memory — a tiny in-process key/value store + namespace helpers.

agent_v4_1 used LangGraph's ``InMemoryStore``; this package depends on no
LangGraph, so it ships its own minimal store with the same ``put``/``get``
interface. Production would swap :class:`MemoryStore` for a SQLite/Postgres
implementation behind the identical helper functions below.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class MemoryStore:
    """A namespaced dict store: ``(namespace, key) -> value-dict``."""

    def __init__(self) -> None:
        self._data: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        self._data[(namespace, key)] = value

    def get(self, namespace: tuple[str, ...], key: str) -> dict[str, Any] | None:
        return self._data.get((namespace, key))


# ---- namespaces ----
NS_ADDRESSES: tuple[str, ...] = ("addresses",)
NS_PAYMENT: tuple[str, ...] = ("payment",)
NS_ORDERS: tuple[str, ...] = ("orders",)
NS_PREFS: tuple[str, ...] = ("preferences",)


def build_store() -> MemoryStore:
    return MemoryStore()


# ---- addresses ----
def saved_addresses(store: MemoryStore, user_id: str) -> list[dict[str, Any]]:
    item = store.get(NS_ADDRESSES, user_id)
    return item.get("addresses", []) if item else []


def remember_address(store: MemoryStore, user_id: str, address: dict[str, Any]) -> None:
    addrs = saved_addresses(store, user_id)
    key = (address.get("street"), address.get("zip_code"))
    if any((a.get("street"), a.get("zip_code")) == key for a in addrs):
        return
    addrs.append(address)
    store.put(NS_ADDRESSES, user_id, {"addresses": addrs})


# ---- payment ----
def saved_payment(store: MemoryStore, user_id: str) -> dict[str, Any] | None:
    return store.get(NS_PAYMENT, user_id)


def remember_payment(
    store: MemoryStore, user_id: str, method: str, card_token: str | None = None
) -> None:
    payload: dict[str, Any] = {"method": method}
    if card_token:
        payload["card_last4"] = card_token[-4:]
        payload["card_token_ref"] = card_token
    store.put(NS_PAYMENT, user_id, payload)


# ---- orders ----
def remember_order(store: MemoryStore, user_id: str, receipt: dict[str, Any]) -> None:
    item = store.get(NS_ORDERS, user_id)
    history = item.get("orders", []) if item else []
    history.append({**receipt, "ts": datetime.now(UTC).isoformat()})
    store.put(NS_ORDERS, user_id, {"orders": history})


def recent_orders(store: MemoryStore, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    item = store.get(NS_ORDERS, user_id)
    if not item:
        return []
    return list(reversed(item.get("orders", [])))[:limit]


# ---- preferences ----
def preferences(store: MemoryStore, user_id: str) -> dict[str, Any]:
    item = store.get(NS_PREFS, user_id)
    return item if item else {}


def set_preference(store: MemoryStore, user_id: str, key: str, value: Any) -> None:
    prefs = preferences(store, user_id)
    prefs[key] = value
    store.put(NS_PREFS, user_id, prefs)
