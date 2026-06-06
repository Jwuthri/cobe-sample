"""Long-term memory — a tiny namespaced KV store + helpers.

agent_v2 used LangGraph's ``InMemoryStore`` (``BaseStore`` interface:
``get`` / ``put`` returning items with a ``.value``). We replicate that
exact interface here with a dependency-free in-process store so the
helper functions below are unchanged. Swap ``Store`` for an Agno
``db``-backed or Postgres store in production; the interface is identical.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# ---- namespaces ----
NS_ADDRESSES: tuple[str, ...] = ("addresses",)
NS_PAYMENT: tuple[str, ...] = ("payment",)
NS_ORDERS: tuple[str, ...] = ("orders",)
NS_PREFS: tuple[str, ...] = ("preferences",)


class _Item:
    """Mirrors LangGraph's stored-item shape (``.value`` holds the dict)."""

    __slots__ = ("value",)

    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value


class Store:
    """Minimal namespaced KV store. Same surface as LangGraph's BaseStore."""

    def __init__(self) -> None:
        self._data: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}

    def get(self, namespace: tuple[str, ...], key: str) -> _Item | None:
        value = self._data.get((namespace, key))
        return _Item(value) if value is not None else None

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        self._data[(namespace, key)] = value


def build_store() -> Store:
    """Make a fresh in-memory store. Swap for a DB-backed store in prod."""
    return Store()


# ---- addresses ----
def saved_addresses(store: Store, user_id: str) -> list[dict[str, Any]]:
    item = store.get(NS_ADDRESSES, user_id)
    if not item:
        return []
    return item.value.get("addresses", [])


def remember_address(store: Store, user_id: str, address: dict[str, Any]) -> None:
    addrs = saved_addresses(store, user_id)
    # Dedup by (street, zip).
    key = (address.get("street"), address.get("zip_code"))
    if any((a.get("street"), a.get("zip_code")) == key for a in addrs):
        return
    addrs.append(address)
    store.put(NS_ADDRESSES, user_id, {"addresses": addrs})


# ---- payment ----
def saved_payment(store: Store, user_id: str) -> dict[str, Any] | None:
    item = store.get(NS_PAYMENT, user_id)
    return item.value if item else None


def remember_payment(store: Store, user_id: str, method: str, card_token: str | None = None) -> None:
    payload: dict[str, Any] = {"method": method}
    if card_token:
        payload["card_last4"] = card_token[-4:]
        payload["card_token_ref"] = card_token  # in prod we'd store a vault id
    store.put(NS_PAYMENT, user_id, payload)


# ---- orders ----
def remember_order(store: Store, user_id: str, receipt: dict[str, Any]) -> None:
    item = store.get(NS_ORDERS, user_id)
    history = item.value.get("orders", []) if item else []
    history.append({**receipt, "ts": datetime.now(UTC).isoformat()})
    store.put(NS_ORDERS, user_id, {"orders": history})


def recent_orders(store: Store, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    item = store.get(NS_ORDERS, user_id)
    if not item:
        return []
    return list(reversed(item.value.get("orders", [])))[:limit]


# ---- preferences ----
def preferences(store: Store, user_id: str) -> dict[str, Any]:
    item = store.get(NS_PREFS, user_id)
    return item.value if item else {}


def set_preference(store: Store, user_id: str, key: str, value: Any) -> None:
    prefs = preferences(store, user_id)
    prefs[key] = value
    store.put(NS_PREFS, user_id, prefs)
