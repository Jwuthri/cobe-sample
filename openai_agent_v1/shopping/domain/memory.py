"""Long-term memory — a tiny self-contained store + namespace helpers.

agent_v4_1 used langgraph's ``InMemoryStore``; this clean-room port keeps the
same ``get`` / ``put`` surface (an item with a ``.value`` dict) but implements it
in a dozen lines so the OpenAI Agents SDK port carries no langgraph dependency.
Production would swap ``InMemoryStore`` for a SQLite/Postgres-backed store; the
interface is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---- namespaces ----
NS_ADDRESSES: tuple[str, ...] = ("addresses",)
NS_PAYMENT: tuple[str, ...] = ("payment",)
NS_ORDERS: tuple[str, ...] = ("orders",)
NS_PREFS: tuple[str, ...] = ("preferences",)


@dataclass
class _Item:
    """A stored record — mirrors langgraph's store Item (exposes ``.value``)."""

    value: dict[str, Any]


@dataclass
class InMemoryStore:
    """Namespaced key→dict store with langgraph-compatible get/put semantics."""

    _data: dict[tuple[tuple[str, ...], str], _Item] = field(default_factory=dict)

    def get(self, namespace: tuple[str, ...], key: str) -> _Item | None:
        return self._data.get((tuple(namespace), key))

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        self._data[(tuple(namespace), key)] = _Item(value=dict(value))


BaseStore = InMemoryStore  # name parity with the v4_1 typing surface


def build_store() -> InMemoryStore:
    return InMemoryStore()


# ---- addresses ----
def saved_addresses(store: BaseStore, user_id: str) -> list[dict[str, Any]]:
    item = store.get(NS_ADDRESSES, user_id)
    if not item:
        return []
    return item.value.get("addresses", [])


def remember_address(store: BaseStore, user_id: str, address: dict[str, Any]) -> None:
    addrs = saved_addresses(store, user_id)
    key = (address.get("street"), address.get("zip_code"))
    if any((a.get("street"), a.get("zip_code")) == key for a in addrs):
        return
    addrs.append(address)
    store.put(NS_ADDRESSES, user_id, {"addresses": addrs})


# ---- payment ----
def saved_payment(store: BaseStore, user_id: str) -> dict[str, Any] | None:
    item = store.get(NS_PAYMENT, user_id)
    return item.value if item else None


def remember_payment(
    store: BaseStore, user_id: str, method: str, card_token: str | None = None
) -> None:
    payload: dict[str, Any] = {"method": method}
    if card_token:
        payload["card_last4"] = card_token[-4:]
        payload["card_token_ref"] = card_token
    store.put(NS_PAYMENT, user_id, payload)


# ---- orders ----
def remember_order(store: BaseStore, user_id: str, receipt: dict[str, Any]) -> None:
    item = store.get(NS_ORDERS, user_id)
    history = item.value.get("orders", []) if item else []
    history.append({**receipt, "ts": datetime.now(UTC).isoformat()})
    store.put(NS_ORDERS, user_id, {"orders": history})


def recent_orders(store: BaseStore, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    item = store.get(NS_ORDERS, user_id)
    if not item:
        return []
    return list(reversed(item.value.get("orders", [])))[:limit]


# ---- preferences ----
def preferences(store: BaseStore, user_id: str) -> dict[str, Any]:
    item = store.get(NS_PREFS, user_id)
    return item.value if item else {}


def set_preference(store: BaseStore, user_id: str, key: str, value: Any) -> None:
    prefs = preferences(store, user_id)
    prefs[key] = value
    store.put(NS_PREFS, user_id, prefs)
