"""Long-term memory — a tiny self-contained key/value store.

agent_v4_1 used LangGraph's ``InMemoryStore``; this package has no LangGraph
dependency, so we ship a minimal namespaced KV store with the same ``get`` /
``put`` surface plus the typed helpers the shopping tools use. Production would
swap :class:`MemoryStore` for a SQLite/Postgres-backed implementation behind the
same interface (or Agno's own ``db`` — but a deterministic typed KV store is a
better fit for saved addresses/orders than Agno's LLM-extracted user memories).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# ---- namespaces ----
NS_ADDRESSES = "addresses"
NS_PAYMENT = "payment"
NS_ORDERS = "orders"
NS_PREFS = "preferences"


class MemoryStore:
    """Namespaced dict-of-dicts. ``get`` returns the stored value (or None)."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], dict[str, Any]] = {}

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        return self._data.get((namespace, key))

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        self._data[(namespace, key)] = value


def build_store() -> MemoryStore:
    return MemoryStore()


# ---- addresses ----
def saved_addresses(store: MemoryStore, user_id: str) -> list[dict[str, Any]]:
    item = store.get(NS_ADDRESSES, user_id)
    if not item:
        return []
    return item.get("addresses", [])


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
    return store.get(NS_PREFS, user_id) or {}


def set_preference(store: MemoryStore, user_id: str, key: str, value: Any) -> None:
    prefs = preferences(store, user_id)
    prefs[key] = value
    store.put(NS_PREFS, user_id, prefs)
