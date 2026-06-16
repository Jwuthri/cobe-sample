"""Long-term memory — a dead-simple in-memory key/value store.

This stands in for a real database (SQLite/Postgres). It keeps a few namespaces of
per-user data that survive across sessions: saved addresses, the last payment
method, and the user's order history. ``confirm_checkout`` writes here on success;
``order_status`` reads from it.

The interface (``put`` / ``get``) is intentionally tiny so swapping in a real
backend later is a one-file change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# namespaces
NS_ADDRESSES = "addresses"
NS_PAYMENT = "payment"
NS_ORDERS = "orders"
NS_PREFS = "preferences"


class MemoryStore:
    """A nested dict ``{namespace: {user_id: value}}``."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def get(self, namespace: str, user_id: str) -> Any | None:
        return self._data.get(namespace, {}).get(user_id)

    def put(self, namespace: str, user_id: str, value: Any) -> None:
        self._data.setdefault(namespace, {})[user_id] = value


# --------------------------------------------------------------------------- #
# typed helpers over the store (the store stays generic; meaning lives here)
# --------------------------------------------------------------------------- #
def saved_addresses(store: MemoryStore, user_id: str) -> list[dict[str, Any]]:
    return (store.get(NS_ADDRESSES, user_id) or {}).get("addresses", [])


def remember_address(store: MemoryStore, user_id: str, address: dict[str, Any]) -> None:
    addrs = saved_addresses(store, user_id)
    key = (address.get("street"), address.get("zip_code"))
    if any((a.get("street"), a.get("zip_code")) == key for a in addrs):
        return
    addrs.append(address)
    store.put(NS_ADDRESSES, user_id, {"addresses": addrs})


def remember_payment(
    store: MemoryStore, user_id: str, method: str, card_token: str | None = None
) -> None:
    payload: dict[str, Any] = {"method": method}
    if card_token:
        payload["card_last4"] = card_token[-4:]
        payload["card_token_ref"] = card_token
    store.put(NS_PAYMENT, user_id, payload)


def remember_order(store: MemoryStore, user_id: str, receipt: dict[str, Any]) -> None:
    history = (store.get(NS_ORDERS, user_id) or {}).get("orders", [])
    history.append({**receipt, "ts": datetime.now(UTC).isoformat()})
    store.put(NS_ORDERS, user_id, {"orders": history})


def recent_orders(store: MemoryStore, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    history = (store.get(NS_ORDERS, user_id) or {}).get("orders", [])
    return list(reversed(history))[:limit]
