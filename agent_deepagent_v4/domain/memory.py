"""Long-term memory helpers over a LangGraph ``BaseStore``.

The store is passed to ``create_deep_agent(store=...)`` and reaches tools via
``runtime.store``. Swap ``InMemoryStore`` for a DB-backed store in prod; the
``put`` / ``get`` interface is identical. All helpers no-op gracefully when
``store`` is ``None`` so memory stays optional.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

NS_ADDRESSES: tuple[str, ...] = ("addresses",)
NS_PAYMENT: tuple[str, ...] = ("payment",)
NS_ORDERS: tuple[str, ...] = ("orders",)


def build_store() -> InMemoryStore:
    return InMemoryStore()


# ---- addresses ----
def saved_addresses(store: BaseStore | None, user_id: str) -> list[dict[str, Any]]:
    if store is None:
        return []
    item = store.get(NS_ADDRESSES, user_id)
    return item.value.get("addresses", []) if item else []


def remember_address(store: BaseStore | None, user_id: str, address: dict[str, Any]) -> None:
    if store is None:
        return
    addrs = saved_addresses(store, user_id)
    key = (address.get("street"), address.get("zip_code"))
    if any((a.get("street"), a.get("zip_code")) == key for a in addrs):
        return
    addrs.append(address)
    store.put(NS_ADDRESSES, user_id, {"addresses": addrs})


# ---- payment ----
def remember_payment(store: BaseStore | None, user_id: str, method: str, card_token: str | None = None) -> None:
    if store is None:
        return
    payload: dict[str, Any] = {"method": method}
    if card_token:
        payload["card_last4"] = card_token[-4:]
    store.put(NS_PAYMENT, user_id, payload)


# ---- orders ----
def remember_order(store: BaseStore | None, user_id: str, receipt: dict[str, Any]) -> None:
    if store is None:
        return
    item = store.get(NS_ORDERS, user_id)
    history = item.value.get("orders", []) if item else []
    history.append({**receipt, "ts": datetime.now(UTC).isoformat()})
    store.put(NS_ORDERS, user_id, {"orders": history})


def recent_orders(store: BaseStore | None, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    if store is None:
        return []
    item = store.get(NS_ORDERS, user_id)
    if not item:
        return []
    return list(reversed(item.value.get("orders", [])))[:limit]
