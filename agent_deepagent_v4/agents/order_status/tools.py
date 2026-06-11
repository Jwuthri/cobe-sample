"""Order-status tools — read the mock orders DB and the user's own history."""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool

from agent_deepagent_v4.context import ShopContext
from agent_deepagent_v4.domain.memory import recent_orders
from agent_deepagent_v4.domain.orders_db import ORDERS, get_order


@tool
def get_order_status(order_id: str, runtime: ToolRuntime[ShopContext] = None) -> str:
    """Look up an order by id (ORD-* or RCPT-*).

    Checks the user's own placed orders (long-term memory) first, then the
    global mock orders table.
    """
    store = getattr(runtime, "store", None)
    ctx = getattr(runtime, "context", None)
    if store is not None and ctx is not None:
        for o in recent_orders(store, ctx.user_id, limit=20):
            if str(o.get("receipt_id", "")).upper() == order_id.upper():
                items = ", ".join(i.get("product_id", "?") for i in o.get("items", []))
                return f"Receipt {o['receipt_id']}: total ${o['total']}, items=[{items}], placed {o.get('ts', '?')}"
    order = get_order(order_id)
    if order is None:
        return f"unknown order: {order_id}"
    tail = f", tracking: {order.tracking_url}" if order.tracking_url else ""
    return f"Order {order.id} is {order.status}, items={order.items}{tail}"


@tool
def list_recent_orders(limit: int = 5, runtime: ToolRuntime[ShopContext] = None) -> str:
    """List the user's recent orders (memory first, then a few mocked fallbacks)."""
    out: list[str] = []
    store = getattr(runtime, "store", None)
    ctx = getattr(runtime, "context", None)
    if store is not None and ctx is not None:
        for o in recent_orders(store, ctx.user_id, limit=limit):
            out.append(f"{o['receipt_id']}: ${o['total']} ({o.get('ts', '?')})")
    if not out:
        for o in list(ORDERS.values())[:limit]:
            out.append(f"{o.id}: {o.status}")
    return "\n".join(out) if out else "no orders found"


ORDER_STATUS_TOOLS = [get_order_status, list_recent_orders]
