"""Order-status tools — look up a past order (global DB + the user's saved history)."""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from lg_agent.shopping.context import ShoppingContext
from lg_agent.shopping.domain.memory import recent_orders
from lg_agent.shopping.domain.orders import ORDERS, get_order


@tool
def get_order_status(order_id: str, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Look up an order by id (global orders DB + the user's saved order history)."""
    if runtime and runtime.store and runtime.context:
        for o in recent_orders(runtime.store, runtime.context.user_id, limit=20):
            if o.get("receipt_id", "").upper() == order_id.upper():
                items = ", ".join(i.get("product_id", "?") for i in o.get("items", []))
                return (
                    f"Receipt {o['receipt_id']}: total ${o['total']}, "
                    f"items=[{items}], placed {o.get('ts', '?')}"
                )
    order = get_order(order_id)
    if order is None:
        return f"unknown order: {order_id}"
    tail = f", tracking: {order.tracking_url}" if order.tracking_url else ""
    return f"Order {order.id} is {order.status}, items={order.items}{tail}"


@tool
def list_recent_orders(limit: int = 5, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """List the user's recent orders from memory, then a few mocked fallbacks."""
    out: list[str] = []
    if runtime and runtime.store and runtime.context:
        for o in recent_orders(runtime.store, runtime.context.user_id, limit=limit):
            out.append(f"{o['receipt_id']}: ${o['total']} ({o.get('ts', '?')})")
    if not out:
        for o in list(ORDERS.values())[:limit]:
            out.append(f"{o.id}: {o.status}")
    return "\n".join(out) if out else "no orders found"


__all__ = ["get_order_status", "list_recent_orders"]
