"""Order-status tools.

``get_order_status`` reads from the mock orders DB (legacy ORD-* ids)
AND from the user's own order history in long-term memory.
``list_recent_orders`` reads from long-term memory.
"""

from __future__ import annotations

from agent_v2.memory import recent_orders
from agent_v2.runtime import RuntimeContext
from agent_v2.tools.orders_db import ORDERS, get_order
from langchain.tools import ToolRuntime
from langchain_core.tools import tool


@tool
def get_order_status(order_id: str, runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Look up an order by id. Checks both the global orders DB and the user's
    own order history saved in long-term memory."""
    # First check the long-term memory (user's own past orders).
    if runtime and runtime.store and runtime.context:
        for o in recent_orders(runtime.store, runtime.context.user_id, limit=20):
            if o.get("receipt_id", "").upper() == order_id.upper():
                items = ", ".join(i.get("product_id", "?") for i in o.get("items", []))
                return (
                    f"Receipt {o['receipt_id']}: total ${o['total']}, "
                    f"items=[{items}], placed {o.get('ts', '?')}"
                )
    # Fall back to the mock orders table.
    order = get_order(order_id)
    if order is None:
        return f"unknown order: {order_id}"
    tail = f", tracking: {order.tracking_url}" if order.tracking_url else ""
    return f"Order {order.id} is {order.status}, items={order.items}{tail}"


@tool
def list_recent_orders(limit: int = 5, runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """List the user's recent orders from long-term memory, then a few mocked
    fallback orders if the user has none."""
    out: list[str] = []
    if runtime and runtime.store and runtime.context:
        for o in recent_orders(runtime.store, runtime.context.user_id, limit=limit):
            out.append(f"{o['receipt_id']}: ${o['total']} ({o.get('ts','?')})")
    if not out:
        for o in list(ORDERS.values())[:limit]:
            out.append(f"{o.id}: {o.status}")
    return "\n".join(out) if out else "no orders found"
