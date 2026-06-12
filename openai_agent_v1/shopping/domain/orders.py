"""Mock order DB consumed by the order_status tools. Self-contained copy."""

from __future__ import annotations

from pydantic import BaseModel


class Order(BaseModel):
    id: str
    status: str
    items: list[str]
    tracking_url: str | None = None


ORDERS: dict[str, Order] = {
    "ORD-7": Order(
        id="ORD-7",
        status="shipped",
        items=["P-1", "P-4"],
        tracking_url="https://track.example/ORD-7",
    ),
    "ORD-8": Order(id="ORD-8", status="processing", items=["P-3"]),
    "ORD-9": Order(
        id="ORD-9", status="delivered", items=["P-2"], tracking_url="https://track.example/ORD-9"
    ),
}


def get_order(order_id: str) -> Order | None:
    return ORDERS.get(order_id.upper().strip())
