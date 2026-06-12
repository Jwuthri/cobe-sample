"""Typed "rich reply" blocks + deterministic assembly.

The blocks (product list / order status / checkout summary) are built from the
accumulated :class:`StepResult` list + the live cart, with ids/prices copied
verbatim from what the sub-agents produced. The LLM never writes these, so it
cannot hallucinate a product id or a total — this is the hallucination firewall,
and it's why the writer can stream freely (the facts are already grounded).
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from openai_agent_v1.core.step_result import StepResult
from openai_agent_v1.shopping.domain.orders import Order, get_order


# =============================================================================
# block models
# =============================================================================
class ProductCard(BaseModel):
    id: str
    name: str
    price: str
    tags: list[str] = Field(default_factory=list)


class OrderLine(BaseModel):
    id: str
    name: str
    qty: int
    line_total: str


class ProductRecoBlock(BaseModel):
    kind: Literal["product_reco"] = "product_reco"
    products: list[ProductCard] = Field(default_factory=list)
    added_ids: list[str] = Field(default_factory=list)
    serviceability: str | None = None


class OrderStatusBlock(BaseModel):
    kind: Literal["order_status"] = "order_status"
    order: Order | None = None
    raw: str | None = None


class CheckoutBlock(BaseModel):
    kind: Literal["checkout"] = "checkout"
    items: list[OrderLine] = Field(default_factory=list)
    subtotal: str | None = None
    grand_total: str | None = None
    ready_to_confirm: bool = False
    confirmed: bool = False
    receipt_id: str | None = None
    asks: list[str] = Field(default_factory=list)


Block = Annotated[
    Union[ProductRecoBlock, OrderStatusBlock, CheckoutBlock],
    Field(discriminator="kind"),
]


# =============================================================================
# helpers
# =============================================================================
def _order_from_raw(raw: str | None) -> Order | None:
    if not raw:
        return None
    match = re.search(r"(ORD-\d+)", raw, re.IGNORECASE)
    if not match:
        return None
    return get_order(match.group(1))


def _checkout_block(cart, asks: list[str]) -> CheckoutBlock:
    items = [
        OrderLine(id=i.product_id, name=i.name, qty=i.quantity, line_total=f"{i.line_total:.2f}")
        for i in cart.items
    ]
    return CheckoutBlock(
        items=items,
        subtotal=f"{cart.subtotal:.2f}",
        grand_total=f"{cart.grand_total:.2f}" if cart.grand_total is not None else None,
        ready_to_confirm=cart.ready_to_confirm(),
        confirmed=cart.confirmed,
        receipt_id=cart.receipt_id,
        asks=list(asks),
    )


# =============================================================================
# assembly
# =============================================================================
def build_blocks(
    step_results: list[StepResult], cart, block_by_sop: dict[str, str | None]
) -> list[dict]:
    """Assemble the turn's typed blocks from step results + cart.

    ``block_by_sop`` maps a sub-agent name to the block kind it produces
    (``{"product_rec": "product_reco", ...}``). Conversational turns yield ``[]``.
    """
    blocks: list = []
    checkout_done = False
    for sr in step_results:
        kind = block_by_sop.get(sr.sop)
        details = sr.details or {}

        if kind == "product_reco":
            products = [ProductCard(**p) for p in (details.get("products") or [])]
            added = list(details.get("added") or [])
            serv = details.get("serviceability")
            serv_raw = (
                serv.get("raw") if isinstance(serv, dict) else serv if isinstance(serv, str) else None
            )
            if products or added or serv_raw:
                blocks.append(
                    ProductRecoBlock(products=products, added_ids=added, serviceability=serv_raw)
                )
        elif kind == "order_status":
            raw = details.get("raw")
            if raw:
                blocks.append(OrderStatusBlock(order=_order_from_raw(raw), raw=raw))
        elif kind == "checkout" and not checkout_done:
            blocks.append(_checkout_block(cart, sr.asks))
            checkout_done = True

    return [b.model_dump(mode="json") for b in blocks]


__all__ = [
    "ProductCard",
    "OrderLine",
    "ProductRecoBlock",
    "OrderStatusBlock",
    "CheckoutBlock",
    "Block",
    "build_blocks",
]
