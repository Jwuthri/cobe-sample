"""Typed output blocks for the writer's "rich reply" envelope.

The writer no longer emits only a string. It emits a :class:`WriterReply`:

    {message: str, blocks: [Block, ...]}

- ``message`` is the conversational prose (FAQ / string answers + glue) — it
  remains the human-facing text and rides in ``AgentState.draft_response``,
  so every text-based path (gate, validator, emit, CLI, UI) is unchanged.
- ``blocks`` is an ordered list of **typed** payloads (a product list, an
  order status, a checkout summary). A single turn can carry 0, 1, or many
  blocks — e.g. "what hoodies do you have, and where's my order?" yields a
  ``product_reco`` block + an ``order_status`` block. Blocks are assembled
  deterministically from what the leaves already produced (see
  :func:`agent_v4.writer.build_blocks`), so ids/prices are verbatim.

Money fields are ``str`` (matching the existing ``details`` / ``serialize_state``
conventions); raw ``Decimal`` round-trips awkwardly through JSON.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from agent_v4.tools.orders_db import Order
from pydantic import BaseModel, Field


class ProductCard(BaseModel):
    """One catalog product. Mirrors the dict already in details["products"]."""

    id: str
    name: str
    price: str
    tags: list[str] = Field(default_factory=list)


class OrderLine(BaseModel):
    """One line in a checkout summary."""

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
    order: Order | None = None  # reuse the orders-DB model when we can resolve it
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


class WriterReply(BaseModel):
    """The full writer output: prose + typed blocks."""

    message: str
    blocks: list[Block] = Field(default_factory=list)


__all__ = [
    "ProductCard",
    "OrderLine",
    "ProductRecoBlock",
    "OrderStatusBlock",
    "CheckoutBlock",
    "Block",
    "WriterReply",
]
