"""Deterministic rich-reply block assembly — shared by BOTH variants.

The typed blocks (``ProductRecoBlock`` / ``OrderStatusBlock`` / ``CheckoutBlock``)
are built from the accumulated :class:`~agent_v4.step_result.StepResult` list +
the live cart, with ids/prices copied verbatim from what the leaves produced. The
LLM never writes these, so they cannot hallucinate a product id or a total — this
is why block assembly is kept identical and deterministic regardless of whether
the supervisor or a separate writer authors the prose ``message``.

This mirrors :func:`agent_v4.writer.build_blocks` but reads from a plain
``list[StepResult]`` + ``cart`` instead of an ``AgentState`` (v5 has no outer
graph state), reusing v4's ``_order_from_raw`` / ``_checkout_block`` so output is
byte-for-byte compatible with v4.
"""

from __future__ import annotations

from agent_v4.leaves import LEAVES_BY_NAME
from agent_v4.output_schemas import OrderStatusBlock, ProductCard, ProductRecoBlock
from agent_v4.step_result import StepResult
from agent_v4.writer import _checkout_block, _order_from_raw


def build_blocks(step_results: list[StepResult], cart) -> list[dict]:
    """Assemble the turn's typed blocks from accumulated step results + cart."""
    blocks: list = []
    checkout_done = False
    for sr in step_results:
        spec = LEAVES_BY_NAME.get(sr.sop)
        kind = spec.output_block if spec else None
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


__all__ = ["build_blocks"]
