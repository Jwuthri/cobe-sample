"""Turn each member's tool executions into a :class:`StepResult`.

The LangGraph port parsed LangChain ``ToolMessage``s; here we read Agno's
``ToolExecution`` list (``member_response.tools`` — each has ``tool_name`` and
``result``). The *logic* (product-line regex, cart diff, checkout asks) is reused
from ``agent_v4_1`` where it's framework-agnostic; only the I/O shape changes.
"""

from __future__ import annotations

import re

# Pure, framework-agnostic helpers reused verbatim from the LangGraph package.
from agent_v4_1.shopping.extractors import asks_for_step, checkout_anchor_text  # noqa: F401
from agent_agno_v4_1.context import ShoppingContext
from agent_v4_1.core.step_result import StepResult

PRODUCT_REC = "product_rec"
CHECKOUT = "checkout"
ORDER_STATUS = "order_status"

# "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


# =============================================================================
# tool-execution helpers — a "tool" is an Agno ToolExecution (tool_name, result)
# =============================================================================
def _results(tools, names: tuple[str, ...]) -> list[str]:
    return [str(t.result) for t in tools if getattr(t, "tool_name", None) in names]


def _has_tool(tools, name: str) -> bool:
    return any(getattr(t, "tool_name", None) == name for t in tools)


def extract_products(tools) -> list[dict]:
    products: list[dict] = []
    seen: set[str] = set()
    for content in _results(tools, ("search_products", "get_product")):
        for line in content.splitlines():
            match = _PRODUCT_LINE_RE.match(line.strip())
            if not match:
                continue
            pid = match.group(1)
            if pid in seen:
                continue
            seen.add(pid)
            products.append(
                {
                    "id": pid,
                    "name": match.group(2),
                    "price": match.group(3),
                    "tags": [t.strip() for t in match.group(4).split(",")],
                }
            )
    return products


def extract_serviceability(tools) -> dict | None:
    results = _results(tools, ("check_serviceability",))
    for content in reversed(results):
        if content.strip():
            return {"raw": content.strip()}
    return None


def extract_order(tools) -> dict | None:
    for content in _results(tools, ("get_order_status", "list_recent_orders")):
        if content.strip() and "unknown order" not in content.lower():
            return {"raw": content.strip()}
    return None


# =============================================================================
# StepResult extractors: (ctx, member_tools, before) -> StepResult
# =============================================================================
def extract_product_rec(ctx: ShoppingContext, tools, before: dict[str, int]) -> StepResult:
    cart = ctx.cart_service.cart
    before = before or {}
    products = extract_products(tools)
    serviceability = extract_serviceability(tools)
    viewed_cart = _has_tool(tools, "get_cart_summary")

    after = {i.product_id: i.quantity for i in cart.items}
    added = [pid for pid in after if after[pid] > before.get(pid, 0)]
    removed = [pid for pid in before if pid not in after]
    decreased = [pid for pid in after if pid in before and after[pid] < before[pid]]
    cart_changed = bool(added or removed or decreased)

    def _cart_lines() -> list[dict]:
        return [
            {"id": i.product_id, "name": i.name, "qty": i.quantity, "price": str(i.unit_price)}
            for i in cart.items
        ]

    next_sop: str | None = None
    asks: list[str] = []
    details: dict | None = None

    if added:
        summary = f"added {', '.join(added)} to cart"
        next_sop = CHECKOUT
        details = {"added": added}
        if products:
            details["products"] = products
    elif removed or decreased:
        changed = removed + decreased
        verb = "removed" if removed and not decreased else "updated"
        summary = f"{verb} cart ({', '.join(changed)})"
        details = {"cart_edit": {"removed": removed, "decreased": decreased, "items": _cart_lines()}}
    elif serviceability:
        summary = "answered a serviceability question"
        details = {"serviceability": serviceability}
        if products:
            details["products"] = products
    elif products:
        summary = f"catalog returned {len(products)} matching product(s)"
        asks = ["pick a product id (e.g. P-1) to add to your cart"]
        details = {"products": products}
    elif viewed_cart and cart.items:
        summary = "showed the cart"
        details = {"cart_edit": {"removed": [], "decreased": [], "items": _cart_lines()}}
    else:
        summary = "no products matched the user's query"
        asks = ["clarify what you're looking for"]

    return StepResult(
        sop=PRODUCT_REC,
        summary=summary,
        asks=asks,
        next_sop=next_sop,
        details=details,
        cart_diff={"items": len(cart.items)} if cart_changed else None,
    )


def extract_checkout(ctx: ShoppingContext, tools, before: dict[str, int]) -> StepResult:
    cart = ctx.cart_service.cart
    asks: list[str] = []
    if cart.step.value.startswith("collecting_"):
        asks = asks_for_step(cart.step.value, cart)
    elif cart.ready_to_confirm() and not cart.confirmed:
        asks = ["explicit yes to place the order"]
    summary = f"checkout subagent finished at step={cart.step.value}; items={len(cart.items)}"
    return StepResult(
        sop=CHECKOUT,
        summary=summary,
        asks=asks,
        next_sop=None,
        cart_diff={"step": cart.step.value},
    )


def extract_order_status(ctx: ShoppingContext, tools, before: dict[str, int]) -> StepResult:
    order_details = extract_order(tools)
    return StepResult(
        sop=ORDER_STATUS,
        summary="looked up order status" if order_details else "could not find a matching order",
        asks=[] if order_details else ["confirm the order id"],
        next_sop=None,
        details=order_details,
    )


# member name -> (extractor, writer-block kind)
EXTRACTORS = {
    PRODUCT_REC: extract_product_rec,
    CHECKOUT: extract_checkout,
    ORDER_STATUS: extract_order_status,
}
BLOCK_BY_SOP = {PRODUCT_REC: "product_reco", CHECKOUT: "checkout", ORDER_STATUS: "order_status"}


__all__ = [
    "PRODUCT_REC",
    "CHECKOUT",
    "ORDER_STATUS",
    "checkout_anchor_text",
    "extract_product_rec",
    "extract_checkout",
    "extract_order_status",
    "EXTRACTORS",
    "BLOCK_BY_SOP",
]
