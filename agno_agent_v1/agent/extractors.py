"""Per-sub-agent domain hooks: input builders, snapshots, extractors, summaries.

These are the small plug-in callables a :class:`~agno_agent_v1.agent.subagent.SubagentSpec`
wires into the generic wrapper. They are the only genuinely agent-specific logic.

A sub-agent's work is read from Agno's ``RunOutput.tools`` — a list of
``ToolExecution`` objects (``tool_name`` / ``tool_args`` / ``result``). The
extractors parse those tool results + diff the live cart into a grounded
``StepResult``. The orchestrator LLM only ever reads the terse ``summary``; the
rich ``details`` feeds the deterministic block builder.
"""

from __future__ import annotations

import re
from typing import Any

from agno.models.message import Message

from agno_agent_v1.agent.context import ShoppingContext, StepResult

# Sub-agent ids — the single vocabulary for sop names / block keys / routing.
PRODUCT_REC = "product_rec"
CHECKOUT = "checkout"
ORDER_STATUS = "order_status"

# Matches a line from search_products / get_product:
#   "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


# --------------------------------------------------------------------------- #
# uniform access to an Agno ToolExecution
# --------------------------------------------------------------------------- #
def _tool_name(tc: Any) -> str | None:
    return getattr(tc, "tool_name", None) or getattr(tc, "name", None)


def _tool_result(tc: Any) -> str:
    return str(getattr(tc, "result", "") or "")


# --------------------------------------------------------------------------- #
# tool-result extraction
# --------------------------------------------------------------------------- #
def extract_products(tool_calls: list[Any]) -> list[dict]:
    """Parse search_products / get_product results into structured products."""
    products: list[dict] = []
    seen: set[str] = set()
    for tc in tool_calls:
        if _tool_name(tc) not in ("search_products", "get_product"):
            continue
        for line in _tool_result(tc).splitlines():
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


def extract_serviceability(tool_calls: list[Any]) -> dict | None:
    for tc in reversed(tool_calls):
        if _tool_name(tc) != "check_serviceability":
            continue
        content = _tool_result(tc).strip()
        if content:
            return {"raw": content}
    return None


def extract_order(tool_calls: list[Any]) -> dict | None:
    for tc in tool_calls:
        if _tool_name(tc) not in ("get_order_status", "list_recent_orders"):
            continue
        content = _tool_result(tc).strip()
        if content and "unknown order" not in content.lower():
            return {"raw": content}
    return None


# --------------------------------------------------------------------------- #
# checkout asks
# --------------------------------------------------------------------------- #
def asks_for_step(step_value: str, cart) -> list[str]:
    if step_value == "collecting_identity":
        return ["first name", "last name"]
    if step_value == "collecting_address":
        return ["street", "city", "state", "zip code"]
    if step_value == "awaiting_serviceability":
        return ["(internal: serviceability lookup)"]
    if step_value == "collecting_delivery":
        opts = ", ".join(cart.serviceable_options) or "available delivery options"
        return [f"delivery option ({opts})"]
    if step_value == "collecting_payment":
        return ["payment method (card / cash / wallet)", "card_token if paying by card"]
    return []


# --------------------------------------------------------------------------- #
# snapshots + context-isolated input builders
# --------------------------------------------------------------------------- #
def cart_quantities(ctx: ShoppingContext) -> dict[str, int]:
    """Snapshot {product_id: qty} BEFORE product_rec runs — for diffing."""
    return {i.product_id: i.quantity for i in ctx.cart_service.cart.items}


def with_cart_note(ctx: ShoppingContext, query: str) -> list[Message]:
    """product_rec input: a volatile cart note (structured state) + the instruction.

    Context-isolated — NO conversation history. The orchestrator already resolved
    any reference into ``query``; the only ambient context the sub-agent needs is
    the live cart, so it can edit lines without re-searching.
    """
    cart = ctx.cart_service.cart
    messages: list[Message] = []
    if cart.items:
        note = (
            "Current cart: "
            + "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
            + ". To edit it, use remove_item / set_quantity — do NOT search the "
            "catalog to remove or change an item already in the cart."
        )
        messages.append(Message(role="system", content=note))
    messages.append(Message(role="user", content=query))
    return messages


def checkout_input(ctx: ShoppingContext, query: str) -> list[Message]:
    """checkout input: just the instruction — the progress anchor comes from the
    checkout skill, and the cart is the source of truth (no history needed)."""
    return [Message(role="user", content=query)]


# --------------------------------------------------------------------------- #
# recall snippets — domain-rendered text the orchestrator remembers next turn
# --------------------------------------------------------------------------- #
def _shown_products_recall(products: list[dict]) -> str:
    listed = "; ".join(
        f"{p['id']} {p['name']} ${p['price']}"
        + (f" [{', '.join(p.get('tags', []))}]" if p.get("tags") else "")
        for p in products
    )
    return (
        "Recently shown products (resolve references like 'the green one', 'it', "
        f"'the second one' to THESE exact ids): {listed}"
    )


# --------------------------------------------------------------------------- #
# StepResult extractors: (ctx, tool_calls, before) -> StepResult
# --------------------------------------------------------------------------- #
def extract_product_rec(ctx: ShoppingContext, tool_calls: list[Any], before: Any) -> StepResult:
    cart = ctx.cart_service.cart
    before = before or {}
    products = extract_products(tool_calls)
    serviceability = extract_serviceability(tool_calls)
    viewed_cart = any(_tool_name(tc) == "get_cart_summary" for tc in tool_calls)

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
        recall=_shown_products_recall(products) if products else None,
    )


def extract_checkout(ctx: ShoppingContext, tool_calls: list[Any], before: Any) -> StepResult:
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


def extract_order_status(ctx: ShoppingContext, tool_calls: list[Any], before: Any) -> StepResult:
    order_details = extract_order(tool_calls)
    return StepResult(
        sop=ORDER_STATUS,
        summary="looked up order status" if order_details else "could not find a matching order",
        asks=[] if order_details else ["confirm the order id"],
        next_sop=None,
        details=order_details,
        recall=f"Recently looked up order: {order_details['raw']}" if order_details else None,
    )


# --------------------------------------------------------------------------- #
# terse summaries (the only thing the orchestrator LLM reads)
# --------------------------------------------------------------------------- #
def summarize_product_rec(sr: StepResult, ctx: ShoppingContext) -> str:
    hint = " (you can proceed to checkout)" if sr.next_sop == CHECKOUT else ""
    return f"{sr.summary}{hint}"


def summarize_checkout(sr: StepResult, ctx: ShoppingContext) -> str:
    asks_note = f" Needs from user: {', '.join(sr.asks)}." if sr.asks else ""
    confirmed = " ORDER CONFIRMED." if ctx.cart_service.cart.confirmed else ""
    return f"{sr.summary}.{asks_note}{confirmed}"


__all__ = [
    "PRODUCT_REC",
    "CHECKOUT",
    "ORDER_STATUS",
    "extract_products",
    "extract_serviceability",
    "extract_order",
    "asks_for_step",
    "cart_quantities",
    "with_cart_note",
    "checkout_input",
    "extract_product_rec",
    "extract_checkout",
    "extract_order_status",
    "summarize_product_rec",
    "summarize_checkout",
]
