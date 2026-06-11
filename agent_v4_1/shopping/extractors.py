"""Per-sub-agent domain hooks: input builders, snapshots, extractors, summaries.

These are the small plug-in callables a :class:`SubagentSpec` wires into the
generic :func:`agent_v4_1.core.subagent.make_subagent_tool` skeleton. They are
the only genuinely agent-specific logic — ported from agent_v4's wrapper
factories / agent_v5's subagent functions.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_v4_1.core.step_result import StepResult

# Sub-agent ids — the single vocabulary for sop names / block keys / routing.
PRODUCT_REC = "product_rec"
CHECKOUT = "checkout"
ORDER_STATUS = "order_status"

# Matches lines from search_products / get_product:
#   "P-2: Black Hoodie — $49.99 [apparel, hoodie, black]"
_PRODUCT_LINE_RE = re.compile(r"^(P-\d+):\s+(.+?)\s+[—\-]\s+\$(\S+)\s+\[(.+?)\]$")


# =============================================================================
# tool-result extraction
# =============================================================================
def extract_products(messages) -> list[dict]:
    """Parse search_products / get_product results into structured products."""
    products: list[dict] = []
    seen: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage) or getattr(m, "name", None) not in (
            "search_products",
            "get_product",
        ):
            continue
        for line in str(m.content).splitlines():
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


def extract_serviceability(messages) -> dict | None:
    for m in reversed(messages):
        if not isinstance(m, ToolMessage) or getattr(m, "name", None) != "check_serviceability":
            continue
        content = str(m.content).strip()
        if content:
            return {"raw": content}
    return None


def extract_order(messages) -> dict | None:
    for m in messages:
        if not isinstance(m, ToolMessage) or getattr(m, "name", None) not in (
            "get_order_status",
            "list_recent_orders",
        ):
            continue
        content = str(m.content).strip()
        if content and "unknown order" not in content.lower():
            return {"raw": content}
    return None


# =============================================================================
# checkout anchor + asks
# =============================================================================
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


_NEXT_STEP_HINT = {
    "collecting_products": "items missing — this shouldn't happen mid-checkout.",
    "collecting_identity": "identity — capture the customer's name with set_customer.",
    "collecting_address": "address — capture the shipping address with set_address.",
    "awaiting_serviceability": "serviceability — call lookup_serviceability().",
    "collecting_delivery": "delivery — set_delivery_option the user chose, then quote_shipping() + compute_tax().",
    "collecting_payment": "payment — attach_payment with the user's method (card needs a token).",
    "awaiting_pricing": (
        "pricing — the cart changed, so the shipping quote and tax are stale. Recompute "
        "NOW yourself: call quote_shipping() then compute_tax(). Do NOT confirm yet — the "
        "refreshed total must be shown so the user can approve it."
    ),
    "ready_to_confirm": "ready — if the user's latest message is an explicit yes/confirm, call confirm_checkout(); otherwise do nothing.",
    "confirmed": "order already placed — do nothing.",
}


def checkout_anchor_text(cart) -> str:
    """Deterministic 'what's done / what's next' block injected each checkout turn.

    The cart is the source of truth, so we render its state explicitly instead of
    making the model rediscover it from a growing thread. ``cart.step`` drives the
    single NEXT STEP.
    """
    c = cart

    def mark(done: bool, value: str) -> str:
        return f"✓ {value}".rstrip() if done else "— not provided"

    name = f"{c.customer.first_name or ''} {c.customer.last_name or ''}".strip()
    identity = mark(bool(c.customer.first_name), name)
    address = mark(
        c.address.is_complete(),
        f"{c.address.street}, {c.address.city} {c.address.zip_code}",
    )
    if c.serviceable is True:
        serviceability = f"✓ ships here (options: {', '.join(c.serviceable_options)})"
    elif c.serviceable is False:
        serviceability = "✗ NOT serviceable — ask for a different address"
    else:
        serviceability = "— not checked"
    delivery = mark(bool(c.delivery_option), c.delivery_option or "")
    payment = mark(bool(c.payment_method), c.payment_method or "")
    if c.shipping_is_fresh() and c.tax_is_fresh():
        pricing = f"✓ shipping {c.shipping.cost} + tax {c.tax.amount} → total {c.grand_total}"
    elif c.delivery_option:
        pricing = "✗ STALE — cart changed; recompute with quote_shipping() then compute_tax()"
    else:
        pricing = "— not computed"

    return (
        "Checkout progress (authoritative — never redo a ✓ field):\n"
        f"  identity:       {identity}\n"
        f"  address:        {address}\n"
        f"  serviceability: {serviceability}\n"
        f"  delivery:       {delivery}\n"
        f"  payment:        {payment}\n"
        f"  pricing:        {pricing}\n"
        f"Resume from: {_NEXT_STEP_HINT.get(c.step.value, 'the next missing field.')}\n"
        "Advance using the user's latest message + automatic internal steps; stop "
        "at the first field that needs info the user hasn't given."
    )


# =============================================================================
# snapshots + input builders
# =============================================================================
def cart_quantities(ctx) -> dict[str, int]:
    """Snapshot of {product_id: qty} before product_rec runs — for diffing."""
    return {i.product_id: i.quantity for i in ctx.cart_service.cart.items}


def with_cart_note(ctx, history, query) -> dict:
    """product_rec input: recent history + a volatile cart note + the instruction.

    The cart note is a SUFFIX (right before the instruction), not a prefix — a
    changing note at the front would break the cacheable history prefix each turn.
    """
    cart = ctx.cart_service.cart
    messages = list(history)
    if cart.items:
        note = (
            "Current cart: "
            + "; ".join(f"{i.product_id} {i.name} x{i.quantity}" for i in cart.items)
            + ". To edit it, use remove_item / set_quantity — do NOT search the "
            "catalog to remove or change an item already in the cart."
        )
        messages.append(SystemMessage(content=note))
    messages.append(HumanMessage(content=query))
    return {"messages": messages}


def checkout_input(ctx, history, query) -> dict:
    """checkout input: just the instruction — the progress anchor comes from the
    cart_anchor middleware, and the cart is the source of truth (no history needed)."""
    return {"messages": [HumanMessage(content=query)]}


# =============================================================================
# StepResult extractors (ctx, result_messages, before) -> StepResult
# =============================================================================
def extract_product_rec(ctx, messages, before) -> StepResult:
    cart = ctx.cart_service.cart
    before = before or {}
    products = extract_products(messages)
    serviceability = extract_serviceability(messages)
    viewed_cart = any(getattr(m, "name", None) == "get_cart_summary" for m in messages)

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


def extract_checkout(ctx, messages, before) -> StepResult:
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


def extract_order_status(ctx, messages, before) -> StepResult:
    order_details = extract_order(messages)
    return StepResult(
        sop=ORDER_STATUS,
        summary="looked up order status" if order_details else "could not find a matching order",
        asks=[] if order_details else ["confirm the order id"],
        next_sop=None,
        details=order_details,
    )


# =============================================================================
# terse summaries (the only thing the orchestrator LLM reads)
# =============================================================================
def summarize_product_rec(sr: StepResult, ctx) -> str:
    hint = " (you can proceed to checkout)" if sr.next_sop == CHECKOUT else ""
    return f"{sr.summary}{hint}"


def summarize_checkout(sr: StepResult, ctx) -> str:
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
    "checkout_anchor_text",
    "cart_quantities",
    "with_cart_note",
    "checkout_input",
    "extract_product_rec",
    "extract_checkout",
    "extract_order_status",
    "summarize_product_rec",
    "summarize_checkout",
]
