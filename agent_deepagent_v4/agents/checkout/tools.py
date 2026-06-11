"""Tools for the checkout agent — fulfillment only.

Notably there is **no add-item tool here**: items are added by the product
agent, so the checkout agent structurally cannot re-add (and double) a line.
Checkout advances the order field-by-field and places it safely:

  * Every mutator is a thin wrapper over ``CartService`` (the cart enforces its
    own freshness/serviceability invariants).
  * ``confirm_checkout`` is the safe-checkout chokepoint: it refuses while
    ``cart.blockers()`` is non-empty, then **pauses for human approval** via
    LangGraph's ``interrupt()`` before charging. Only an approved resume sets
    ``cart.confirmed`` and mints a receipt.
"""

from __future__ import annotations

from typing import Literal

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from agent_deepagent_v4.context import ShopContext
from agent_deepagent_v4.domain.cart import CheckoutStep
from agent_deepagent_v4.domain.memory import remember_address, remember_order, remember_payment
from agent_deepagent_v4.domain.service import CartError

_NEXT_STEP_HINT = {
    "collecting_products": "items missing — this shouldn't happen mid-checkout.",
    "collecting_identity": "identity — capture the name with set_customer.",
    "collecting_address": "address — capture the shipping address with set_address.",
    "awaiting_serviceability": "serviceability — call lookup_serviceability().",
    "collecting_delivery": "delivery — set_delivery_option, then quote_shipping() + compute_tax().",
    "collecting_payment": "payment — attach_payment (card needs a token).",
    "ready_to_confirm": "ready — on an explicit yes, call confirm_checkout(); else do nothing.",
    "confirmed": "order already placed — do nothing.",
}


def _progress_block(cart) -> str:
    def mark(done: bool, value: str) -> str:
        return f"✓ {value}".rstrip() if done else "— not provided"

    name = f"{cart.customer.first_name or ''} {cart.customer.last_name or ''}".strip()
    identity = mark(bool(cart.customer.first_name), name)
    address = mark(cart.address.is_complete(), f"{cart.address.street}, {cart.address.city} {cart.address.zip_code}")
    if cart.serviceable is True:
        serviceability = f"✓ ships here (options: {', '.join(cart.serviceable_options)})"
    elif cart.serviceable is False:
        serviceability = "✗ NOT serviceable — ask for a different address"
    else:
        serviceability = "— not checked"
    return (
        "Checkout progress (authoritative — never redo a ✓ field):\n"
        f"  identity:       {identity}\n"
        f"  address:        {address}\n"
        f"  serviceability: {serviceability}\n"
        f"  delivery:       {mark(bool(cart.delivery_option), cart.delivery_option or '')}\n"
        f"  payment:        {mark(bool(cart.payment_method), cart.payment_method or '')}\n"
        f"Resume from: {_NEXT_STEP_HINT.get(cart.step.value, 'the next missing field.')}"
    )


@tool
def checkout_progress(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Show the authoritative 'what's done / what's next' checkout state.

    Call this FIRST each turn to see where the order stands before acting.
    """
    return _progress_block(runtime.context.cart_service.cart)


@tool
def set_customer(
    first_name: str, last_name: str, email: str | None = None, runtime: ToolRuntime[ShopContext] = None
) -> str:
    """Save the customer's first and last name (and optional email)."""
    return runtime.context.cart_service.set_customer(first_name, last_name, email)


@tool
def set_address(
    street: str,
    city: str,
    zip_code: str,
    state: str | None = None,
    country: str = "US",
    runtime: ToolRuntime[ShopContext] = None,
) -> str:
    """Save the structured shipping address. Invalidates serviceability/shipping/tax."""
    return runtime.context.cart_service.set_address(street, city, zip_code, state=state, country=country)


@tool
def lookup_serviceability(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Check whether the saved address is serviceable; sets the delivery options."""
    try:
        return runtime.context.cart_service.lookup_serviceability()
    except CartError as e:
        return f"error: {e}"


@tool
def set_delivery_option(
    option: Literal["2h", "4h", "next_day", "standard"], runtime: ToolRuntime[ShopContext] = None
) -> str:
    """Pick a delivery option (must be one of the serviceable options)."""
    try:
        return runtime.context.cart_service.set_delivery_option(option)
    except CartError as e:
        return f"error: {e}"


@tool
def quote_shipping(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Compute and store a shipping quote for the current cart inputs."""
    try:
        return runtime.context.cart_service.quote_shipping()
    except CartError as e:
        return f"error: {e}"


@tool
def compute_tax(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Compute and store tax for the current cart."""
    try:
        return runtime.context.cart_service.compute_tax()
    except CartError as e:
        return f"error: {e}"


@tool
def apply_promo(code: str, runtime: ToolRuntime[ShopContext] = None) -> str:
    """Apply a promo code (e.g. WELCOME10, SHOES20)."""
    try:
        return runtime.context.cart_service.apply_promo(code)
    except CartError as e:
        return f"error: {e}"


@tool
def attach_payment(
    method: Literal["card", "cash", "wallet"],
    card_token: str | None = None,
    runtime: ToolRuntime[ShopContext] = None,
) -> str:
    """Set the payment method; for 'card', provide a card_token (mocked, any string)."""
    try:
        return runtime.context.cart_service.attach_payment(method, card_token=card_token)
    except CartError as e:
        return f"error: {e}"


@tool
def cart_summary(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Full multi-line cart summary: items, totals, freshness, payment, blockers."""
    c = runtime.context.cart_service.cart
    lines = [
        f"Cart {c.cart_id} step={c.step.value}",
        f"Customer: {c.customer.first_name or '?'} {c.customer.last_name or '?'}",
        f"Address: {c.address.street or '?'}, {c.address.city or '?'} {c.address.zip_code or '?'} {c.address.country}",
        f"Serviceable: {c.serviceable} options={c.serviceable_options}",
        f"Items ({len(c.items)}):",
    ]
    for i in c.items:
        lines.append(f"  - {i.product_id} {i.name} ×{i.quantity} @ ${i.unit_price:.2f}")
    lines.append(f"Subtotal: ${c.subtotal:.2f}")
    lines.append(f"Shipping: {('$%.2f / %dh' % (c.shipping.cost, c.shipping.eta_hours)) if c.shipping_is_fresh() else 'stale/missing'}")
    lines.append(f"Tax: {('$%.2f' % c.tax.amount) if c.tax_is_fresh() else 'stale/missing'}")
    lines.append(f"Promo: {(c.promo.code + ' -$' + str(c.promo.discount)) if c.promo else 'none'}")
    lines.append(f"Payment: {c.payment_method or '-'} token={'set' if c.card_token else 'none'}")
    if c.grand_total is not None:
        lines.append(f"Grand total: ${c.grand_total:.2f}")
    bs = c.blockers()
    lines.append("Blockers: " + ("none (ready to confirm)" if not bs else ""))
    for b in bs:
        lines.append(f"  - {b.code}: {b.message}")
    return "\n".join(lines)


@tool
def confirm_checkout(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Place the order. SAFE CHECKOUT — call ONLY after the user explicitly approves.

    Three safety layers run here:
      1. Refuses while the cart has blockers (missing field / unserviceable / stale quote).
      2. Pauses for an explicit human approval via interrupt() before charging.
      3. Only an approved resume sets cart.confirmed and mints a receipt.
    """
    ctx = runtime.context
    service = ctx.cart_service
    cart = service.cart

    if cart.confirmed:
        return f"Already placed. Receipt {cart.receipt_id}."

    blockers = cart.blockers()
    if blockers:
        return "error: cannot confirm — blockers: " + "; ".join(f"{b.code} ({b.message})" for b in blockers)

    # Safe checkout: pause for an explicit human approval before charging.
    # interrupt() suspends the run and resumes here with the value passed to
    # Command(resume=...). Requires a checkpointer + thread_id. The approval can
    # be skipped via the require_approval context flag (fast path / tests).
    if getattr(ctx, "require_approval", True):
        decision = interrupt(
            {
                "type": "confirm_order",
                "message": "Please approve placing this order.",
                "summary": {
                    "items": [{"id": i.product_id, "name": i.name, "qty": i.quantity} for i in cart.items],
                    "grand_total": f"{cart.grand_total:.2f}" if cart.grand_total is not None else None,
                    "payment_method": cart.payment_method,
                    "ship_to": f"{cart.address.city} {cart.address.zip_code}",
                },
            }
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        if not approved:
            reason = decision.get("reason") if isinstance(decision, dict) else None
            return f"Order NOT placed — customer declined at the approval step{(': ' + reason) if reason else '.'}"

    try:
        result = service.confirm()
    except CartError as e:
        return f"error: {e}"

    # Persist to long-term memory (no-ops if no store is attached).
    store = runtime.store
    remember_address(store, ctx.user_id, cart.address.model_dump())
    if cart.payment_method:
        remember_payment(store, ctx.user_id, cart.payment_method, cart.card_token)
    remember_order(
        store,
        ctx.user_id,
        {
            "receipt_id": cart.receipt_id,
            "items": [i.model_dump() for i in cart.items],
            "total": str(cart.grand_total),
        },
    )
    return result


CHECKOUT_TOOLS = [
    checkout_progress,
    set_customer,
    set_address,
    lookup_serviceability,
    set_delivery_option,
    quote_shipping,
    compute_tax,
    apply_promo,
    attach_payment,
    cart_summary,
    confirm_checkout,
]

__all__ = ["CHECKOUT_TOOLS", "CheckoutStep"]
