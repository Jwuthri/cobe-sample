"""Constrained checkout tools (used by the checkout agent).

Each tool reads the live cart for this turn from
``run_context.dependencies["cart_service"]`` (the v3 analogue of v2's
``runtime.context.cart_service``). Skill-gating is NOT in the tool bodies
anymore — it's enforced centrally by :func:`agent_v3.gating.skill_gate_hook`
(attached at the agent level). ``confirm_checkout`` keeps its *domain*
gate (``ready_to_confirm``), which is cart-invariant logic, not skill
ordering.
"""

from __future__ import annotations

from typing import Literal

from agno.run.base import RunContext

from agent_v3.checkout.service import CartError
from agent_v3.deps import get_cart_service, get_store
from agent_v3.memory import remember_address, remember_order, remember_payment


# ----- products (always available) -----
def add_item(product_id: str, quantity: int = 1, run_context: RunContext = None) -> str:
    """Add a product (e.g. 'P-1') to the cart with quantity (default 1)."""
    try:
        return get_cart_service(run_context).add_item(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


def remove_item(product_id: str, run_context: RunContext = None) -> str:
    """Remove a product from the cart."""
    try:
        return get_cart_service(run_context).remove_item(product_id)
    except CartError as e:
        return f"error: {e}"


def set_quantity(product_id: str, quantity: int, run_context: RunContext = None) -> str:
    """Set the quantity of a cart line. quantity=0 removes the line."""
    try:
        return get_cart_service(run_context).set_quantity(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


# ----- identity (gated by collect_identity) -----
def set_customer(
    first_name: str,
    last_name: str,
    email: str | None = None,
    run_context: RunContext = None,
) -> str:
    """Save the customer's first and last name (and optional email)."""
    return get_cart_service(run_context).set_customer(first_name, last_name, email)


# ----- address (gated by collect_address) -----
def set_address(
    street: str,
    city: str,
    zip_code: str,
    state: str | None = None,
    country: str = "US",
    run_context: RunContext = None,
) -> str:
    """Save a structured shipping address. Invalidates serviceability, shipping, tax."""
    return get_cart_service(run_context).set_address(
        street, city, zip_code, state=state, country=country
    )


# ----- serviceability (gated by lookup_serviceability) -----
def lookup_serviceability(run_context: RunContext = None) -> str:
    """Check whether the saved address is serviceable; sets serviceable_options."""
    try:
        return get_cart_service(run_context).lookup_serviceability()
    except CartError as e:
        return f"error: {e}"


# ----- delivery (gated by collect_delivery) -----
def set_delivery_option(
    option: Literal["2h", "4h", "next_day", "standard"],
    run_context: RunContext = None,
) -> str:
    """Pick a delivery option (must be in serviceable_options)."""
    try:
        return get_cart_service(run_context).set_delivery_option(option)
    except CartError as e:
        return f"error: {e}"


def quote_shipping(run_context: RunContext = None) -> str:
    """Compute and store a shipping quote for the current cart inputs."""
    try:
        return get_cart_service(run_context).quote_shipping()
    except CartError as e:
        return f"error: {e}"


def compute_tax(run_context: RunContext = None) -> str:
    """Compute and store tax for the current cart."""
    try:
        return get_cart_service(run_context).compute_tax()
    except CartError as e:
        return f"error: {e}"


def apply_promo(code: str, run_context: RunContext = None) -> str:
    """Apply a promo code (e.g. WELCOME10, SHOES20)."""
    # Promo can be applied any time after items exist — no skill gate needed.
    try:
        return get_cart_service(run_context).apply_promo(code)
    except CartError as e:
        return f"error: {e}"


# ----- payment (gated by collect_payment) -----
def attach_payment(
    method: Literal["card", "cash", "wallet"],
    card_token: str | None = None,
    run_context: RunContext = None,
) -> str:
    """Set the payment method; for 'card', provide a card_token (mocked)."""
    try:
        return get_cart_service(run_context).attach_payment(method, card_token=card_token)
    except CartError as e:
        return f"error: {e}"


# ----- confirm (gated by collect_payment skill AND ready_to_confirm state) -----
def confirm_checkout(run_context: RunContext = None) -> str:
    """Place the order. Only call after the user has explicitly approved the
    order summary (the writer presents it and asks for a 'yes')."""
    service = get_cart_service(run_context)
    if not service.cart.ready_to_confirm():
        blockers = "; ".join(b.code for b in service.cart.blockers())
        return f"error: cannot confirm — blockers: {blockers}"
    try:
        result = service.confirm()
    except CartError as e:
        return f"error: {e}"
    # Persist to long-term memory.
    store = get_store(run_context)
    user_id = getattr(run_context, "user_id", None)
    if store is not None and user_id:
        remember_address(store, user_id, service.cart.address.model_dump())
        if service.cart.payment_method:
            remember_payment(store, user_id, service.cart.payment_method, service.cart.card_token)
        remember_order(
            store,
            user_id,
            {
                "receipt_id": service.cart.receipt_id,
                "items": [i.model_dump() for i in service.cart.items],
                "total": str(service.cart.grand_total),
            },
        )
    return result


# ----- summary (always available) -----
def get_cart_summary(run_context: RunContext = None) -> str:
    """Return a multi-line summary of the cart, including step + blockers."""
    c = get_cart_service(run_context).cart
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
    lines.append(
        f"Shipping: {'$%.2f / %dh' % (c.shipping.cost, c.shipping.eta_hours) if c.shipping_is_fresh() else 'stale/missing'}"
    )
    lines.append(f"Tax: {'$%.2f' % c.tax.amount if c.tax_is_fresh() else 'stale/missing'}")
    lines.append(f"Promo: {c.promo.code + ' -$' + str(c.promo.discount) if c.promo else 'none'}")
    lines.append(f"Payment: {c.payment_method or '-'} token={'set' if c.card_token else 'none'}")
    if c.grand_total is not None:
        lines.append(f"Grand total: ${c.grand_total:.2f}")
    bs = c.blockers()
    if bs:
        lines.append("Blockers:")
        for b in bs:
            lines.append(f"  - {b.code}: {b.message}")
    else:
        lines.append("Blockers: none (ready to confirm)")
    return "\n".join(lines)


CHECKOUT_TOOLS = [
    add_item,
    remove_item,
    set_quantity,
    set_customer,
    set_address,
    lookup_serviceability,
    set_delivery_option,
    quote_shipping,
    compute_tax,
    apply_promo,
    attach_payment,
    confirm_checkout,
    get_cart_summary,
]
