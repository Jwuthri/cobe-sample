"""Constrained checkout tools.

Every tool is decorated with ``@tool`` and accepts a
``ToolRuntime[RuntimeContext]`` parameter. The runtime gives access
to:

  - ``runtime.state["skills_loaded"]``: which sub-skills are loaded.
    Tools refuse if their required skill isn't loaded.
  - ``runtime.context.cart_service``: the cart for this turn.
  - ``runtime.context.user_id``: long-term memory key.
  - ``runtime.store``: the long-term memory store (auto-injected by
    langgraph when the agent was compiled with a store=).
"""

from __future__ import annotations

from typing import Literal

from agent_v4.checkout.service import CartError
from agent_v4.memory import remember_address, remember_order, remember_payment
from agent_v4.runtime import RuntimeContext
from langchain.tools import ToolRuntime
from langchain_core.tools import tool


def _require_skill(runtime: ToolRuntime[RuntimeContext], skill_name: str) -> str | None:
    """Return None if the skill is loaded, else an error string to short-circuit."""
    loaded = (runtime.state or {}).get("skills_loaded", []) or []
    if skill_name in loaded:
        return None
    return (
        f"Error: this tool requires the '{skill_name}' skill. "
        f"Call load_skill('{skill_name}') first."
    )


# ----- products (always available) -----
@tool
def add_item(
    product_id: str, quantity: int = 1, runtime: ToolRuntime[RuntimeContext] = None
) -> str:
    """Add a product (e.g. 'P-1') to the cart with quantity (default 1)."""
    try:
        return runtime.context.cart_service.add_item(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


@tool
def remove_item(product_id: str, runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Remove a product from the cart."""
    try:
        return runtime.context.cart_service.remove_item(product_id)
    except CartError as e:
        return f"error: {e}"


@tool
def set_quantity(
    product_id: str, quantity: int, runtime: ToolRuntime[RuntimeContext] = None
) -> str:
    """Set the quantity of a cart line. quantity=0 removes the line."""
    try:
        return runtime.context.cart_service.set_quantity(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


# ----- identity (requires collect_identity) -----
@tool
def set_customer(
    first_name: str,
    last_name: str,
    email: str | None = None,
    runtime: ToolRuntime[RuntimeContext] = None,
) -> str:
    """Save the customer's first and last name (and optional email)."""
    err = _require_skill(runtime, "collect_identity")
    if err:
        return err
    return runtime.context.cart_service.set_customer(first_name, last_name, email)


# ----- address (requires collect_address) -----
@tool
def set_address(
    street: str,
    city: str,
    zip_code: str,
    state: str | None = None,
    country: str = "US",
    runtime: ToolRuntime[RuntimeContext] = None,
) -> str:
    """Save a structured shipping address. Invalidates serviceability, shipping, tax."""
    err = _require_skill(runtime, "collect_address")
    if err:
        return err
    return runtime.context.cart_service.set_address(
        street, city, zip_code, state=state, country=country
    )


# ----- serviceability (requires lookup_serviceability skill) -----
@tool
def lookup_serviceability(runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Check whether the saved address is serviceable; sets serviceable_options."""
    err = _require_skill(runtime, "lookup_serviceability")
    if err:
        return err
    try:
        return runtime.context.cart_service.lookup_serviceability()
    except CartError as e:
        return f"error: {e}"


# ----- delivery (requires collect_delivery) -----
@tool
def set_delivery_option(
    option: Literal["2h", "4h", "next_day", "standard"],
    runtime: ToolRuntime[RuntimeContext] = None,
) -> str:
    """Pick a delivery option (must be in serviceable_options)."""
    err = _require_skill(runtime, "collect_delivery")
    if err:
        return err
    try:
        return runtime.context.cart_service.set_delivery_option(option)
    except CartError as e:
        return f"error: {e}"


@tool
def quote_shipping(runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Compute and store a shipping quote for the current cart inputs."""
    err = _require_skill(runtime, "collect_delivery")
    if err:
        return err
    try:
        return runtime.context.cart_service.quote_shipping()
    except CartError as e:
        return f"error: {e}"


@tool
def compute_tax(runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Compute and store tax for the current cart."""
    err = _require_skill(runtime, "collect_delivery")
    if err:
        return err
    try:
        return runtime.context.cart_service.compute_tax()
    except CartError as e:
        return f"error: {e}"


@tool
def apply_promo(code: str, runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Apply a promo code (e.g. WELCOME10, SHOES20)."""
    # Promo can be applied any time after items exist — no skill gate needed.
    try:
        return runtime.context.cart_service.apply_promo(code)
    except CartError as e:
        return f"error: {e}"


# ----- payment (requires collect_payment) -----
@tool
def attach_payment(
    method: Literal["card", "cash", "wallet"],
    card_token: str | None = None,
    runtime: ToolRuntime[RuntimeContext] = None,
) -> str:
    """Set the payment method; for 'card', provide a card_token (mocked)."""
    err = _require_skill(runtime, "collect_payment")
    if err:
        return err
    try:
        return runtime.context.cart_service.attach_payment(method, card_token=card_token)
    except CartError as e:
        return f"error: {e}"


# ----- confirm (gated by skill AND state) -----
@tool
def confirm_checkout(runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Place the order. Pauses for human approval via the HITL middleware."""
    err = _require_skill(runtime, "collect_payment")
    if err:
        return err
    ctx = runtime.context
    service = ctx.cart_service
    if not service.cart.ready_to_confirm():
        blockers = "; ".join(b.code for b in service.cart.blockers())
        return f"error: cannot confirm — blockers: {blockers}"
    try:
        result = service.confirm()
    except CartError as e:
        return f"error: {e}"
    # Persist to long-term memory.
    if runtime.store is not None:
        remember_address(runtime.store, ctx.user_id, service.cart.address.model_dump())
        if service.cart.payment_method:
            remember_payment(
                runtime.store,
                ctx.user_id,
                service.cart.payment_method,
                service.cart.card_token,
            )
        remember_order(
            runtime.store,
            ctx.user_id,
            {
                "receipt_id": service.cart.receipt_id,
                "items": [i.model_dump() for i in service.cart.items],
                "total": str(service.cart.grand_total),
            },
        )
    return result


# ----- summary (always available) -----
@tool
def get_cart_summary(runtime: ToolRuntime[RuntimeContext] = None) -> str:
    """Return a multi-line summary of the cart, including step + blockers."""
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
