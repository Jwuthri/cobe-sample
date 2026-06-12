"""Checkout tools — capture identity → address → delivery → payment, then confirm.

These drive the purchase flow. ``confirm_checkout`` is gated by the cart's
invariant ``blockers()`` (the real safety net) and persists the order to long-term
memory on success. Note there is no ``add_item`` here — adding products is
product_rec's job, which makes a double-add structurally impossible.
"""

from __future__ import annotations

from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from lg_agent.shopping.context import ShoppingContext
from lg_agent.shopping.domain.cart_service import CartError
from lg_agent.shopping.domain.memory import remember_address, remember_order, remember_payment


@tool
def set_customer(
    first_name: str,
    last_name: str,
    email: str | None = None,
    runtime: ToolRuntime[ShoppingContext] = None,
) -> str:
    """Save the customer's first and last name (and optional email).

    Only call this with a name the user actually stated. It rejects field labels /
    addresses (e.g. "Shipping address") — never guess a name to fill the field.
    """
    try:
        return runtime.context.cart_service.set_customer(first_name, last_name, email)
    except CartError as e:
        return f"error: {e}"


@tool
def set_address(
    street: str,
    city: str,
    zip_code: str,
    state: str | None = None,
    country: str = "US",
    runtime: ToolRuntime[ShoppingContext] = None,
) -> str:
    """Save a structured shipping address. Invalidates serviceability, shipping, tax."""
    return runtime.context.cart_service.set_address(
        street, city, zip_code, state=state, country=country
    )


@tool
def lookup_serviceability(runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Check whether the saved address is serviceable; sets serviceable_options."""
    try:
        return runtime.context.cart_service.lookup_serviceability()
    except CartError as e:
        return f"error: {e}"


@tool
def set_delivery_option(
    option: Literal["2h", "4h", "next_day", "standard"],
    runtime: ToolRuntime[ShoppingContext] = None,
) -> str:
    """Pick a delivery option (must be in serviceable_options)."""
    try:
        return runtime.context.cart_service.set_delivery_option(option)
    except CartError as e:
        return f"error: {e}"


@tool
def quote_shipping(runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Compute and store a shipping quote for the current cart inputs."""
    try:
        return runtime.context.cart_service.quote_shipping()
    except CartError as e:
        return f"error: {e}"


@tool
def compute_tax(runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Compute and store tax for the current cart."""
    try:
        return runtime.context.cart_service.compute_tax()
    except CartError as e:
        return f"error: {e}"


@tool
def apply_promo(code: str, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Apply a promo code (e.g. WELCOME10, SHOES20)."""
    try:
        return runtime.context.cart_service.apply_promo(code)
    except CartError as e:
        return f"error: {e}"


@tool
def attach_payment(
    method: Literal["card", "cash", "wallet"],
    card_token: str | None = None,
    runtime: ToolRuntime[ShoppingContext] = None,
) -> str:
    """Set the payment method; for 'card', provide a card_token (mocked)."""
    try:
        return runtime.context.cart_service.attach_payment(method, card_token=card_token)
    except CartError as e:
        return f"error: {e}"


@tool
def confirm_checkout(runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Place the order. Refuses while the cart has blockers; persists to memory."""
    ctx = runtime.context
    service = ctx.cart_service
    if not service.cart.ready_to_confirm():
        blockers = "; ".join(b.code for b in service.cart.blockers())
        return f"error: cannot confirm — blockers: {blockers}"
    try:
        result = service.confirm()
    except CartError as e:
        return f"error: {e}"
    if runtime.store is not None:
        remember_address(runtime.store, ctx.user_id, service.cart.address.model_dump())
        if service.cart.payment_method:
            remember_payment(
                runtime.store, ctx.user_id, service.cart.payment_method, service.cart.card_token
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


__all__ = [
    "set_customer",
    "set_address",
    "lookup_serviceability",
    "set_delivery_option",
    "quote_shipping",
    "compute_tax",
    "apply_promo",
    "attach_payment",
    "confirm_checkout",
]
