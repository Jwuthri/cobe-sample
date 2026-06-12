"""Cart-content tools — thin wrappers over ``CartService`` mutations + a summary.

``get_cart_summary`` is a read used by both product_rec and checkout. Every tool
reads the live cart from ``runtime.context.cart_service`` and turns a ``CartError``
into a short ``error: ...`` string the model can react to.
"""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from lg_agent.shopping.context import ShoppingContext
from lg_agent.shopping.domain.cart_service import CartError


@tool
def add_item(product_id: str, quantity: int = 1, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Add a product (e.g. 'P-1') to the cart with quantity (default 1)."""
    try:
        return runtime.context.cart_service.add_item(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


@tool
def remove_item(product_id: str, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Remove a product from the cart."""
    try:
        return runtime.context.cart_service.remove_item(product_id)
    except CartError as e:
        return f"error: {e}"


@tool
def set_quantity(product_id: str, quantity: int, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Set the quantity of a cart line. quantity=0 removes the line."""
    try:
        return runtime.context.cart_service.set_quantity(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


@tool
def get_cart_summary(runtime: ToolRuntime[ShoppingContext] = None) -> str:
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


__all__ = ["add_item", "remove_item", "set_quantity", "get_cart_summary"]
