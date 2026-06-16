"""Every action the system can take — thin wrappers over the domain.

Each function is an ADK tool (a plain callable; ADK builds the schema from its type
hints + docstring). Stateless lookups (catalog / serviceability) are pure functions;
anything that touches the cart takes ``tool_context: ToolContext`` and resolves the
one shared :class:`CartService` from the registry (``deps_from(tool_context)``).

The agent files pick which of these they expose (see each ``LlmAgent(tools=[...])``).
Keeping the actions in one file makes "what can the system do?" answerable at a
glance, while each agent still declares exactly which actions it is allowed.

Every tool returns a short string the model reads back. A rejected mutation comes
back as ``"error: ..."`` (not an exception) so the model can recover gracefully.
"""

from __future__ import annotations

from typing import Literal

from google.adk.tools import ToolContext

from google_adk_agent_v1.domain import catalog
from google_adk_agent_v1.domain.cart_service import CartError
from google_adk_agent_v1.domain.memory import (
    recent_orders,
    remember_address,
    remember_order,
    remember_payment,
)
from google_adk_agent_v1.domain.orders import ORDERS, get_order
from google_adk_agent_v1.domain.serviceability import lookup as lookup_zip
from google_adk_agent_v1.runtime import deps_from


# =========================================================================== #
# catalog + serviceability (stateless — no cart access)
# =========================================================================== #
def search_products(query: str, limit: int = 5) -> str:
    """Search the catalog by free-text query. Returns one product per line."""
    products = catalog.search(query, limit=limit)
    if not products:
        return f"No products match '{query}'."
    return "\n".join(f"{p.id}: {p.name} — ${p.price:.2f} [{', '.join(p.tags)}]" for p in products)


def get_product(product_id: str) -> str:
    """Return details for a single product id (e.g. 'P-1')."""
    p = catalog.get(product_id)
    if not p:
        return f"unknown product: {product_id}"
    return f"{p.id}: {p.name} — ${p.price:.2f} [{', '.join(p.tags)}]"


def check_serviceability(zip_code: str) -> str:
    """Check whether we ship to a given ZIP code and which delivery options exist there
    (e.g. 'do you deliver to 94110?')."""
    z = (zip_code or "").strip()
    if not z:
        return "I need a ZIP code to check serviceability."
    if not any(ch.isdigit() for ch in z):
        # A city/area name, not a ZIP — we only look up by ZIP, so ask for one
        # instead of confidently (and wrongly) reporting "not serviceable".
        return f"I check serviceability by ZIP code. What's the ZIP for '{z}'?"
    result = lookup_zip(z)
    if result is None:
        return f"We don't currently ship to zip {z}."
    return (
        f"Yes, we ship to zip {z} ({result.city}, {result.country}). "
        f"Available delivery options: {', '.join(result.options)}."
    )


# =========================================================================== #
# cart edits
# =========================================================================== #
def add_item(tool_context: ToolContext, product_id: str, quantity: int = 1) -> str:
    """Add a product (e.g. 'P-1') to the cart with the given quantity (default 1)."""
    try:
        return deps_from(tool_context).cart_service.add_item(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


def remove_item(tool_context: ToolContext, product_id: str) -> str:
    """Remove a product line from the cart."""
    try:
        return deps_from(tool_context).cart_service.remove_item(product_id)
    except CartError as e:
        return f"error: {e}"


def set_quantity(tool_context: ToolContext, product_id: str, quantity: int) -> str:
    """Set the quantity of a cart line. quantity=0 removes the line."""
    try:
        return deps_from(tool_context).cart_service.set_quantity(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


def get_cart_summary(tool_context: ToolContext) -> str:
    """Return a multi-line summary of the cart, including step + blockers."""
    c = deps_from(tool_context).cart_service.cart
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
    lines.append("Blockers: " + ("none (ready to confirm)" if not bs else ""))
    for b in bs:
        lines.append(f"  - {b.code}: {b.message}")
    return "\n".join(lines)


# =========================================================================== #
# checkout (identity → address → delivery → payment → confirm)
# =========================================================================== #
def set_customer(
    tool_context: ToolContext,
    first_name: str,
    last_name: str,
    email: str | None = None,
) -> str:
    """Save the customer's first and last name (and optional email).

    Only call this with a name the user actually stated. It rejects field labels /
    addresses (e.g. "Shipping address") — never guess a name to fill the field.
    """
    try:
        return deps_from(tool_context).cart_service.set_customer(first_name, last_name, email)
    except CartError as e:
        return f"error: {e}"


def set_address(
    tool_context: ToolContext,
    street: str,
    city: str,
    zip_code: str,
    state: str | None = None,
    country: str = "US",
) -> str:
    """Save a structured shipping address. Invalidates serviceability, shipping, tax."""
    return deps_from(tool_context).cart_service.set_address(
        street, city, zip_code, state=state, country=country
    )


def lookup_serviceability(tool_context: ToolContext) -> str:
    """Check whether the saved address is serviceable; sets serviceable_options."""
    try:
        return deps_from(tool_context).cart_service.lookup_serviceability()
    except CartError as e:
        return f"error: {e}"


def set_delivery_option(
    tool_context: ToolContext,
    option: Literal["2h", "4h", "next_day", "standard"],
) -> str:
    """Pick a delivery option (must be one of the serviceable_options)."""
    try:
        return deps_from(tool_context).cart_service.set_delivery_option(option)
    except CartError as e:
        return f"error: {e}"


def quote_shipping(tool_context: ToolContext) -> str:
    """Compute and store a shipping quote for the current cart inputs."""
    try:
        return deps_from(tool_context).cart_service.quote_shipping()
    except CartError as e:
        return f"error: {e}"


def compute_tax(tool_context: ToolContext) -> str:
    """Compute and store tax for the current cart."""
    try:
        return deps_from(tool_context).cart_service.compute_tax()
    except CartError as e:
        return f"error: {e}"


def apply_promo(tool_context: ToolContext, code: str) -> str:
    """Apply a promo code (e.g. WELCOME10, SHOES20)."""
    try:
        return deps_from(tool_context).cart_service.apply_promo(code)
    except CartError as e:
        return f"error: {e}"


def attach_payment(
    tool_context: ToolContext,
    method: Literal["card", "cash", "wallet"],
    card_token: str | None = None,
) -> str:
    """Set the payment method; for 'card', provide a card_token (mocked)."""
    try:
        return deps_from(tool_context).cart_service.attach_payment(method, card_token=card_token)
    except CartError as e:
        return f"error: {e}"


def confirm_checkout(tool_context: ToolContext) -> str:
    """Place the order. Refuses while the cart has blockers; persists to memory."""
    deps = deps_from(tool_context)
    service = deps.cart_service
    if not service.cart.ready_to_confirm():
        blockers = "; ".join(b.code for b in service.cart.blockers())
        return f"error: cannot confirm — blockers: {blockers}"
    try:
        result = service.confirm()
    except CartError as e:
        return f"error: {e}"
    # persist to long-term memory on success
    remember_address(deps.store, deps.user_id, service.cart.address.model_dump())
    if service.cart.payment_method:
        remember_payment(deps.store, deps.user_id, service.cart.payment_method, service.cart.card_token)
    remember_order(
        deps.store,
        deps.user_id,
        {
            "receipt_id": service.cart.receipt_id,
            "items": [i.model_dump() for i in service.cart.items],
            "total": str(service.cart.grand_total),
        },
    )
    return result


# =========================================================================== #
# order status
# =========================================================================== #
def get_order_status(tool_context: ToolContext, order_id: str) -> str:
    """Look up an order by id (global orders DB + the user's saved order history)."""
    deps = deps_from(tool_context)
    for o in recent_orders(deps.store, deps.user_id, limit=20):
        if o.get("receipt_id", "").upper() == order_id.upper():
            items = ", ".join(i.get("product_id", "?") for i in o.get("items", []))
            return f"Receipt {o['receipt_id']}: total ${o['total']}, items=[{items}], placed {o.get('ts', '?')}"
    order = get_order(order_id)
    if order is None:
        return f"unknown order: {order_id}"
    tail = f", tracking: {order.tracking_url}" if order.tracking_url else ""
    return f"Order {order.id} is {order.status}, items={order.items}{tail}"


def list_recent_orders(tool_context: ToolContext, limit: int = 5) -> str:
    """List the user's recent orders from memory, then a few mocked fallbacks."""
    deps = deps_from(tool_context)
    out = [f"{o['receipt_id']}: ${o['total']} ({o.get('ts', '?')})" for o in recent_orders(deps.store, deps.user_id, limit=limit)]
    if not out:
        out = [f"{o.id}: {o.status}" for o in list(ORDERS.values())[:limit]]
    return "\n".join(out) if out else "no orders found"
