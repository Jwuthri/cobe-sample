"""All shopping tools in one module — constrained wrappers over the domain.

Every tool takes a ``ToolRuntime[ShoppingContext]`` and reads the live cart via
``runtime.context.cart_service``. Unlike agent_v4 there is **no skill gating**
(``_require_skill`` is gone): the checkout sub-agent drives its flow from the
cart's ``step`` + an injected progress anchor, so a permission layer on tools
added nothing. ``confirm_checkout`` is still gated by the cart's invariant
``blockers()`` — the real safety net.
"""

from __future__ import annotations

from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from agent_v4_1.shopping.context import ShoppingContext
from agent_v4_1.shopping.domain import catalog
from agent_v4_1.shopping.domain.cart_service import CartError
from agent_v4_1.shopping.domain.memory import (
    recent_orders,
    remember_address,
    remember_order,
    remember_payment,
)
from agent_v4_1.shopping.domain.orders import ORDERS, get_order
from agent_v4_1.shopping.domain.serviceability import lookup as lookup_zip


# =============================================================================
# catalog / serviceability (stateless — used by product_rec)
# =============================================================================
@tool
def search_products(query: str, limit: int = 5) -> str:
    """Search the catalog by free-text query. Returns one product per line."""
    products = catalog.search(query, limit=limit)
    if not products:
        return f"No products match '{query}'."
    return "\n".join(f"{p.id}: {p.name} — ${p.price:.2f} [{', '.join(p.tags)}]" for p in products)


@tool
def get_product(product_id: str) -> str:
    """Return details for a single product id (e.g. 'P-1')."""
    p = catalog.get(product_id)
    if not p:
        return f"unknown product: {product_id}"
    return f"{p.id}: {p.name} — ${p.price:.2f} [{', '.join(p.tags)}]"


@tool
def check_serviceability(zip_code: str) -> str:
    """Check whether we ship to a given zip code and which delivery options are
    available there (e.g. 'do you deliver to 94110?')."""
    z = (zip_code or "").strip()
    if not z:
        return "I need a zip code to check serviceability."
    result = lookup_zip(z)
    if result is None:
        return f"We don't currently ship to zip {z}."
    options = ", ".join(result.options)
    return (
        f"Yes, we ship to zip {z} ({result.city}, {result.country}). "
        f"Available delivery options: {options}."
    )


# =============================================================================
# cart contents (used by product_rec)
# =============================================================================
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


# =============================================================================
# checkout fields (used by checkout) — no skill gating
# =============================================================================
@tool
def set_customer(
    first_name: str,
    last_name: str,
    email: str | None = None,
    runtime: ToolRuntime[ShoppingContext] = None,
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


# =============================================================================
# order status (used by order_status)
# =============================================================================
@tool
def get_order_status(order_id: str, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """Look up an order by id (global orders DB + the user's saved order history)."""
    if runtime and runtime.store and runtime.context:
        for o in recent_orders(runtime.store, runtime.context.user_id, limit=20):
            if o.get("receipt_id", "").upper() == order_id.upper():
                items = ", ".join(i.get("product_id", "?") for i in o.get("items", []))
                return (
                    f"Receipt {o['receipt_id']}: total ${o['total']}, "
                    f"items=[{items}], placed {o.get('ts', '?')}"
                )
    order = get_order(order_id)
    if order is None:
        return f"unknown order: {order_id}"
    tail = f", tracking: {order.tracking_url}" if order.tracking_url else ""
    return f"Order {order.id} is {order.status}, items={order.items}{tail}"


@tool
def list_recent_orders(limit: int = 5, runtime: ToolRuntime[ShoppingContext] = None) -> str:
    """List the user's recent orders from memory, then a few mocked fallbacks."""
    out: list[str] = []
    if runtime and runtime.store and runtime.context:
        for o in recent_orders(runtime.store, runtime.context.user_id, limit=limit):
            out.append(f"{o['receipt_id']}: ${o['total']} ({o.get('ts', '?')})")
    if not out:
        for o in list(ORDERS.values())[:limit]:
            out.append(f"{o.id}: {o.status}")
    return "\n".join(out) if out else "no orders found"


# Tool groupings used by the agent configs / platform registration.
PRODUCT_REC_TOOLS = [
    search_products,
    get_product,
    check_serviceability,
    add_item,
    remove_item,
    set_quantity,
    get_cart_summary,
]

# checkout intentionally has NO add_item (that's product_rec's job — excluding it
# makes a double-add structurally impossible).
CHECKOUT_TOOLS = [
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

ORDER_STATUS_TOOLS = [get_order_status, list_recent_orders]

ALL_TOOLS = [
    *PRODUCT_REC_TOOLS,
    set_customer,
    set_address,
    lookup_serviceability,
    set_delivery_option,
    quote_shipping,
    compute_tax,
    apply_promo,
    attach_payment,
    confirm_checkout,
    *ORDER_STATUS_TOOLS,
]


def all_tools() -> list:
    """De-duplicated list of every shopping tool, for platform registration."""
    seen: dict[str, object] = {}
    for t in ALL_TOOLS:
        seen[t.name] = t
    return list(seen.values())


__all__ = [
    "PRODUCT_REC_TOOLS",
    "CHECKOUT_TOOLS",
    "ORDER_STATUS_TOOLS",
    "all_tools",
]
