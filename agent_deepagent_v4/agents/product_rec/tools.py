"""Tools for the product agent: browse the catalog + edit cart CONTENTS.

Catalog lookups are stateless. Cart edits mutate the shared cart via
``runtime.context.cart_service`` — the same instance every other subagent
sees, so an item added here is visible to the checkout agent in the same turn.
"""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool

from agent_deepagent_v4.context import ShopContext
from agent_deepagent_v4.domain import catalog
from agent_deepagent_v4.domain.serviceability import lookup as serviceability_lookup
from agent_deepagent_v4.domain.service import CartError


@tool
def search_products(query: str, limit: int = 5) -> str:
    """Search the catalog by free-text query (empty query = full catalog).

    Returns one product per line: ``P-2: Black Hoodie — $49.99 [apparel, hoodie, black]``.
    """
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
    """Answer 'do you ship to <zip>?' and list the delivery options there.

    Anonymous pre-purchase lookup — it does NOT change the cart's address.
    """
    z = (zip_code or "").strip()
    if not z:
        return "I need a zip code to check serviceability."
    result = serviceability_lookup(z)
    if result is None:
        return f"We don't currently ship to zip {z}."
    return f"Yes, we ship to zip {z} ({result.city}, {result.country}). Options: {', '.join(result.options)}."


@tool
def add_item(product_id: str, quantity: int = 1, runtime: ToolRuntime[ShopContext] = None) -> str:
    """Add a product (e.g. 'P-2') to the cart with quantity (default 1)."""
    try:
        return runtime.context.cart_service.add_item(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


@tool
def remove_item(product_id: str, runtime: ToolRuntime[ShopContext] = None) -> str:
    """Remove a product line from the cart."""
    try:
        return runtime.context.cart_service.remove_item(product_id)
    except CartError as e:
        return f"error: {e}"


@tool
def set_quantity(product_id: str, quantity: int, runtime: ToolRuntime[ShopContext] = None) -> str:
    """Set the quantity of a cart line. quantity=0 removes the line."""
    try:
        return runtime.context.cart_service.set_quantity(product_id, quantity)
    except CartError as e:
        return f"error: {e}"


@tool
def view_cart(runtime: ToolRuntime[ShopContext] = None) -> str:
    """Show the current cart contents and subtotal (read-only)."""
    cart = runtime.context.cart_service.cart
    if not cart.items:
        return "Cart is empty."
    lines = [f"{i.product_id} {i.name} x{i.quantity} @ ${i.unit_price:.2f}" for i in cart.items]
    return "Cart:\n" + "\n".join(lines) + f"\nSubtotal: ${cart.subtotal:.2f}"


PRODUCT_REC_TOOLS = [
    search_products,
    get_product,
    check_serviceability,
    add_item,
    remove_item,
    set_quantity,
    view_cart,
]
