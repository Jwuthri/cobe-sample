"""Stateless catalog tools (used by product_rec). Plain functions = Agno tools.

Output format is load-bearing: the product_rec step parses these lines
with a regex (``P-2: Black Hoodie — $49.99 [apparel, hoodie, black]``) to
build the structured product list the writer renders. Keep it identical.
"""

from __future__ import annotations

from agent_v3.checkout import catalog


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
