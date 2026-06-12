"""Catalog + serviceability tools (stateless — used by the product_rec sub-agent)."""

from __future__ import annotations

from langchain_core.tools import tool

from lg_agent.shopping.domain import catalog
from lg_agent.shopping.domain.serviceability import lookup as lookup_zip


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


__all__ = ["search_products", "get_product", "check_serviceability"]
