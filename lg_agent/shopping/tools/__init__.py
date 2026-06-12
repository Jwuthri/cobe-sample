"""Leaf tools, grouped by the sub-agent that uses them.

A sub-agent config references its tools *by name* (registry specs); these groups
hold the concrete tool objects so they can be (a) registered into the global
``TOOLS`` registry and (b) turned into config specs via :func:`registry_specs`.
"""

from __future__ import annotations

from lg_agent.shopping.tools.cart import add_item, get_cart_summary, remove_item, set_quantity
from lg_agent.shopping.tools.catalog import check_serviceability, get_product, search_products
from lg_agent.shopping.tools.checkout import (
    apply_promo,
    attach_payment,
    compute_tax,
    confirm_checkout,
    lookup_serviceability,
    quote_shipping,
    set_address,
    set_customer,
    set_delivery_option,
)
from lg_agent.shopping.tools.orders import get_order_status, list_recent_orders

# Browse + cart management.
PRODUCT_REC_TOOLS = [
    search_products,
    get_product,
    check_serviceability,
    add_item,
    remove_item,
    set_quantity,
    get_cart_summary,
]

# Drive a purchase. Intentionally NO add_item (that's product_rec's job — excluding
# it makes a double-add structurally impossible).
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


def registry_specs(tools: list) -> list[dict]:
    """Turn a group of tool objects into config registry specs (referenced by name)."""
    return [{"kind": "registry", "name": t.name} for t in tools]


def all_tools() -> list:
    """De-duplicated list of every shopping tool, for platform registration."""
    seen: dict[str, object] = {}
    for t in [*PRODUCT_REC_TOOLS, *CHECKOUT_TOOLS, *ORDER_STATUS_TOOLS]:
        seen[t.name] = t
    return list(seen.values())


__all__ = [
    "PRODUCT_REC_TOOLS",
    "CHECKOUT_TOOLS",
    "ORDER_STATUS_TOOLS",
    "registry_specs",
    "all_tools",
]
