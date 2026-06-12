"""Shopping domain — the mock e-commerce model (pure logic, no agent concepts).

This layer knows nothing about LLMs, tools, or streaming. It is the source of
truth the tools mutate and the writer reports on.
"""

from __future__ import annotations

from lg_agent.shopping.domain.cart import (
    Address,
    Blocker,
    Cart,
    CartItem,
    CheckoutStep,
    Customer,
    DeliveryOption,
    PaymentMethod,
)
from lg_agent.shopping.domain.cart_service import CartError, CartService
from lg_agent.shopping.domain.catalog import CATALOG, Product
from lg_agent.shopping.domain.orders import ORDERS, Order, get_order

__all__ = [
    "Address",
    "Blocker",
    "Cart",
    "CartItem",
    "CheckoutStep",
    "Customer",
    "DeliveryOption",
    "PaymentMethod",
    "CartError",
    "CartService",
    "CATALOG",
    "Product",
    "ORDERS",
    "Order",
    "get_order",
]
