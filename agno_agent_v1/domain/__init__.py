"""Shopping domain — the mock e-commerce model.

Pure business logic: no Agno, no LangChain, no framework imports. This package is
the *specification* of correct shopping behavior (cart math, the checkout step
machine, the confirmation gate). The agent layer drives it but never reimplements
its rules.
"""

from __future__ import annotations

from agno_agent_v1.domain.cart import (
    Address,
    Blocker,
    Cart,
    CartItem,
    CheckoutStep,
    Customer,
    DeliveryOption,
    PaymentMethod,
)
from agno_agent_v1.domain.cart_service import CartError, CartService
from agno_agent_v1.domain.catalog import CATALOG, Product
from agno_agent_v1.domain.memory import MemoryStore, build_store
from agno_agent_v1.domain.orders import ORDERS, Order, get_order

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
    "MemoryStore",
    "build_store",
    "ORDERS",
    "Order",
    "get_order",
]
