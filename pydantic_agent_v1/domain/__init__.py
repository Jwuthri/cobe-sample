"""The shopping domain — a mock e-commerce model expressed as pure logic.

This layer knows nothing about LLMs, agents, tools, or streaming. It is the source
of truth the tools mutate and the writer reports on. Start with :mod:`cart` — the
checkout state machine — then :mod:`cart_service` for the mutation rules.
"""

from __future__ import annotations

from pydantic_agent_v1.domain.cart import (
    Address,
    Blocker,
    Cart,
    CartItem,
    CheckoutStep,
    Customer,
    DeliveryOption,
    PaymentMethod,
)
from pydantic_agent_v1.domain.cart_service import CartError, CartService
from pydantic_agent_v1.domain.catalog import CATALOG, Product, get, search
from pydantic_agent_v1.domain.memory import MemoryStore
from pydantic_agent_v1.domain.orders import ORDERS, Order, get_order

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
    "get",
    "search",
    "MemoryStore",
    "ORDERS",
    "Order",
    "get_order",
]
