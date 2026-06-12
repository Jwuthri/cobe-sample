"""Shopping domain — the mock e-commerce model (self-contained, no v4_1 imports)."""

from __future__ import annotations

from openai_agent_v1.shopping.domain.cart import (
    Address,
    Blocker,
    Cart,
    CartItem,
    CheckoutStep,
    Customer,
    DeliveryOption,
    PaymentMethod,
)
from openai_agent_v1.shopping.domain.cart_service import CartError, CartService
from openai_agent_v1.shopping.domain.catalog import CATALOG, Product
from openai_agent_v1.shopping.domain.orders import ORDERS, Order, get_order

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
