"""Shopping-specific middleware, registered by name so configs can declare them.

Mapped onto the OpenAI Agents SDK via the :class:`PortMiddleware` hooks:

  * ``cart_anchor``      — contributes the deterministic "Checkout progress" block
    to the checkout agent's dynamic instructions (re-rendered each run, so a cart
    mutation updates it for free; nothing is persisted to state).
  * ``empty_cart_guard`` — hides the ``checkout`` tool from the orchestrator while
    the cart is empty (its ``tool_enabled`` gate is composed into the delegate's
    ``is_enabled`` by the factory), so an "add X" can never route to checkout —
    enforced structurally rather than by prompt.
"""

from __future__ import annotations

from typing import Any

from openai_agent_v1.core.middleware import PortMiddleware
from openai_agent_v1.core.registry import MIDDLEWARE
from openai_agent_v1.shopping.extractors import checkout_anchor_text

CHECKOUT_TOOL_NAME = "checkout"


class CartAnchorMiddleware(PortMiddleware):
    """Append the checkout progress block to the agent's instructions (transient)."""

    def transform_instructions(self, run_ctx: Any, agent: Any, base: str) -> str:
        ctx = getattr(run_ctx, "context", None)
        cart_service = getattr(ctx, "cart_service", None)
        if cart_service is None:
            return base
        return f"{base}\n\n{checkout_anchor_text(cart_service.cart)}"


class EmptyCartGuardMiddleware(PortMiddleware):
    """Disable the checkout tool while the cart is empty."""

    def tool_enabled(self, run_ctx: Any, agent: Any, tool_name: str) -> bool:
        if tool_name != CHECKOUT_TOOL_NAME:
            return True
        ctx = getattr(run_ctx, "context", None)
        cart_service = getattr(ctx, "cart_service", None)
        if cart_service is None:
            return True
        return bool(cart_service.cart.items)


def cart_anchor() -> PortMiddleware:
    return CartAnchorMiddleware()


def empty_cart_guard() -> PortMiddleware:
    return EmptyCartGuardMiddleware()


def register_shopping_middleware() -> None:
    for name, factory in (("cart_anchor", cart_anchor), ("empty_cart_guard", empty_cart_guard)):
        if not MIDDLEWARE.has(name):
            MIDDLEWARE.register(name, factory)


__all__ = [
    "CartAnchorMiddleware",
    "EmptyCartGuardMiddleware",
    "cart_anchor",
    "empty_cart_guard",
    "register_shopping_middleware",
    "CHECKOUT_TOOL_NAME",
]
