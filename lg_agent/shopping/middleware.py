"""Tenant middleware — the two custom behaviors the shopping agents declare by name.

  * ``cart_anchor``      — injects the deterministic "Checkout progress" block as a
    transient SystemMessage on every checkout model call (re-rendered each call, so
    a mid-loop cart mutation updates it for free; nothing persisted to state).
  * ``empty_cart_guard`` — strips the ``checkout`` tool from the orchestrator's
    options when the cart is empty, so an "add X" can never route to checkout
    (enforced structurally, not by prompt).

Both implement sync + async ``wrap_model_call`` (the orchestrator runs async; a
sub-agent runs sync via the re-pump stream).
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from lg_agent.core.registry import MIDDLEWARE
from lg_agent.shopping.agents.subagents.checkout import checkout_progress

CHECKOUT_TOOL_NAME = "checkout"


class CartAnchorMiddleware(AgentMiddleware):
    """Prepend the checkout progress block to every model call (transient)."""

    def _apply(self, request: Any) -> Any:
        ctx = getattr(getattr(request, "runtime", None), "context", None)
        cart_service = getattr(ctx, "cart_service", None)
        if cart_service is not None:
            anchor = SystemMessage(content=checkout_progress(cart_service.cart))
            request = request.override(messages=[anchor, *request.messages])
        return request

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._apply(request))


class EmptyCartGuardMiddleware(AgentMiddleware):
    """Remove the checkout tool from the model's options when the cart is empty."""

    def _apply(self, request: Any) -> Any:
        ctx = getattr(getattr(request, "runtime", None), "context", None)
        cart_service = getattr(ctx, "cart_service", None)
        cart_empty = bool(cart_service) and not cart_service.cart.items
        if cart_empty and request.tools:
            kept = [t for t in request.tools if getattr(t, "name", None) != CHECKOUT_TOOL_NAME]
            if len(kept) != len(request.tools):
                request = request.override(tools=kept)
        return request

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._apply(request))


def cart_anchor() -> AgentMiddleware:
    return CartAnchorMiddleware()


def empty_cart_guard() -> AgentMiddleware:
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
