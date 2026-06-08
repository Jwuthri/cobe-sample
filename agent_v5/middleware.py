"""Deterministic safety nets, ported from v4's supervisor to middleware.

v4 enforced two hard guarantees in Python (not prompt text):
  * **empty-cart override** — the classifier could never send an "add X" to the
    checkout leaf when the cart was empty;
  * **bounded loop** — ``MAX_ITERATIONS`` capped how many leaves ran per turn.

In the tool-calling topology there is no classifier to override, so the
empty-cart guarantee moves into a ``wrap_model_call`` middleware that simply
*removes the ``checkout`` tool from the model's options whenever the cart is
empty*. The model then structurally cannot call checkout — the same guarantee,
enforced before the model chooses, which is strictly stronger than a
system-prompt rule the LLM might ignore. The loop cap is the built-in
``ToolCallLimitMiddleware`` (wired in :mod:`agent_v5.supervisor`).
"""

from __future__ import annotations

from agent_v5.subagents import CHECKOUT_TOOL_NAME
from langchain.agents.middleware import wrap_model_call


@wrap_model_call
def empty_cart_guard(request, handler):
    """Strip the ``checkout`` tool from this model call when the cart has no items.

    Mirrors v4 ``supervisor.py`` step 5b: an empty cart routes any shopping
    intent to ``product_rec`` (which can add items + hand off), never to
    checkout (which has nothing to do without items).
    """
    ctx = getattr(getattr(request, "runtime", None), "context", None)
    cart_service = getattr(ctx, "cart_service", None)
    cart_empty = bool(cart_service) and not cart_service.cart.items

    if cart_empty and request.tools:
        kept = [t for t in request.tools if getattr(t, "name", None) != CHECKOUT_TOOL_NAME]
        if len(kept) != len(request.tools):
            request = request.override(tools=kept)
    return handler(request)


__all__ = ["empty_cart_guard"]
