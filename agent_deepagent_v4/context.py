"""Runtime context shared by the orchestrator and EVERY subagent.

deepagents propagates runtime context to subagents unchanged, so a single
``ShopContext`` carrying the live ``CartService`` is how isolated subagent
contexts all read and mutate **one** cart. Tools reach it through
``runtime: ToolRuntime[ShopContext]`` → ``runtime.context.cart_service``.

This is per-run config (immutable shape, mutable cart *contents*); it is not
graph state, so it never gets serialized into the checkpoint — only the cart
object's reference is shared, and mutations made by one subagent's tools are
visible to the next.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_deepagent_v4.domain.service import CartService


@dataclass
class ShopContext:
    user_id: str
    session_id: str
    cart_service: CartService
    # Feature flag: when True (default), confirm_checkout pauses for an explicit
    # human approval (interrupt) before placing the order. Set False for a
    # fast-path / non-interactive run where the conversational "yes" suffices.
    require_approval: bool = True
