"""``ShoppingContext`` — the per-turn context carrying the live cart.

Extends the generic :class:`agent_v4_1.core.context.TurnContext` with a
``cart_service`` handle. The same instance flows orchestrator → sub-agent tool →
sub-agent invoke, so every tool mutates one shared cart within a turn.

Field names are a superset of what the shopping tools read
(``runtime.context.cart_service`` / ``runtime.context.user_id``), so the tools
bind to this class directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_v4_1.core.context import TurnContext
from agent_v4_1.shopping.domain import CartService


@dataclass
class ShoppingContext(TurnContext):
    cart_service: CartService = field(default_factory=CartService)


__all__ = ["ShoppingContext"]
