"""openai_agent_v1 shopping tenant — the demo e-commerce assistant.

Registers its leaf tools + middleware into ``core`` and exposes the streaming
:class:`ShoppingSession`. ``core`` never imports this package — wiring flows
shopping → core.
"""

from __future__ import annotations

from openai_agent_v1.shopping.session import ShoppingSession

__all__ = ["ShoppingSession"]
