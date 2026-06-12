"""Shopping tenant — the demo e-commerce assistant built on ``lg_agent.core``.

Importing this package populates the capability registries (tools / middleware /
guardrails) so a :class:`ShoppingSession` can build its agents.
"""

from __future__ import annotations

from lg_agent.shopping.setup import register_shopping

register_shopping()  # populate the registries on import (idempotent)

from lg_agent.shopping.session import ShoppingSession  # noqa: E402

__all__ = ["ShoppingSession", "register_shopping"]
