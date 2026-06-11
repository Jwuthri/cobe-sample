"""The shopping tenant — registers leaf tools/guardrails into ``core`` and wires
the coordinate-mode supervisor team + the streaming session."""

from __future__ import annotations

from agent_agno_v1.shopping.platform import build_supervisor, register_shopping_platform
from agent_agno_v1.shopping.session import ShoppingSession

__all__ = ["ShoppingSession", "build_supervisor", "register_shopping_platform"]
