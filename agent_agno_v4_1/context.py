"""``ShoppingContext`` — the per-turn shared state handed to Agno via ``dependencies``.

Agno shallow-copies the ``dependencies`` dict per run but passes the *values* by
reference, so the live :class:`CartService`, the memory store, and the
``step_results`` accumulator all propagate from the team leader down into every
member's tools (and back out). This is the Agno analogue of v4_1's LangGraph
``ShoppingContext`` carried on ``runtime.context``.

Tools reach it with ``run_context.dependencies["ctx"]`` (Agno injects
``run_context: RunContext`` by exact parameter name).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_v4_1.core.step_result import StepResult
from agent_v4_1.shopping.domain import CartService


@dataclass
class ShoppingContext:
    user_id: str = "demo"
    session_id: str = "sess-agno-v41"
    cart_service: CartService = field(default_factory=CartService)
    store: Any = None
    # Sub-agent extractors append one StepResult per invocation; the writer and
    # the deterministic block builder both read this list.
    step_results: list[StepResult] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)

    def as_dependencies(self) -> dict[str, Any]:
        """The dict passed to ``team.arun(dependencies=...)`` / ``Agent(...)``."""
        return {"ctx": self}


def ctx_from(run_context: Any) -> "ShoppingContext":
    """Pull the ShoppingContext out of an Agno RunContext (tool-side helper)."""
    deps = getattr(run_context, "dependencies", None) or {}
    return deps["ctx"]


__all__ = ["ShoppingContext", "ctx_from"]
