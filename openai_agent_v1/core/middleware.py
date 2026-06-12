"""Built-in platform middleware — the OpenAI-SDK analogue of v4_1's middleware.

The SDK has no ``wrap_model_call`` / ``wrap_tool_call`` hook chain, so a
"middleware" here is a small object with three optional duties the factory folds
into a native SDK construct:

  * ``transform_instructions`` → contributes to the agent's **dynamic instructions**
    callable (this is how ``cart_anchor`` injects the checkout progress block, and
    how skills inject their "Available skills" block — re-rendered each run).
  * ``tool_enabled`` → contributes to each tool's **is_enabled** predicate (this is
    how ``empty_cart_guard`` hides the checkout tool while the cart is empty).
  * ``max_turns`` → contributes a per-run **turn budget** (``tool_call_limit`` /
    ``max_turns`` map onto ``Runner.run(max_turns=…)``).

``log_tool_calls`` / ``model_call_counter`` are no-ops here: the UI events they
produced in v4_1 are emitted structurally by the session + sub-agent engine
(``ctx.emit``), so they need no per-call hook.
"""

from __future__ import annotations

from typing import Any

from openai_agent_v1.core.registry import MIDDLEWARE


class PortMiddleware:
    """Base class — every hook is an opt-in no-op by default."""

    #: A per-run turn budget this middleware contributes (``None`` = no opinion).
    max_turns: int | None = None

    def transform_instructions(self, run_ctx: Any, agent: Any, base: str) -> str:
        """Return the (possibly augmented) system prompt for this model call."""
        return base

    def tool_enabled(self, run_ctx: Any, agent: Any, tool_name: str) -> bool:
        """Return False to hide ``tool_name`` from the model for this run."""
        return True

    def extra_tools(self) -> list[Any]:
        """Extra tools this middleware contributes (e.g. skills' ``load_skill``)."""
        return []


class _Noop(PortMiddleware):
    def __init__(self, **_: Any) -> None:
        pass


class _TurnBudget(PortMiddleware):
    """Carries a per-run turn budget (``max_turns`` / ``tool_call_limit``)."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns


def model_call_counter() -> PortMiddleware:
    return _Noop()


def max_turns(max_turns: int = 30) -> PortMiddleware:
    """Cap model turns per run (v4's MAX_ITERATIONS analogue)."""
    return _TurnBudget(max_turns)


def tool_call_limit(run_limit: int = 4, exit_behavior: str = "end") -> PortMiddleware:
    """Cap the orchestrator's loop. A tool call is ~one model turn, so we budget
    ``run_limit`` tool turns + a small buffer for the final ``DONE`` turn."""
    return _TurnBudget(run_limit + 4)


def log_tool_calls(log_prefix: str = "") -> PortMiddleware:
    """No-op: tool_start/tool_end UI events are emitted by the session engine."""
    return _Noop()


def register_builtin_middleware() -> None:
    """Idempotently register the built-in middleware factories."""
    for name, factory in (
        ("model_call_counter", model_call_counter),
        ("max_turns", max_turns),
        ("tool_call_limit", tool_call_limit),
        ("log_tool_calls", log_tool_calls),
    ):
        if not MIDDLEWARE.has(name):
            MIDDLEWARE.register(name, factory)


__all__ = [
    "PortMiddleware",
    "model_call_counter",
    "max_turns",
    "tool_call_limit",
    "log_tool_calls",
    "register_builtin_middleware",
]
