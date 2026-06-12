"""Name → item registries for the four pluggable capability kinds.

On-the-fly agents reference capabilities by name; the registries resolve those
names. One generic :class:`Registry` backs all four singletons. A duplicate
registration *raises* (pass ``replace=True`` to override intentionally) — a silent
overwrite is a debugging trap.

A tenant populates these once at import time (see
:func:`lg_agent.shopping.setup.register_shopping`).
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name → item store with metadata (so a builder UI could enumerate it)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._items: dict[str, T] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def register(self, name: str, item: T, *, replace: bool = False, **meta: Any) -> None:
        if name in self._items and not replace:
            raise ValueError(
                f"{self.label} {name!r} already registered; pass replace=True to override"
            )
        self._items[name] = item
        self._meta[name] = meta

    def get(self, name: str) -> T:
        if name not in self._items:
            raise ValueError(
                f"Unknown {self.label}: {name!r}. Available: {sorted(self._items)}"
            )
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def catalog(self) -> list[dict[str, Any]]:
        return [{"name": name, **meta} for name, meta in self._meta.items()]


# The four registries. Item types are loose (Any) to avoid importing langchain here.
TOOLS: Registry[Any] = Registry("tool")
SKILLS: Registry[Any] = Registry("skill")
MIDDLEWARE: Registry[Any] = Registry("middleware")  # name -> factory(**params) -> AgentMiddleware
GUARDRAILS: Registry[Any] = Registry("guardrail")  # type -> factory(GuardrailSpec) -> AgentMiddleware


def register_tool(tool: Any, *, description: str | None = None, replace: bool = False) -> None:
    """Register a LangChain tool object (must expose ``.name``)."""
    name = getattr(tool, "name", None)
    if not name:
        raise ValueError("tool must have a .name attribute")
    TOOLS.register(
        name,
        tool,
        replace=replace,
        description=description if description is not None else (getattr(tool, "description", "") or ""),
    )


__all__ = ["Registry", "TOOLS", "SKILLS", "MIDDLEWARE", "GUARDRAILS", "register_tool"]
