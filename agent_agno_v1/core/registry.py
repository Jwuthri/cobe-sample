"""Typed name → item registries for the two pluggable kinds (tools, guardrails).

One generic :class:`Registry` backs both singletons. It *raises* on a duplicate
name (pass ``replace=True`` to override intentionally) — a silent overwrite is a
debugging trap. The shopping platform populates these in
:func:`agent_agno_v1.shopping.platform.register_shopping_platform`.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → item store with metadata, for a future builder UI to enumerate."""

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
            raise ValueError(f"Unknown {self.label}: {name!r}. Available: {sorted(self._items)}")
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def catalog(self) -> list[dict[str, Any]]:
        return [{"name": name, **meta} for name, meta in self._meta.items()]


# The two registries. Types are loose (Any) to avoid importing agno here.
TOOLS: Registry[Any] = Registry("tool")
GUARDRAILS: Registry[Any] = Registry("guardrail")  # type -> factory(GuardrailSpec) -> rule


def register_tool(tool: Any, *, name: str | None = None, replace: bool = False) -> None:
    """Register an Agno tool (a ``Function`` from ``@tool`` exposes ``.name``)."""
    tool_name = name or getattr(tool, "name", None)
    if not tool_name:
        raise ValueError("tool must have a .name attribute or an explicit name")
    TOOLS.register(
        tool_name,
        tool,
        replace=replace,
        description=getattr(tool, "description", "") or "",
    )


__all__ = ["Registry", "TOOLS", "GUARDRAILS", "register_tool"]
