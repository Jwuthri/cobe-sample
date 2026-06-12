"""Typed name → item registries for the four pluggable kinds.

One generic :class:`Registry` backs all four singletons. It *raises* on a
duplicate name (pass ``replace=True`` to override intentionally) — a silent
overwrite is a debugging trap. The shopping platform populates these in
:func:`openai_agent_v1.shopping.platform.register_shopping_platform`; the EXAMPLE
config's tools are stubbed by tests.
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


# The four registries. Types are loose (Any) to avoid importing the SDK here.
TOOLS: Registry[Any] = Registry("tool")
SKILLS: Registry[Any] = Registry("skill")
MIDDLEWARE: Registry[Any] = Registry("middleware")  # name -> factory(**params) -> PortMiddleware
GUARDRAILS: Registry[Any] = Registry("guardrail")  # type -> factory(GuardrailSpec) -> Guardrail


def register_tool(tool: Any, *, description: str | None = None, replace: bool = False) -> None:
    """Register an SDK function tool object (must expose ``.name``)."""
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
