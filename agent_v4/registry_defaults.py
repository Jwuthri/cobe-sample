"""Register the platform's built-in tools, skills, and middleware.

This is the v4 equivalent of v2's scattered ``import`` statements at the
top of each ``sops/*`` module — except now everything a leaf can reference
lives in one place, addressable by name, and surfaced to a future builder
UI via :func:`agent_v4.configurable.build_catalog`.

``register_platform_defaults()`` is idempotent so it's safe to call at
import time from several entry points.
"""

from __future__ import annotations

from agent_v4.configurable import (
    MIDDLEWARE_REGISTRY,
    TOOL_REGISTRY,
    register_skill,
    register_tool,
)
from agent_v4.middleware import SkillsMiddleware, log_tool_calls
from agent_v4.skills import CHECKOUT_SKILLS
from agent_v4.tools.catalog_tools import get_product, search_products
from agent_v4.tools.checkout_tools import CHECKOUT_TOOLS
from agent_v4.tools.order_tools import get_order_status, list_recent_orders
from agent_v4.tools.serviceability_tools import check_serviceability

# Tools available to leaves, addressed by their ``.name``. ``add_item`` and
# the other constrained checkout tools come in via CHECKOUT_TOOLS.
_PLATFORM_TOOLS = [
    *CHECKOUT_TOOLS,
    search_products,
    get_product,
    check_serviceability,
    get_order_status,
    list_recent_orders,
]


def _make_log_tool_calls(**_params):
    """Factory for the observability middleware (stateless, reused as-is)."""
    return log_tool_calls


def _make_skills_middleware(*, skills: list[str] | None = None, **_):
    """Factory: build a SkillsMiddleware from registered skill names.

    Leaf configs normally declare skills via ``AgentConfig.skills`` (which
    ``build_agent`` attaches automatically); this factory exists so skills
    can also be wired through the generic middleware list if desired.
    """
    from agent_v4.configurable import SKILL_REGISTRY

    resolved = [SKILL_REGISTRY.get(name) for name in (skills or [])]
    return SkillsMiddleware(resolved)


_registered = False


def register_platform_defaults() -> None:
    """Load built-in tools, skills, and middleware factories (idempotent)."""
    global _registered
    if _registered:
        return

    for tool in _PLATFORM_TOOLS:
        if not TOOL_REGISTRY.has(tool.name):
            register_tool(tool)

    for skill in CHECKOUT_SKILLS:
        register_skill(skill)

    if not MIDDLEWARE_REGISTRY.has("log_tool_calls"):
        MIDDLEWARE_REGISTRY.register(
            "log_tool_calls",
            _make_log_tool_calls,
            label="Stream tool_start/tool_end events for the UI",
            category="observability",
        )
    if not MIDDLEWARE_REGISTRY.has("skills"):
        MIDDLEWARE_REGISTRY.register(
            "skills",
            _make_skills_middleware,
            label="Load-on-demand skills + the load_skill tool",
            category="capability",
        )

    _registered = True


__all__ = ["register_platform_defaults"]
