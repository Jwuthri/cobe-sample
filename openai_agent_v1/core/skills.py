"""Skills — instruction bundles the model can pull into context on demand.

A skill is a one-line ``description`` (always visible in the system prompt) plus
a full ``content`` body the model loads via the ``load_skill`` tool. Skills are
pure instructions — there is no ``unlocks`` tool-gating (the shopping checkout
drives its flow from cart state, not a permission system).

Ported to the OpenAI Agents SDK: the "Available skills" block is contributed to
the agent's **dynamic instructions** (re-rendered every run with up-to-date
"(loaded)" markers), and ``load_skill`` is a ``function_tool`` that records the
load on the shared ``ctx.skills_loaded`` and returns the skill body.
"""

from __future__ import annotations

from typing import Any

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel

from openai_agent_v1.core.middleware import PortMiddleware


class Skill(BaseModel):
    name: str
    description: str = ""
    content: str


def render_available_block(skills: list[Skill], loaded: list[str]) -> str:
    lines = ["Available skills (call load_skill to load):"]
    for s in skills:
        marker = " (loaded)" if s.name in loaded else ""
        lines.append(f"  - {s.name}{marker}: {s.description}")
    return "\n".join(lines)


def make_load_skill_tool(skills: list[Skill]):
    """Factory: a ``load_skill`` tool bound to a given skill set."""
    by_name = {s.name: s for s in skills}

    @function_tool(name_override="load_skill")
    async def load_skill(wrapper: RunContextWrapper[Any], skill_name: str) -> str:
        """Load a skill by name so its full instructions become available.

        Available skill names are listed in the system prompt under
        'Available skills'.
        """
        skill = by_name.get(skill_name)
        if skill is None:
            options = ", ".join(by_name.keys())
            return f"unknown skill '{skill_name}'. options: {options}"
        loaded = getattr(wrapper.context, "skills_loaded", None)
        if isinstance(loaded, list) and skill_name not in loaded:
            loaded.append(skill_name)
        return f"Loaded skill: {skill_name}\n\n{skill.content}"

    return load_skill


class SkillsMiddleware(PortMiddleware):
    """Inject the available-skills block + provide the ``load_skill`` tool."""

    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills
        self._tools = [make_load_skill_tool(skills)]

    def transform_instructions(self, run_ctx: Any, agent: Any, base: str) -> str:
        loaded = list(getattr(getattr(run_ctx, "context", None), "skills_loaded", []) or [])
        block = render_available_block(self.skills, loaded)
        return f"{block}\n\n{base}"

    def extra_tools(self) -> list[Any]:
        return list(self._tools)


__all__ = [
    "Skill",
    "SkillsMiddleware",
    "make_load_skill_tool",
    "render_available_block",
]
