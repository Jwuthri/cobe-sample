"""Skills — instruction bundles the model can pull into context on demand.

A skill is a one-line ``description`` (always visible in the system prompt) plus
a full ``content`` body the model loads via the ``load_skill`` tool. Unlike v4,
skills here are pure instructions — there is no ``unlocks`` tool-gating (the
shopping checkout drives its flow from cart state, not a permission system).

The "Available skills" block is injected **transiently** via ``wrap_model_call``
(``request.override(messages=...)``) — re-rendered fresh every model call with
up-to-date "(loaded)" markers, never persisted into state (v4's ``before_model``
append accumulated a copy per call).
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired


class Skill(BaseModel):
    name: str
    description: str = ""
    content: str


def _skills_loaded_reducer(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Append-without-duplicates reducer for ``skills_loaded``."""
    out = list(left or [])
    for name in right or []:
        if name not in out:
            out.append(name)
    return out


class SkillsAgentState(AgentState):
    """AgentState extension exposing ``skills_loaded`` to tools/middleware."""

    skills_loaded: NotRequired[Annotated[list[str], _skills_loaded_reducer]]


def make_load_skill_tool(skills: list[Skill]):
    """Factory: a ``load_skill`` tool bound to a given skill set."""
    by_name = {s.name: s for s in skills}

    @tool
    def load_skill(
        skill_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Load a skill by name so its full instructions become available.

        Available skill names are listed in the system prompt under
        'Available skills'.
        """
        skill = by_name.get(skill_name)
        if skill is None:
            options = ", ".join(by_name.keys())
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"unknown skill '{skill_name}'. options: {options}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        return Command(
            update={
                "skills_loaded": [skill_name],
                "messages": [
                    ToolMessage(
                        content=f"Loaded skill: {skill_name}\n\n{skill.content}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return load_skill


def render_available_block(skills: list[Skill], loaded: list[str]) -> str:
    lines = ["Available skills (call load_skill to load):"]
    for s in skills:
        marker = " (loaded)" if s.name in loaded else ""
        lines.append(f"  - {s.name}{marker}: {s.description}")
    return "\n".join(lines)


class SkillsMiddleware(AgentMiddleware):
    """Provide the ``load_skill`` tool + inject the available-skills block.

    The block is prepended to the model's messages per call via ``wrap_model_call``
    (transient — nothing is written back to state).
    """

    state_schema = SkillsAgentState

    def __init__(self, skills: list[Skill]) -> None:
        super().__init__()
        self.skills = skills
        self.tools = [make_load_skill_tool(skills)]

    def _apply(self, request: Any) -> Any:
        loaded = (getattr(request, "state", None) or {}).get("skills_loaded", []) or []
        block = render_available_block(self.skills, loaded)
        return request.override(messages=[SystemMessage(content=block), *request.messages])

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._apply(request))


__all__ = [
    "Skill",
    "SkillsAgentState",
    "SkillsMiddleware",
    "make_load_skill_tool",
    "render_available_block",
]
