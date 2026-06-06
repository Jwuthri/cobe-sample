"""SkillsMiddleware — owns the skill registry, the load_skill tool, and
the per-call injection of an 'Available skills' block into the prompt.

The agent's ``state_schema`` is extended with ``skills_loaded:
list[str]`` so constrained tools can read it via
``runtime.state['skills_loaded']``.
"""

from __future__ import annotations

from typing import Annotated, Any

from agent_v4.skills.base import Skill, make_load_skill_tool, render_available_block
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from typing_extensions import NotRequired, TypedDict


def _skills_loaded_reducer(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Append-without-duplicates reducer."""
    out = list(left or [])
    for name in right or []:
        if name not in out:
            out.append(name)
    return out


class SkillsAgentState(AgentState):
    """AgentState extension exposing ``skills_loaded`` to constrained tools."""

    skills_loaded: NotRequired[Annotated[list[str], _skills_loaded_reducer]]


class SkillsMiddleware(AgentMiddleware):
    """Inject available-skills block + provide the ``load_skill`` tool.

    Usage:
        create_agent(model, tools=..., middleware=[SkillsMiddleware(SKILLS), ...])

    The middleware:
      - registers ``load_skill`` as an extra tool (via ``self.tools``)
      - extends state with ``skills_loaded``
      - on ``before_model``, appends an "Available skills" block to
        the conversation as a SystemMessage so the model knows what's
        available and what's currently loaded
    """

    state_schema = SkillsAgentState

    def __init__(self, skills: list[Skill]) -> None:
        super().__init__()
        self.skills = skills
        self.tools = [make_load_skill_tool(skills)]

    def before_model(self, state: SkillsAgentState, runtime: Runtime) -> dict[str, Any] | None:
        loaded = state.get("skills_loaded", []) or []
        block = render_available_block(self.skills, loaded)
        # Append as a SystemMessage so it's clearly meta-information.
        return {"messages": [SystemMessage(content=block)]}
