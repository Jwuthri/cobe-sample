"""Skill type + load_skill tool.

A skill is a unit of specialized instructions for a sub-task. The
agent sees one-line descriptions of all available skills in its
system prompt, then calls ``load_skill(name)`` to pull the full
content into its context. Loading a skill also unlocks the tools
listed in that skill's ``unlocks`` field — every constrained tool
checks ``runtime.state["skills_loaded"]`` and refuses if its required
skill isn't loaded.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command


class Skill(TypedDict):
    name: str
    description: str  # 1 line, always in the system prompt
    content: str  # full body, loaded on demand
    unlocks: list[str]  # tool names this skill makes callable


def make_load_skill_tool(skills: list[Skill]):
    """Factory: returns a ``load_skill`` tool bound to a given skill set.

    ``tool_call_id`` is annotated with ``InjectedToolCallId`` so LangChain
    injects the *real* tool_call_id at runtime instead of letting the
    model invent one. This is required because we return a ``Command``
    that includes a ``ToolMessage``: the ToolMessage's id must match the
    AIMessage's tool_call id, or ToolNode raises a validation error.
    """

    by_name = {s["name"]: s for s in skills}

    @tool
    def load_skill(
        skill_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Load a skill by name so its instructions and tools become available.

        Use this BEFORE calling any tool the skill unlocks. Available
        skill names are listed in the system prompt under
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
                        content=(
                            f"Loaded skill: {skill_name}\n"
                            f"Unlocks tools: {', '.join(skill['unlocks']) or '(none)'}\n\n"
                            f"{skill['content']}"
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return load_skill


def render_available_block(skills: list[Skill], loaded: list[str]) -> str:
    """Format the skills section that's prepended to the system prompt."""
    lines = ["Available skills (call load_skill to load):"]
    for s in skills:
        marker = " (loaded)" if s["name"] in loaded else ""
        lines.append(f"  - {s['name']}{marker}: {s['description']}")
    return "\n".join(lines)
