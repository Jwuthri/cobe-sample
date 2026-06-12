"""``AgentSpec`` → ``agno.Agent``. One declarative definition per agent.

An :class:`AgentSpec` is the whole definition of an agent as data: its voice
(``prompt``), its actions (``tools``), its state-derived instructions
(``skills``), and its model knobs. :func:`build_agent` compiles that into a
configured ``agno.Agent``.

The skills wiring is the interesting part: if a spec declares skills, the agent's
``instructions`` become a *callable* that, on every run, concatenates the static
prompt with each skill rendered against the live cart (read from
``run_context.dependencies["ctx"]``). That is how the checkout progress anchor
stays fresh without a hand-maintained thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agno.agent import Agent
from agno.run import RunContext

from agno_agent_v1.agent.models import resolve_model
from agno_agent_v1.agent.skills import Skill


@dataclass(frozen=True)
class AgentSpec:
    """A single agent's full definition, as data."""

    name: str
    description: str  # one-liner; for a sub-agent this is its routing surface
    prompt: str  # the static system prompt (voice)
    tools: list[Any] = field(default_factory=list)  # action functions
    skills: list[Skill] = field(default_factory=list)  # state-derived instructions
    model_id: str | None = None
    temperature: float | None = None
    tool_call_limit: int | None = None


def _skill_instructions(spec: AgentSpec):
    """Build the callable-instructions function for a spec that declares skills."""

    def instructions(run_context: RunContext) -> str:
        ctx = (run_context.dependencies or {}).get("ctx")
        parts = [spec.prompt]
        if ctx is not None:
            cart = ctx.cart_service.cart
            parts.extend(skill.render(cart) for skill in spec.skills)
        return "\n\n".join(parts)

    return instructions


def build_agent(spec: AgentSpec, *, tools: list[Any] | None = None) -> Agent:
    """Compile ``spec`` into a configured Agno Agent.

    ``tools`` overrides ``spec.tools`` (used for the orchestrator, whose tools are
    the dynamically-built sub-agent wrappers rather than static functions).
    """
    instructions = _skill_instructions(spec) if spec.skills else spec.prompt
    resolved_tools = tools if tools is not None else list(spec.tools)
    return Agent(
        name=spec.name,
        instructions=instructions,
        model=resolve_model(spec.model_id, spec.temperature),
        tools=resolved_tools or None,
        tool_call_limit=spec.tool_call_limit,
        # Stateless + plain: the session owns the transcript, blocks are built
        # deterministically, and we don't want Agno injecting markdown/knowledge
        # scaffolding into the carefully-tuned prompts.
        markdown=False,
        telemetry=False,
        search_knowledge=False,
        add_history_to_context=False,
    )


__all__ = ["AgentSpec", "build_agent"]
