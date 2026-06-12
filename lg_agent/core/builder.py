"""``build_agent`` — compile an :class:`AgentConfig` into a ``create_agent`` graph.

The whole config → ``create_agent`` mapping lives here:

  * model        → ``resolve_model`` (init_chat_model with temperature/max_tokens)
  * system_prompt → prompt + ``instructions`` appended as bullet rules
  * tools        → registry lookups + compiled HTTP tools (+ the ``load_skill`` tool
                   contributed by SkillsMiddleware)
  * skills       → a single ``SkillsMiddleware`` (first, so its block is visible)
  * guardrails   → safety middleware
  * middleware   → platform middleware factories, resolved by name
  * output_format → ``response_format`` (raw JSON-schema dict)

Middleware order: ``[skills?, *guardrails, *middleware]``. The runtime args
``context_schema`` / ``checkpointer`` / ``store`` / ``delegates`` are NOT config —
they are wiring passed here at assembly time.
"""

from __future__ import annotations

from typing import Any, Mapping

from langchain.agents import create_agent

from lg_agent.core.config import AgentConfig, CustomSkillSpec, RegistrySkillSpec, to_config
from lg_agent.core.guardrails import compile_guardrail_middleware
from lg_agent.core.model import resolve_model
from lg_agent.core.registry import MIDDLEWARE, SKILLS
from lg_agent.core.skills import Skill, SkillsMiddleware
from lg_agent.core.tools import resolve_tools


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_") or "agent"


def _compose_prompt(cfg: AgentConfig) -> str:
    prompt = cfg.system_prompt.strip()
    if cfg.instructions:
        bullets = "\n".join(f"- {i}" for i in cfg.instructions)
        prompt += f"\n\n## Additional instructions\n{bullets}"
    return prompt


def _resolve_skills(cfg: AgentConfig) -> list[Skill]:
    out: list[Skill] = []
    for spec in cfg.skills:
        if isinstance(spec, RegistrySkillSpec):
            out.append(SKILLS.get(spec.name))
        elif isinstance(spec, CustomSkillSpec):
            out.append(
                Skill(
                    name=spec.name,
                    description=spec.description,
                    content=spec.skill,
                    unlocks=spec.unlocks,
                )
            )
    return out


def _compile_middleware(cfg: AgentConfig) -> list[Any]:
    middleware: list[Any] = []
    skills = _resolve_skills(cfg)
    if skills:
        middleware.append(SkillsMiddleware(skills))
    middleware.extend(compile_guardrail_middleware(cfg.guardrails))
    for spec in cfg.middleware:
        factory = MIDDLEWARE.get(spec.name)
        middleware.append(factory(**spec.params))
    return middleware


def build_agent(
    config: AgentConfig | Mapping[str, Any],
    *,
    context_schema: Any | None = None,
    checkpointer: Any | None = None,
    store: Any | None = None,
    name: str | None = None,
    delegates: list[Any] | None = None,
) -> Any:
    """Compile a declarative config (or raw dict) into a LangChain agent graph.

    ``delegates`` are sub-agents-already-wrapped-as-tools, supplied at assembly time
    rather than declared in the config (a sub-agent carries Python hooks, so it is
    not pure-JSON config). This is what makes an orchestrator:
    ``build_agent(ORCHESTRATOR, delegates=[...])``.
    """
    cfg = to_config(config)
    tools = resolve_tools(cfg.tools) + list(delegates or [])
    return create_agent(
        model=resolve_model(cfg.model),
        tools=tools,
        system_prompt=_compose_prompt(cfg),
        response_format=cfg.output_format,
        middleware=_compile_middleware(cfg),
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        name=name or _slug(cfg.name),
    )


__all__ = ["build_agent"]
