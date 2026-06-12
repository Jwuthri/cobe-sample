"""``build_agent`` — compile an :class:`AgentConfig` into an SDK ``Agent``.

The whole config → ``agents.Agent`` mapping lives here:

  * model        → ``resolve_model_name`` + ``resolve_model_settings``
  * prompt       → system_prompt + instructions appended as bullet rules, then
                   folded through each middleware's ``transform_instructions`` to
                   produce the agent's **dynamic instructions** (so cart_anchor /
                   skills blocks re-render every run)
  * tools        → registry lookups + compiled HTTP tools + skill tools + delegates,
                   each wrapped with a composed ``is_enabled`` (empty_cart_guard)
  * skills       → a ``SkillsMiddleware`` (first, so its block leads)
  * middleware   → platform middleware factories, resolved by name; they also
                   contribute a per-run ``max_turns`` budget
  * output_format→ a raw-JSON-schema ``output_type`` adapter (or None)

``context`` / ``store`` are runtime args (passed to the run), not config.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping

from agents import Agent, AgentOutputSchemaBase

from openai_agent_v1.core.config import AgentConfig, CustomSkillSpec, RegistrySkillSpec, to_config
from openai_agent_v1.core.middleware import PortMiddleware
from openai_agent_v1.core.models import resolve_model_name, resolve_model_settings
from openai_agent_v1.core.registry import MIDDLEWARE, SKILLS
from openai_agent_v1.core.skills import Skill, SkillsMiddleware
from openai_agent_v1.core.tools import resolve_tools

_DEFAULT_MAX_TURNS = 12


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
            out.append(Skill(name=spec.name, description=spec.description, content=spec.skill))
    return out


def _compile_middleware(cfg: AgentConfig) -> list[PortMiddleware]:
    middleware: list[PortMiddleware] = []
    skills = _resolve_skills(cfg)
    if skills:
        middleware.append(SkillsMiddleware(skills))
    for spec in cfg.middleware:
        factory = MIDDLEWARE.get(spec.name)
        middleware.append(factory(**spec.params))
    return middleware


def _gates_tools(mws: list[PortMiddleware]) -> bool:
    return any(type(mw).tool_enabled is not PortMiddleware.tool_enabled for mw in mws)


def _apply_tool_gate(tool: Any, mws: list[PortMiddleware]) -> Any:
    """Return a copy of ``tool`` whose ``is_enabled`` ANDs the middleware gates.

    Copies (``dataclasses.replace``) so a shared registry tool is never mutated.
    """
    existing = getattr(tool, "is_enabled", True)
    name = getattr(tool, "name", "")

    async def predicate(wrapper: Any, agent: Any) -> bool:
        if callable(existing):
            ok = existing(wrapper, agent)
            ok = await ok if inspect.isawaitable(ok) else ok
        else:
            ok = bool(existing)
        if not ok:
            return False
        for mw in mws:
            if not mw.tool_enabled(wrapper, agent, name):
                return False
        return True

    try:
        return dataclasses.replace(tool, is_enabled=predicate)
    except Exception:  # not a dataclass / frozen — fall back to in-place
        try:
            tool.is_enabled = predicate
        except Exception:
            pass
        return tool


class _RawJSONSchemaOutput(AgentOutputSchemaBase):
    """Adapt a raw JSON-schema dict (config ``output_format``) to ``output_type``."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self._schema.get("title", "Output")

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return False

    def validate_json(self, json_str: str) -> Any:
        import json

        return json.loads(json_str)


def _make_instructions(base: str, mws: list[PortMiddleware]):
    """Dynamic-instructions callable: fold ``base`` through each middleware."""
    transforms = [
        mw for mw in mws if type(mw).transform_instructions is not PortMiddleware.transform_instructions
    ]
    if not transforms:
        return base

    def instructions(run_ctx: Any, agent: Any) -> str:
        prompt = base
        for mw in transforms:
            prompt = mw.transform_instructions(run_ctx, agent, prompt)
        return prompt

    return instructions


def build_agent(
    config: AgentConfig | Mapping[str, Any],
    *,
    context: Any | None = None,  # accepted for parity; the SDK binds context at run time
    store: Any | None = None,
    name: str | None = None,
    delegates: list[Any] | None = None,
) -> Agent:
    """Compile a declarative config (or raw dict) into an SDK ``Agent``.

    ``delegates`` are sub-agents-already-wrapped-as-tools, supplied at assembly
    time rather than declared in the config (they carry Python extractors, so they
    are not pure-JSON config). This is what makes an orchestrator:
    ``build_agent(ORCHESTRATOR, delegates=[...])``.
    """
    cfg = to_config(config)
    mws = _compile_middleware(cfg)

    tools = resolve_tools(cfg.tools)
    for mw in mws:
        tools.extend(mw.extra_tools())
    tools = list(tools) + list(delegates or [])

    if _gates_tools(mws):
        tools = [_apply_tool_gate(t, mws) for t in tools]

    base_prompt = _compose_prompt(cfg)
    instructions = _make_instructions(base_prompt, mws)

    budgets = [mw.max_turns for mw in mws if mw.max_turns is not None]
    max_turns = min(budgets) if budgets else _DEFAULT_MAX_TURNS

    output_type = _RawJSONSchemaOutput(cfg.output_format) if cfg.output_format else None

    agent = Agent(
        name=name or _slug(cfg.name),
        instructions=instructions,
        tools=tools,
        model=resolve_model_name(cfg.model.provider_model),
        model_settings=resolve_model_settings(cfg.model),
        output_type=output_type,
    )
    # Stash the resolved per-run turn budget so run sites can honor it.
    try:
        agent._port_max_turns = max_turns  # type: ignore[attr-defined]
    except Exception:
        pass
    return agent


def agent_max_turns(agent: Any, default: int = _DEFAULT_MAX_TURNS) -> int:
    return int(getattr(agent, "_port_max_turns", default) or default)


__all__ = ["build_agent", "agent_max_turns"]
