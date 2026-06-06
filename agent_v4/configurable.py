"""Declarative agent configuration → LangChain ``create_agent``.

One JSON-serializable :class:`AgentConfig` drives how a leaf agent is
built: model, prompt, tools, skills, guardrails, middleware, and
(optionally) structured output. ``build_agent(cfg)`` compiles it into the
exact same ``create_agent`` call the hand-written ``sops/*`` modules made
in agent_v2 — only now the *definition* is data, not code.

This is the agent-builder described in the design doc, adapted to v4 and
with the doc's rough edges fixed:

  * ``ModelConfig`` fields (temperature, max_tokens) are actually applied
    to the model — the doc passed only the model name to ``create_agent``.
  * ``build_agent`` forwards ``context_schema`` / ``checkpointer`` /
    ``store`` so cart-mutating tools and per-session memory keep working.
  * **Skills** are first-class (the checkout leaf is built on them), via a
    skill registry + an auto-attached ``SkillsMiddleware``.
  * Declarative ``HttpTool`` strips placeholder-consumed args from the
    request payload (the doc leaked e.g. ``api_token`` into the body).

What stays generic here: registries, specs, and the compile functions.
What the *platform* registers (the concrete tools/skills/middleware) lives
in :mod:`agent_v4.registry_defaults`; the concrete leaf configs live in
:mod:`agent_v4.leaves`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

import httpx
from langchain.agents.middleware import AgentMiddleware, PIIMiddleware
from langchain.agents.middleware.types import hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Registries
# =============================================================================
class Registry:
    """Name → item store for tools, skills, guardrail/middleware factories."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._items: dict[str, Any] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def register(self, name: str, item: Any, **meta: Any) -> None:
        self._items[name] = item
        self._meta[name] = meta

    def has(self, name: str) -> bool:
        return name in self._items

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise ValueError(
                f"Unknown {self.label}: {name!r}. Available: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def catalog(self) -> list[dict[str, Any]]:
        return [{"name": name, **meta} for name, meta in self._meta.items()]


TOOL_REGISTRY = Registry("tool")
SKILL_REGISTRY = Registry("skill")
GUARDRAIL_REGISTRY = Registry("guardrail")
MIDDLEWARE_REGISTRY = Registry("middleware")


def register_tool(tool: Any, *, label: str | None = None, description: str | None = None) -> None:
    """Register a LangChain tool object (must expose ``.name``)."""
    name = getattr(tool, "name", None)
    if not name:
        raise ValueError("Tool must have a .name attribute")
    meta: dict[str, Any] = {}
    if label:
        meta["label"] = label
    meta["description"] = description if description is not None else (
        getattr(tool, "description", "") or ""
    )
    TOOL_REGISTRY.register(name, tool, **meta)


def register_skill(skill: dict[str, Any]) -> None:
    """Register a skill dict (``{name, description, content, unlocks}``)."""
    name = skill.get("name")
    if not name:
        raise ValueError("Skill must have a 'name'")
    SKILL_REGISTRY.register(name, skill, description=skill.get("description", ""))


# =============================================================================
# Model
# =============================================================================
class ModelConfig(BaseModel):
    """How to build the chat model. Serializable; ``build()`` realizes it.

    ``model=None`` resolves to the env default (see ``agent_v4.llm``). A
    ``provider:model`` form (e.g. ``"openai:gpt-4.1-mini"``) is accepted;
    the bare OpenAI form is the common case and routes through
    ``ChatOpenAI`` to match v2 exactly.
    """

    model: str | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def _resolved_name(self) -> str:
        if self.model:
            return self.model
        from agent_v4.llm.openai_llm import model_name

        return model_name()

    def build(self) -> Any:
        name = self._resolved_name()
        if ":" in name and not name.startswith("openai:"):
            # Non-OpenAI provider → defer to langchain's universal factory.
            return init_chat_model(
                name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.extra,
            )
        if name.startswith("openai:"):
            name = name.split(":", 1)[1]
        kwargs: dict[str, Any] = {"model": name, "temperature": self.temperature}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        kwargs.update(self.extra)
        return ChatOpenAI(**kwargs)


# =============================================================================
# Tool specifications (polymorphic)
# =============================================================================
class _ToolBase(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True


class RegistryTool(_ToolBase):
    """Reference a platform-registered tool by name."""

    kind: Literal["registry"] = "registry"


class HttpTool(_ToolBase):
    """Customer-specific declarative HTTP tool.

    ``parameters`` is a JSON Schema for arguments the LLM supplies.
    ``{placeholders}`` in url/headers are filled from those arguments and
    are NOT re-sent in the request body/query (so secrets passed as e.g.
    ``Authorization: Bearer {api_token}`` don't leak into the payload).
    """

    kind: Literal["http"] = "http"
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 20.0


ToolSpec = Annotated[Union[RegistryTool, HttpTool], Field(discriminator="kind")]

_PLACEHOLDER_KEYS = re.compile(r"\{(\w+)\}")


def _template_keys(*templates: str) -> set[str]:
    keys: set[str] = set()
    for t in templates:
        keys.update(_PLACEHOLDER_KEYS.findall(t))
    return keys


def _format_template(template: str, args: dict[str, Any]) -> str:
    try:
        return template.format(**args)
    except (KeyError, IndexError):
        return template


def _compile_http_tool(spec: HttpTool) -> StructuredTool:
    consumed = _template_keys(spec.url, *spec.headers.values())

    def _call(**kwargs: Any) -> str:
        url = _format_template(spec.url, kwargs)
        headers = {k: _format_template(v, kwargs) for k, v in spec.headers.items()}
        # Don't echo placeholder-only args (e.g. api_token) back in the payload.
        payload = {k: v for k, v in kwargs.items() if k not in consumed}
        with httpx.Client(timeout=spec.timeout_s) as client:
            if spec.method == "GET":
                response = client.get(url, headers=headers, params=payload)
            else:
                response = client.request(spec.method, url, headers=headers, json=payload)
        response.raise_for_status()
        return response.text

    return StructuredTool.from_function(
        func=_call,
        name=spec.name,
        description=spec.description or f"HTTP {spec.method} tool",
        args_schema=spec.parameters or None,
    )


def _compile_tools(cfg: AgentConfig) -> list[Any]:
    tools: list[Any] = []
    for spec in cfg.tools:
        if not spec.enabled:
            continue
        if isinstance(spec, RegistryTool):
            tools.append(TOOL_REGISTRY.get(spec.name))
        else:
            tools.append(_compile_http_tool(spec))
    return tools


# =============================================================================
# Skills (declarative reference to the load-on-demand skill registry)
# =============================================================================
class SkillSpec(BaseModel):
    """Reference a registered skill by name, or inline a full skill body."""

    kind: Literal["registry", "inline"] = "registry"
    name: str
    description: str = ""
    content: str = ""
    unlocks: list[str] = Field(default_factory=list)
    enabled: bool = True


def _resolve_skills(cfg: AgentConfig) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for spec in cfg.skills:
        if not spec.enabled:
            continue
        if spec.kind == "registry":
            resolved.append(SKILL_REGISTRY.get(spec.name))
        else:
            resolved.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "content": spec.content,
                    "unlocks": list(spec.unlocks),
                }
            )
    return resolved


def _compile_skills_middleware(cfg: AgentConfig) -> AgentMiddleware | None:
    skills = _resolve_skills(cfg)
    if not skills:
        return None
    # Lazy import keeps this module importable without the domain middleware.
    from agent_v4.middleware import SkillsMiddleware

    return SkillsMiddleware(skills)


# =============================================================================
# Guardrails (safety middleware: blocklist / llm judge / PII)
# =============================================================================
class Guardrail(BaseModel):
    """Generic guardrail spec resolved via :data:`GUARDRAIL_REGISTRY`."""

    type: str
    action: Literal["block", "redact", "mask", "flag"] = "block"
    on_input: bool = True
    on_output: bool = False
    message: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def _last_message_text(state: dict[str, Any], role: str) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and role == "human":
            return _message_content(message)
        if isinstance(message, AIMessage) and role == "ai":
            return _message_content(message)
        if getattr(message, "type", None) == role:
            return _message_content(message)
    return ""


class BlocklistGuardrail(AgentMiddleware):
    """Phrase / regex guardrail."""

    def __init__(
        self,
        phrases: list[str] | None = None,
        patterns: list[str] | None = None,
        action: str = "block",
        message: str | None = None,
        on_input: bool = True,
        on_output: bool = False,
    ) -> None:
        super().__init__()
        self.phrases = tuple(p.lower() for p in (phrases or []))
        self.patterns = tuple(re.compile(p, re.IGNORECASE) for p in (patterns or []))
        self.action = action
        self.message = message or "I can't help with that request."
        self.on_input = on_input
        self.on_output = on_output

    def _matches(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in self.phrases):
            return True
        return any(pattern.search(text) for pattern in self.patterns)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Runtime) -> dict[str, Any] | None:
        if self.on_input and self.action in ("block", "flag"):
            if self._matches(_last_message_text(state, "human")):
                return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: dict[str, Any], runtime: Runtime) -> dict[str, Any] | None:
        if self.on_output and self.action in ("block", "flag"):
            if self._matches(_last_message_text(state, "ai")):
                return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None


class _PolicyVerdict(BaseModel):
    violates: bool = Field(description="True if the message breaks the policy.")
    reason: str = Field(description="Short reason for the decision.")


_POLICY_JUDGE_SYSTEM = """You are a content policy checker.
Decide if the message violates this policy:

<policy>
{policy}
</policy>

Judge by meaning, not keywords. Set violates=true only if it clearly breaks the policy."""


class LLMGuardrail(AgentMiddleware):
    """Semantic guardrail: a small LLM judges against a natural-language policy."""

    def __init__(
        self,
        policy: str,
        model: str = "gpt-4.1-mini",
        action: str = "block",
        message: str | None = None,
        on_input: bool = True,
        on_output: bool = False,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.action = action
        self.message = message or "I'm not able to help with that topic."
        self.on_input = on_input
        self.on_output = on_output
        self._model = model
        self._judge_cache: Any = None

    def _judge(self) -> Any:
        if self._judge_cache is None:
            self._judge_cache = init_chat_model(
                self._model, temperature=0
            ).with_structured_output(_PolicyVerdict)
        return self._judge_cache

    def _violates(self, text: str) -> bool:
        if not text.strip():
            return False
        verdict: _PolicyVerdict = self._judge().invoke(
            [
                SystemMessage(content=_POLICY_JUDGE_SYSTEM.format(policy=self.policy)),
                HumanMessage(content=text),
            ]
        )
        return verdict.violates

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Runtime) -> dict[str, Any] | None:
        if self.on_input and self.action in ("block", "flag"):
            if self._violates(_last_message_text(state, "human")):
                return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: dict[str, Any], runtime: Runtime) -> dict[str, Any] | None:
        if self.on_output and self.action in ("block", "flag"):
            if self._violates(_last_message_text(state, "ai")):
                return {"messages": [AIMessage(content=self.message)], "jump_to": "end"}
        return None


def _guardrail_pii(gr: Guardrail) -> AgentMiddleware:
    strategy = "redact" if gr.action == "redact" else "mask"
    return PIIMiddleware(
        gr.params.get("entity", "email"),
        strategy=strategy,
        apply_to_input=gr.on_input,
    )


def _guardrail_blocklist(gr: Guardrail) -> AgentMiddleware:
    return BlocklistGuardrail(
        phrases=gr.params.get("phrases"),
        patterns=gr.params.get("patterns"),
        action=gr.action,
        message=gr.message,
        on_input=gr.on_input,
        on_output=gr.on_output,
    )


def _guardrail_llm_judge(gr: Guardrail) -> AgentMiddleware:
    policy = gr.params.get("policy")
    if not policy:
        raise ValueError("llm_judge guardrail requires params.policy")
    return LLMGuardrail(
        policy=policy,
        model=gr.params.get("model", "gpt-4.1-mini"),
        action=gr.action,
        message=gr.message,
        on_input=gr.on_input,
        on_output=gr.on_output,
    )


GUARDRAIL_REGISTRY.register("pii", _guardrail_pii, label="PII redaction or masking", category="safety")
GUARDRAIL_REGISTRY.register("blocklist", _guardrail_blocklist, label="Phrase or regex blocklist", category="safety")
GUARDRAIL_REGISTRY.register("llm_judge", _guardrail_llm_judge, label="LLM semantic policy check", category="safety")


def _compile_guardrails(cfg: AgentConfig) -> list[AgentMiddleware]:
    return [
        GUARDRAIL_REGISTRY.get(gr.type)(gr) for gr in cfg.guardrails if gr.enabled
    ]


# =============================================================================
# Middleware (platform behavior, resolved via MIDDLEWARE_REGISTRY)
# =============================================================================
class MiddlewareSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


def _compile_middleware(cfg: AgentConfig) -> list[Any]:
    middleware: list[Any] = []
    for spec in cfg.middleware:
        if not spec.enabled:
            continue
        factory = MIDDLEWARE_REGISTRY.get(spec.name)
        middleware.append(factory(**spec.params))
    return middleware


# =============================================================================
# Agent configuration (single object for UI / DB / API)
# =============================================================================
class AgentConfig(BaseModel):
    """Full declarative definition of one leaf agent.

    Serialize with ``model_dump()`` / ``model_validate()`` for storage and
    APIs. ``context_schema`` / ``checkpointer`` / ``store`` are *runtime*
    concerns passed to :func:`build_agent`, not config fields, so the
    config stays JSON-serializable.
    """

    name: str
    description: str = ""
    model: ModelConfig = Field(default_factory=ModelConfig)

    system_prompt: str
    instructions: list[str] = Field(
        default_factory=list,
        description="Extra bullet rules appended under 'Additional instructions'.",
    )

    tools: list[ToolSpec] = Field(default_factory=list)
    skills: list[SkillSpec] = Field(default_factory=list)
    guardrails: list[Guardrail] = Field(default_factory=list)
    middleware: list[MiddlewareSpec] = Field(default_factory=list)

    output_format: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for structured output (nested objects allowed). None = plain text.",
    )


@dataclass
class AgentCatalog:
    """Everything an agent-builder UI can offer in pickers."""

    tools: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    guardrails: list[dict[str, Any]]
    middleware: list[dict[str, Any]]


def build_catalog() -> AgentCatalog:
    return AgentCatalog(
        tools=TOOL_REGISTRY.catalog(),
        skills=SKILL_REGISTRY.catalog(),
        guardrails=GUARDRAIL_REGISTRY.catalog(),
        middleware=MIDDLEWARE_REGISTRY.catalog(),
    )


# =============================================================================
# Builder
# =============================================================================
def _compose_prompt(cfg: AgentConfig) -> str:
    prompt = cfg.system_prompt.strip()
    if cfg.instructions:
        bullets = "\n".join(f"- {instruction}" for instruction in cfg.instructions)
        prompt += f"\n\n## Additional instructions\n{bullets}"
    return prompt


def build_agent(
    cfg: AgentConfig,
    *,
    checkpointer: Any | None = None,
    store: Any | None = None,
    context_schema: Any | None = None,
) -> Any:
    """Compile :class:`AgentConfig` into a LangChain agent graph (a leaf).

    Middleware order mirrors the v2 hand-written agents:
    ``[skills?, *guardrails, *middleware]`` — skills first (so the
    available-skills block is injected before guardrails/observability run).
    """
    from langchain.agents import create_agent

    compiled_middleware: list[Any] = []
    skills_mw = _compile_skills_middleware(cfg)
    if skills_mw is not None:
        compiled_middleware.append(skills_mw)
    compiled_middleware.extend(_compile_guardrails(cfg))
    compiled_middleware.extend(_compile_middleware(cfg))

    return create_agent(
        model=cfg.model.build(),
        tools=_compile_tools(cfg),
        system_prompt=_compose_prompt(cfg),
        response_format=cfg.output_format,
        middleware=compiled_middleware,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
    )
