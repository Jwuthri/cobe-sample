"""The declarative config contract — one ``AgentConfig`` defines one agent.

This is the whole public schema a tenant writes (the ``EXAMPLE_AGENT_CONFIG`` in
:mod:`agent_v4_1.examples` validates against it verbatim). It is pure data:
JSON-serializable, ``model_dump()`` / ``model_validate()`` round-trips, and
``extra="forbid"`` everywhere so a typo'd key fails loudly at parse time instead
of silently doing nothing.

What it intentionally drops from agent_v4's schema:

  * no ``enabled`` flags — a config you don't want is a config you don't pass.
  * no skill ``unlocks`` / tool gating — skills are instruction bundles, not a
    permission system (the shopping checkout drives its flow from cart state).
  * ``ModelConfig.provider_model`` (a ``"provider:model"`` string) replaces v4's
    bare ``model`` + ``ChatOpenAI`` special-casing; everything routes through
    :func:`langchain.chat_models.init_chat_model`.

Runtime concerns (``context_schema`` / ``checkpointer`` / ``store``) are NOT
config fields — they're passed to :func:`agent_v4_1.core.factory.build_agent`,
keeping the config serializable.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_STRICT = ConfigDict(extra="forbid")
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


# =============================================================================
# Model
# =============================================================================
class ModelConfig(BaseModel):
    """How to build the chat model.

    ``provider_model`` is a ``"provider:model"`` string (e.g. ``"openai:gpt-5-mini"``)
    passed straight to ``init_chat_model``. ``None`` resolves to the env default
    (see :func:`agent_v4_1.core.models.default_provider_model`).
    """

    model_config = _STRICT

    provider_model: str | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)

    @field_validator("provider_model")
    @classmethod
    def _require_provider_prefix(cls, v: str | None) -> str | None:
        if v is not None and ":" not in v:
            raise ValueError(
                "provider_model must be 'provider:model' (e.g. 'openai:gpt-4.1-mini'); "
                f"got {v!r}"
            )
        return v


# =============================================================================
# Tools (discriminated union on "kind")
# =============================================================================
class RegistryToolSpec(BaseModel):
    """Reference a platform-registered tool by name."""

    model_config = _STRICT

    kind: Literal["registry"] = "registry"
    name: str = Field(min_length=1)


class HttpToolSpec(BaseModel):
    """A tenant-defined declarative HTTP tool.

    ``parameters`` is the JSON Schema for the args the model supplies.
    ``{placeholder}`` tokens in ``url`` / ``headers`` are filled from those args
    and stripped from the request body (so a secret passed as
    ``Authorization: Bearer {api_token}`` never leaks into the payload).
    """

    model_config = _STRICT

    kind: Literal["http"] = "http"
    name: str = Field(min_length=1)
    description: str = ""
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    timeout_s: float = Field(20.0, gt=0)

    @field_validator("url")
    @classmethod
    def _url_scheme(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def _placeholders_declared(self) -> "HttpToolSpec":
        placeholders: set[str] = set(_PLACEHOLDER.findall(self.url))
        for value in self.headers.values():
            placeholders.update(_PLACEHOLDER.findall(value))
        declared = set((self.parameters or {}).get("properties", {}).keys())
        missing = placeholders - declared
        if missing:
            raise ValueError(
                f"http tool {self.name!r}: url/headers reference {sorted(missing)} "
                "which are not declared in parameters.properties (the model could "
                "never fill them)"
            )
        return self


ToolSpec = Annotated[Union[RegistryToolSpec, HttpToolSpec], Field(discriminator="kind")]


# =============================================================================
# Skills (discriminated union on "kind")
# =============================================================================
class RegistrySkillSpec(BaseModel):
    """Reference a platform-registered skill by name."""

    model_config = _STRICT

    kind: Literal["registry"] = "registry"
    name: str = Field(min_length=1)


class CustomSkillSpec(BaseModel):
    """Inline a skill: a one-line ``description`` + the full ``skill`` body."""

    model_config = _STRICT

    kind: Literal["custom"] = "custom"
    name: str = Field(min_length=1)
    description: str = ""
    skill: str


SkillSpec = Annotated[Union[RegistrySkillSpec, CustomSkillSpec], Field(discriminator="kind")]


# =============================================================================
# Guardrails
# =============================================================================
class GuardrailSpec(BaseModel):
    """A safety rule resolved via :data:`agent_v4_1.core.registry.GUARDRAILS`.

    ``on_input`` gates the user's message; ``on_output`` gates the agent's reply.
    A rule with no flag set defaults to input-only (the common case).
    """

    model_config = _STRICT

    type: str
    action: Literal["block", "redact", "mask", "flag"] = "block"
    on_input: bool = True
    on_output: bool = False
    message: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Middleware
# =============================================================================
class MiddlewareSpec(BaseModel):
    """A platform middleware factory + its kwargs, resolved by name."""

    model_config = _STRICT

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Agent
# =============================================================================
class AgentConfig(BaseModel):
    """The full declarative definition of one agent."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    description: str = ""
    system_prompt: str = Field(min_length=1)
    instructions: list[str] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=ModelConfig)
    skills: list[SkillSpec] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    guardrails: list[GuardrailSpec] = Field(default_factory=list)
    middleware: list[MiddlewareSpec] = Field(default_factory=list)
    output_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _unique_names_and_schema(self) -> "AgentConfig":
        tool_names = [t.name for t in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("duplicate tool names in config")
        skill_names = [s.name for s in self.skills]
        if len(skill_names) != len(set(skill_names)):
            raise ValueError("duplicate skill names in config")
        if self.output_format is not None:
            if self.output_format.get("type") != "object" or "properties" not in self.output_format:
                raise ValueError(
                    "output_format must be a JSON-schema object "
                    '(\'{"type": "object", "properties": {...}}\')'
                )
        return self


def to_config(config: "AgentConfig | dict[str, Any]") -> AgentConfig:
    """Coerce a raw dict (or an existing ``AgentConfig``) into a validated config."""
    if isinstance(config, AgentConfig):
        return config
    return AgentConfig.model_validate(config)


__all__ = [
    "ModelConfig",
    "RegistryToolSpec",
    "HttpToolSpec",
    "ToolSpec",
    "RegistrySkillSpec",
    "CustomSkillSpec",
    "SkillSpec",
    "GuardrailSpec",
    "MiddlewareSpec",
    "AgentConfig",
    "to_config",
]
