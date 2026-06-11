"""The declarative config contract — one ``AgentConfig`` defines one agent.

Ported from agent_v4_1's schema and retargeted at Agno. It stays pure data:
JSON-serializable, ``model_dump()`` / ``model_validate()`` round-trips, and
``extra="forbid"`` everywhere so a typo'd key fails loudly at parse time.

What changed from agent_v4_1 for the Agno port:

  * added ``id`` + ``role`` — Agno members are addressed by a stable ``id`` (the
    router's delegate target) and described to the leader by ``role``;
  * ``middleware`` is gone — its four jobs map onto native Agno features:
    ``tool_call_limit`` is a first-class field here, the checkout progress anchor
    rides ``session_state``, tool/step observability comes off the event stream,
    and the empty-cart guard is enforced by prompt + tool no-ops;
  * ``skills`` is gone — the shopping checkout drives its flow from cart state,
    so the load_skill chain never paid for itself (already true in v4_1).
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

    ``provider_model`` is a ``"provider:model"`` string (e.g. ``"openai:gpt-4.1-mini"``)
    or a bare OpenAI model id. ``None`` resolves to the env default (see
    :func:`agent_agno_v1.core.models.default_model_id`).
    """

    model_config = _STRICT

    provider_model: str | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)

    @field_validator("provider_model")
    @classmethod
    def _require_provider_prefix(cls, v: str | None) -> str | None:
        if v is not None and ":" in v and not v.split(":", 1)[1]:
            raise ValueError(f"provider_model has an empty model after ':'; got {v!r}")
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
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
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
# Guardrails
# =============================================================================
class GuardrailSpec(BaseModel):
    """A safety rule resolved via :data:`agent_agno_v1.core.registry.GUARDRAILS`.

    ``on_input`` gates the user's message (pre-flight, before the team runs);
    ``on_output`` gates the agent's reply. A rule with no flag set defaults to
    input-only (the common case).
    """

    model_config = _STRICT

    type: str
    action: Literal["block", "redact", "mask", "flag"] = "block"
    on_input: bool = True
    on_output: bool = False
    message: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Agent
# =============================================================================
class AgentConfig(BaseModel):
    """The full declarative definition of one agent (an Agno member or the leader)."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    id: str | None = None
    role: str = ""
    description: str = ""
    system_prompt: str = Field(min_length=1)
    instructions: list[str] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: list[ToolSpec] = Field(default_factory=list)
    guardrails: list[GuardrailSpec] = Field(default_factory=list)
    tool_call_limit: int | None = Field(None, gt=0)
    output_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _unique_names_and_schema(self) -> "AgentConfig":
        tool_names = [t.name for t in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("duplicate tool names in config")
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
    "GuardrailSpec",
    "AgentConfig",
    "to_config",
]
