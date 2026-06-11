"""The config contract — anchored on EXAMPLE_AGENT_CONFIG accepting verbatim."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_v4_1.core.config import (
    AgentConfig,
    CustomSkillSpec,
    GuardrailSpec,
    HttpToolSpec,
    RegistryToolSpec,
)
from agent_v4_1.examples import EXAMPLE_AGENT_CONFIG


def test_example_config_validates_verbatim():
    cfg = AgentConfig.model_validate(EXAMPLE_AGENT_CONFIG)
    assert cfg.name == "Acme Support"
    assert cfg.model.provider_model == "openai:gpt-5-mini"
    assert [t.name for t in cfg.tools] == [
        "check_order_status",
        "create_support_ticket",
        "create_zendesk_ticket",
    ]
    # discriminated unions dispatched by "kind"
    assert isinstance(cfg.tools[0], RegistryToolSpec)
    assert isinstance(cfg.tools[2], HttpToolSpec)
    assert isinstance(cfg.skills[0], CustomSkillSpec)
    assert cfg.skills[0].skill == "long text"


def test_example_config_round_trips():
    cfg = AgentConfig.model_validate(EXAMPLE_AGENT_CONFIG)
    again = AgentConfig.model_validate(cfg.model_dump())
    assert again == cfg


def test_guardrail_flag_defaults():
    # pii entry omits on_input/on_output → input-only by default
    cfg = AgentConfig.model_validate(EXAMPLE_AGENT_CONFIG)
    pii = cfg.guardrails[0]
    assert (pii.type, pii.action, pii.on_input, pii.on_output) == ("pii", "redact", True, False)


def test_extra_forbid_rejects_typos():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"name": "x", "system_prompt": "y", "toolz": []})


def test_http_placeholder_must_be_declared():
    with pytest.raises(ValidationError):
        HttpToolSpec(
            kind="http",
            name="t",
            url="https://api/{api_token}/x",
            parameters={"type": "object", "properties": {}},
        )
    # declared → ok
    HttpToolSpec(
        kind="http",
        name="t",
        url="https://api/{api_token}/x",
        parameters={"type": "object", "properties": {"api_token": {"type": "string"}}},
    )


def test_http_url_scheme_validated():
    with pytest.raises(ValidationError):
        HttpToolSpec(kind="http", name="t", url="ftp://nope")


def test_provider_model_requires_prefix():
    from agent_v4_1.core.config import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(provider_model="gpt-4.1-mini")  # missing provider:
    ModelConfig(provider_model="openai:gpt-4.1-mini")  # ok
    ModelConfig(provider_model=None)  # ok → resolves to env default


def test_duplicate_tool_names_rejected():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(
            {
                "name": "x",
                "system_prompt": "y",
                "tools": [
                    {"kind": "registry", "name": "dup"},
                    {"kind": "registry", "name": "dup"},
                ],
            }
        )


def test_output_format_must_be_object_schema():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(
            {"name": "x", "system_prompt": "y", "output_format": {"type": "string"}}
        )


def test_guardrail_spec_defaults():
    g = GuardrailSpec(type="blocklist")
    assert (g.action, g.on_input, g.on_output) == ("block", True, False)
