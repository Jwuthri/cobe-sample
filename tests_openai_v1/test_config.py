"""The declarative config contract holds (identical to agent_v4_1's)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openai_agent_v1.core.config import AgentConfig, HttpToolSpec, ModelConfig
from openai_agent_v1.examples import EXAMPLE_AGENT_CONFIG


def test_example_config_validates_verbatim():
    cfg = AgentConfig.model_validate(EXAMPLE_AGENT_CONFIG)
    assert cfg.name == "Acme Support"
    assert cfg.model.provider_model == "openai:gpt-5-mini"
    assert {t.name for t in cfg.tools} == {
        "check_order_status",
        "create_support_ticket",
        "create_zendesk_ticket",
    }
    assert [s.name for s in cfg.skills] == ["checkout_flow"]
    assert {g.type for g in cfg.guardrails} == {"pii", "blocklist", "llm_judge"}


def test_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"name": "x", "system_prompt": "y", "bogus": 1})


def test_provider_model_requires_prefix():
    with pytest.raises(ValidationError):
        ModelConfig(provider_model="gpt-5-mini")  # missing "provider:"
    assert ModelConfig(provider_model="openai:gpt-5-mini").provider_model == "openai:gpt-5-mini"


def test_http_tool_undeclared_placeholder_rejected():
    with pytest.raises(ValidationError):
        HttpToolSpec(
            name="t",
            url="https://x/{secret}",
            parameters={"type": "object", "properties": {}},  # secret not declared
        )


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
