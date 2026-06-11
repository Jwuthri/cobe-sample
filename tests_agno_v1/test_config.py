"""The declarative AgentConfig contract + HTTP-tool compilation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_agno_v1.core.config import AgentConfig, HttpToolSpec, to_config
from agent_agno_v1.core.tools import compile_http_tool
from agent_agno_v1.examples import EXAMPLE_AGENT_CONFIG


def test_example_config_validates():
    cfg = to_config(EXAMPLE_AGENT_CONFIG)
    assert cfg.name == "Concierge"
    assert [t.name for t in cfg.tools] == ["search_products", "lookup_account"]
    assert cfg.tool_call_limit == 8


def test_extra_key_forbidden():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"name": "x", "system_prompt": "p", "bogus": 1})


def test_duplicate_tool_names_rejected():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(
            {
                "name": "x",
                "system_prompt": "p",
                "tools": [
                    {"kind": "registry", "name": "dup"},
                    {"kind": "registry", "name": "dup"},
                ],
            }
        )


def test_http_tool_requires_declared_placeholders():
    with pytest.raises(ValidationError):
        HttpToolSpec(
            name="t",
            url="https://api.example.com/{missing}",
            parameters={"type": "object", "properties": {}},
        )


def test_output_format_must_be_object_schema():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(
            {"name": "x", "system_prompt": "p", "output_format": {"type": "string"}}
        )


def test_http_tool_compiles_with_schema():
    spec = HttpToolSpec(
        name="lookup",
        description="look up",
        method="GET",
        url="https://api.example.com/{id}",
        headers={"Authorization": "Bearer {token}"},
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string"}, "token": {"type": "string"}},
            "required": ["id", "token"],
        },
    )
    fn = compile_http_tool(spec)
    assert fn.name == "lookup"
    assert set(fn.parameters["properties"]) == {"id", "token"}
