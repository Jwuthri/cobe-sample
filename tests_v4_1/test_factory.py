"""build_agent: config → create_agent mapping (no real model is built)."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent_v4_1.core import build_agent, register_tool
from agent_v4_1.core.guardrails import BlocklistGuardrail, LLMGuardrail
from agent_v4_1.core.middleware import LogToolCallsMiddleware
from agent_v4_1.core.skills import SkillsMiddleware
from agent_v4_1.examples import EXAMPLE_AGENT_CONFIG
from langchain.agents.middleware import ModelCallLimitMiddleware, PIIMiddleware


def _stub_example_tools():
    @tool
    def check_order_status(order_id: str) -> str:
        """stub"""
        return "ok"

    @tool
    def create_support_ticket(subject: str) -> str:
        """stub"""
        return "ok"

    register_tool(check_order_status, replace=True)
    register_tool(create_support_ticket, replace=True)


def test_build_example_config_maps_model_and_middleware():
    _stub_example_tools()
    captured = {}

    fake = GenericFakeChatModel(messages=iter([AIMessage(content="x")]))

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "AGENT"

    with patch("agent_v4_1.core.models.init_chat_model", return_value=fake) as init_mock, patch(
        "agent_v4_1.core.factory.create_agent", side_effect=fake_create_agent
    ):
        result = build_agent(EXAMPLE_AGENT_CONFIG)

    assert result == "AGENT"
    # model resolved with the exact provider:model + temperature
    init_mock.assert_called_once_with("openai:gpt-5-mini", temperature=0.0)

    # instructions appended under a heading
    assert "## Additional instructions" in captured["system_prompt"]
    assert "- Be concise and empathetic." in captured["system_prompt"]

    # output_format passed through as response_format (raw dict)
    assert captured["response_format"]["type"] == "object"
    assert captured["response_format"]["required"] == ["summary", "status"]

    # middleware order: [skills, *guardrails, *middleware]
    mw = captured["middleware"]
    types = [type(m) for m in mw]
    assert types == [
        SkillsMiddleware,
        PIIMiddleware,
        BlocklistGuardrail,
        LLMGuardrail,
        type(mw[4]),  # _ModelCallCounter (private)
        ModelCallLimitMiddleware,
        LogToolCallsMiddleware,
    ]
    assert mw[4].__class__.__name__ == "_ModelCallCounter"


def test_no_skills_means_no_skills_middleware():
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "AGENT"

    cfg = {"name": "bare", "system_prompt": "hi"}
    with patch(
        "agent_v4_1.core.models.init_chat_model",
        return_value=GenericFakeChatModel(messages=iter([AIMessage(content="x")])),
    ), patch("agent_v4_1.core.factory.create_agent", side_effect=fake_create_agent):
        build_agent(cfg)

    assert all(not isinstance(m, SkillsMiddleware) for m in captured["middleware"])
    assert captured["name"] == "bare"


def test_build_accepts_agentconfig_object_and_raw_dict():
    from agent_v4_1.core.config import AgentConfig

    with patch(
        "agent_v4_1.core.models.init_chat_model",
        return_value=GenericFakeChatModel(messages=iter([AIMessage(content="x")])),
    ), patch("agent_v4_1.core.factory.create_agent", return_value="A"):
        assert build_agent({"name": "a", "system_prompt": "p"}) == "A"
        assert build_agent(AgentConfig(name="a", system_prompt="p")) == "A"
