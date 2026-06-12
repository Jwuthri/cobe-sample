"""build_agent compiles a config into a real SDK Agent (no model call)."""

from __future__ import annotations

import pytest
from agents import Agent, function_tool

from openai_agent_v1.core.factory import agent_max_turns, build_agent
from openai_agent_v1.core.registry import TOOLS
from openai_agent_v1.examples import EXAMPLE_AGENT_CONFIG


@pytest.fixture
def _stub_example_tools():
    """Register the EXAMPLE's two registry tools so it can actually build."""

    @function_tool(name_override="check_order_status")
    def check_order_status(order_id: str) -> str:
        "Look up an order."
        return "ok"

    @function_tool(name_override="create_support_ticket")
    def create_support_ticket(subject: str) -> str:
        "Open a ticket."
        return "ok"

    for t in (check_order_status, create_support_ticket):
        TOOLS.register(t.name, t, replace=True)
    yield


def test_build_minimal_agent():
    agent = build_agent(
        {"name": "Solo", "system_prompt": "You are helpful.", "model": {"provider_model": "openai:gpt-5.4-mini"}}
    )
    assert isinstance(agent, Agent)
    assert agent.model == "gpt-5.4-mini"  # openai: prefix stripped
    assert agent.tools == []


def test_build_example_with_stubbed_tools(_stub_example_tools):
    agent = build_agent(EXAMPLE_AGENT_CONFIG)
    names = {t.name for t in agent.tools}
    # 2 registry + 1 http + load_skill (from the checkout_flow skill)
    assert {"check_order_status", "create_support_ticket", "create_zendesk_ticket"} <= names
    assert "load_skill" in names
    assert agent.model == "gpt-5-mini"
    assert agent.output_type is not None  # output_format → output_type adapter
    assert agent_max_turns(agent) == 30  # from the max_turns middleware


def test_instructions_compose_with_bullets():
    agent = build_agent(
        {
            "name": "B",
            "system_prompt": "Base prompt.",
            "instructions": ["Be brief.", "Be kind."],
        }
    )
    # No skills/cart_anchor → instructions is a static string with the bullets.
    assert isinstance(agent.instructions, str)
    assert "## Additional instructions" in agent.instructions
    assert "- Be brief." in agent.instructions


def test_orchestrator_hides_checkout_when_cart_empty():
    """empty_cart_guard composes into the checkout delegate's is_enabled."""
    from openai_agent_v1.shopping.platform import build_orchestrator

    orch = build_orchestrator()
    checkout = next(t for t in orch.tools if t.name == "checkout")
    assert callable(checkout.is_enabled)  # gated, not a bare True
