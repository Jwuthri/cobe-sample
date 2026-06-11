"""Declarative ``SubAgent`` spec for the order-status agent."""

from __future__ import annotations

from deepagents import SubAgent

from agent_deepagent_v4.agents.order_status.prompt import ORDER_STATUS_PROMPT
from agent_deepagent_v4.agents.order_status.tools import ORDER_STATUS_TOOLS
from agent_deepagent_v4.config import main_model


def build_order_status_subagent() -> SubAgent:
    return {
        "name": "order-status-agent",
        "description": (
            "Look up a PAST order's status, tracking, or delivery. Route here "
            "when the user asks about an existing order (ids look like ORD-* or "
            "RCPT-*), e.g. 'where's my order ORD-7?'."
        ),
        "system_prompt": ORDER_STATUS_PROMPT,
        "tools": ORDER_STATUS_TOOLS,
        "model": main_model(),
    }
