"""Declarative ``SubAgent`` spec for the checkout agent.

Safe checkout lives inside the ``confirm_checkout`` tool: it refuses while the
cart has blockers, then calls ``interrupt()`` to pause for an explicit human
approval before charging (gated by the ``require_approval`` context flag). The
order is only placed on an approved resume. We deliberately do NOT also set
``interrupt_on={"confirm_checkout": True}`` — the framework's HITL middleware
fires *before* the tool runs and would pre-empt our richer approval payload
(order summary + total). One chokepoint, one payload.
"""

from __future__ import annotations

from deepagents import SubAgent

from agent_deepagent_v4.agents.checkout.prompt import CHECKOUT_PROMPT
from agent_deepagent_v4.agents.checkout.skills import CHECKOUT_SKILL_SOURCES
from agent_deepagent_v4.agents.checkout.tools import CHECKOUT_TOOLS
from agent_deepagent_v4.config import main_model


def build_checkout_subagent() -> SubAgent:
    return {
        "name": "checkout-agent",
        "description": (
            "Fulfillment flow for an in-progress purchase: capture identity, "
            "address, delivery option, payment, and place the order on explicit "
            "user approval. Route here when the user provides checkout data "
            "(name, address as part of buying, delivery, payment) or says yes to "
            "confirm. It cannot add or remove products."
        ),
        "system_prompt": CHECKOUT_PROMPT,
        "tools": CHECKOUT_TOOLS,
        "model": main_model(),
        "skills": CHECKOUT_SKILL_SOURCES,
    }
