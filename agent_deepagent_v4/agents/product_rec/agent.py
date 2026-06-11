"""Declarative ``SubAgent`` spec for the product agent.

This is the JSON-object subagent shape the deepagents framework consumes:
``{name, description, system_prompt, tools, model, skills}``. The orchestrator
calls it via ``task(subagent_type="product-agent", ...)``.
"""

from __future__ import annotations

from deepagents import SubAgent

from agent_deepagent_v4.agents.product_rec.prompt import PRODUCT_REC_PROMPT
from agent_deepagent_v4.agents.product_rec.tools import PRODUCT_REC_TOOLS
from agent_deepagent_v4.config import main_model


def build_product_rec_subagent() -> SubAgent:
    return {
        "name": "product-agent",
        "description": (
            "Browse + cart contents. Use for: catalog search, product lookups, "
            "serviceability questions ('do you ship to 94110?'), and ALL cart "
            "content edits — add an item, remove an item, change a quantity, or "
            "'what's in my cart'. Adding an item naturally leads to checkout."
        ),
        "system_prompt": PRODUCT_REC_PROMPT,
        "tools": PRODUCT_REC_TOOLS,
        "model": main_model(),
        # Progressive-disclosure knowledge for resolving references + handoffs.
        "skills": ["shopping"],
    }
