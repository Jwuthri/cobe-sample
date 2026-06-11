"""Declarative ``SubAgent`` spec for the writer agent.

The writer is the single voice that produces the customer-facing text. The
orchestrator always finishes a turn by delegating to it and returns its message
verbatim, so all other subagents can stay terse and factual.
"""

from __future__ import annotations

from deepagents import SubAgent

from agent_deepagent_v4.agents.writer.prompt import WRITER_PROMPT
from agent_deepagent_v4.agents.writer.tools import WRITER_TOOLS
from agent_deepagent_v4.config import writer_model


def build_writer_subagent() -> SubAgent:
    return {
        "name": "writer-agent",
        "description": (
            "Composes the final customer-facing message. ALWAYS delegate the last "
            "step of every turn here, passing the customer's request and the facts "
            "gathered this turn. Return its message to the customer verbatim."
        ),
        "system_prompt": WRITER_PROMPT,
        "tools": WRITER_TOOLS,
        "model": writer_model(),
    }
