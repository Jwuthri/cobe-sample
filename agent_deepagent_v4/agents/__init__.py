"""Subagent builders. Each subpackage holds one agent's prompt, tools, (skills,)
and the declarative ``SubAgent`` spec the orchestrator delegates to."""

from agent_deepagent_v4.agents.checkout.agent import build_checkout_subagent
from agent_deepagent_v4.agents.order_status.agent import build_order_status_subagent
from agent_deepagent_v4.agents.product_rec.agent import build_product_rec_subagent
from agent_deepagent_v4.agents.writer.agent import build_writer_subagent

__all__ = [
    "build_product_rec_subagent",
    "build_checkout_subagent",
    "build_order_status_subagent",
    "build_writer_subagent",
]
