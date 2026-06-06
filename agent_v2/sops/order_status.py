"""Order-status subagent.

Single-purpose: parse an order id (or accept "my latest"), call the
tools, summarize. No skills, no HITL — just create_agent.
"""

from __future__ import annotations

from agent_v2.llm import chat_model
from agent_v2.middleware import log_tool_calls
from agent_v2.runtime import RuntimeContext
from agent_v2.tools.order_tools import get_order_status, list_recent_orders
from langchain.agents import create_agent

ORDER_STATUS_PROMPT = """\
You help users check the status of their orders.

If the user mentions a specific order id (looks like ORD-* or RCPT-*),
call get_order_status. Otherwise call list_recent_orders to show what
they have, then ask which one they want details on.

Be concise. Report the order id, status, and tracking URL if any.
"""


def build_order_status_agent(store=None):
    return create_agent(
        model=chat_model(),
        tools=[get_order_status, list_recent_orders],
        system_prompt=ORDER_STATUS_PROMPT,
        middleware=[log_tool_calls],
        context_schema=RuntimeContext,
        store=store,
    )
