"""Order-status subagent — an Agno ``Agent`` (was ``create_agent`` in v2).

Single-purpose: parse an order id (or accept "my latest"), call the tools,
summarize. No skills, no gating. Invoked per-turn by the workflow step with
``session_state`` + ``dependencies`` threaded through, so its tools see the
long-term store + user id via ``run_context``.
"""

from __future__ import annotations

from agno.agent import Agent

from agent_v3.models import chat_model
from agent_v3.tools.order_tools import get_order_status, list_recent_orders

ORDER_STATUS_PROMPT = """\
You help users check the status of their orders.

If the user mentions a specific order id (looks like ORD-* or RCPT-*),
call get_order_status. Otherwise call list_recent_orders to show what
they have, then ask which one they want details on.

Be concise. Report the order id, status, and tracking URL if any.
"""


def build_order_status_agent() -> Agent:
    return Agent(
        name="order_status",
        model=chat_model(),
        tools=[get_order_status, list_recent_orders],
        instructions=ORDER_STATUS_PROMPT,
        telemetry=False,
    )
