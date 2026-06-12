"""The ``order_status`` sub-agent — look up a past order's status / tracking.

The simplest sub-agent: no input builder (a plain query is enough) and no snapshot.
It contributes its own ``recall`` snippet too, proving the recall mechanism is
generic — a different agent type feeding the same orchestrator memory.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from lg_agent.core.step import StepResult
from lg_agent.core.subagent import SubAgent
from lg_agent.shopping.agents.subagents.names import ORDER_STATUS
from lg_agent.shopping.tools import ORDER_STATUS_TOOLS, registry_specs

MODEL = "openai:gpt-5.4-mini"

PROMPT = """\
You help users check the status of their orders.

If the user mentions a specific order id (looks like ORD-* or RCPT-*),
call get_order_status. Otherwise call list_recent_orders to show what
they have, then ask which one they want details on.

Be concise. Report the order id, status, and tracking URL if any.
"""

DESCRIPTION = (
    "Look up a PAST order's status / tracking (order ids look like ORD-* or "
    "RCPT-*). Pass the user's order question as `query`."
)

CONFIG = {
    "name": ORDER_STATUS,
    "description": "Look up a past order's status / tracking.",
    "system_prompt": PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    "tools": registry_specs(ORDER_STATUS_TOOLS),
    "middleware": [{"name": "log_tool_calls", "params": {"log_prefix": ORDER_STATUS}}],
}


def _parse_order(messages) -> dict | None:
    for m in messages:
        if not isinstance(m, ToolMessage) or getattr(m, "name", None) not in (
            "get_order_status",
            "list_recent_orders",
        ):
            continue
        content = str(m.content).strip()
        if content and "unknown order" not in content.lower():
            return {"raw": content}
    return None


def extract(ctx, messages, before) -> StepResult:
    order = _parse_order(messages)
    return StepResult(
        sop=ORDER_STATUS,
        summary="looked up order status" if order else "could not find a matching order",
        asks=[] if order else ["confirm the order id"],
        next_sop=None,
        details=order,
        recall=f"Recently looked up order: {order['raw']}" if order else None,
    )


SUBAGENT = SubAgent(
    name=ORDER_STATUS,
    description=DESCRIPTION,
    config=CONFIG,
    extract=extract,
    block="order_status",
)

__all__ = ["SUBAGENT", "CONFIG", "PROMPT"]
