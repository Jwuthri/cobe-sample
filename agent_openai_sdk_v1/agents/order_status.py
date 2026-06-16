"""The ``order_status`` worker — look up a PAST order's status / tracking.

The simplest worker: two tools, a tiny extractor. It also contributes its own
``recall`` snippet, proving the cross-turn memory mechanism is generic — a different
worker feeding the same orchestrator memory.
"""

from __future__ import annotations

from typing import Any

from agents import Agent

from agent_openai_sdk_v1.agents import tools
from agent_openai_sdk_v1.agents.names import ORDER_STATUS
from agent_openai_sdk_v1.runtime import MODEL_NAME, ShoppingContext, StepResult, Worker, settings, tool_returns

_BASE_PROMPT = """\
You help users check the status of their orders.

If the user mentions a specific order id (looks like ORD-* or RCPT-*), call
get_order_status. Otherwise call list_recent_orders to show what they have, then
ask which one they want details on.

Be concise. Report the order id, status, and tracking URL if any.

You don't write the user-facing reply — a separate writer does. After calling your
tool, reply with the single word DONE (an internal marker the user never sees).
"""

DESCRIPTION = (
    "Look up a PAST order's status / tracking (order ids look like ORD-* or RCPT-*). "
    "Pass the user's order question as `query`."
)

agent = Agent[ShoppingContext](
    name=ORDER_STATUS,
    model=MODEL_NAME,
    model_settings=settings(0.0),
    instructions=_BASE_PROMPT,
    tools=[tools.get_order_status, tools.list_recent_orders],
)


def _parse_order(returns) -> dict | None:
    for r in returns:
        if r.name in ("get_order_status", "list_recent_orders") and r.content.strip():
            if "unknown order" not in r.content.lower():
                return {"raw": r.content.strip()}
    return None


def extract(ctx: ShoppingContext, items: list[Any]) -> StepResult:
    order = _parse_order(tool_returns(items))
    return StepResult(
        sop=ORDER_STATUS,
        summary="looked up order status" if order else "could not find a matching order",
        asks=[] if order else ["confirm the order id"],
        details=order,
        recall=f"Recently looked up order: {order['raw']}" if order else None,
    )


WORKER = Worker(
    name=ORDER_STATUS,
    agent=agent,
    description=DESCRIPTION,
    extract=extract,
    prompt=_BASE_PROMPT,
    block="order_status",
)
