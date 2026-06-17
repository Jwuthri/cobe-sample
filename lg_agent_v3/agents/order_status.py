"""The ``order_status`` worker — look up a PAST order's status / tracking.

The simplest worker: two tools, no snapshot, a tiny extractor. It also contributes its
own ``recall`` snippet, proving the cross-turn memory mechanism is generic — a
different worker feeding the same orchestrator memory.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from lg_agent_v3.agents import tools
from lg_agent_v3.agents.names import ORDER_STATUS
from lg_agent_v3.runtime import (
    ShoppingDeps,
    StepResult,
    Worker,
    build_model,
    compile_guardrails,
    tool_returns,
)

PROMPT = """\
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


def build(model: Any | None = None, guardrails: list | None = None):
    """Compile the order_status worker (optionally with an injected model + guardrails)."""
    return create_agent(
        model=model or build_model(0.0),
        tools=[tools.get_order_status, tools.list_recent_orders],
        system_prompt=PROMPT,
        context_schema=ShoppingDeps,
        middleware=compile_guardrails(guardrails, ORDER_STATUS),
        name=ORDER_STATUS,
    )


agent = build()


def _parse_order(returns) -> dict | None:
    for r in returns:
        if r.name in ("get_order_status", "list_recent_orders") and r.content.strip():
            if "unknown order" not in r.content.lower():
                return {"raw": r.content.strip()}
    return None


def extract(deps: ShoppingDeps, messages, before) -> StepResult:
    order = _parse_order(tool_returns(messages))
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
    extract=extract,
    prompt=PROMPT,
    block="order_status",
)
