"""The ``order_status`` worker — look up a PAST order's status / tracking.

The simplest worker: two tools, no snapshot, a tiny extractor. It also contributes
its own ``recall`` snippet, proving the cross-turn memory mechanism is generic — a
different worker feeding the same orchestrator memory.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from google_adk_agent_v1.agents import tools
from google_adk_agent_v1.agents.names import ORDER_STATUS
from google_adk_agent_v1.runtime import ShoppingDeps, StepResult, Worker, gen_config, make_model, tool_returns

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

agent = LlmAgent(
    name=ORDER_STATUS,
    model=make_model(),
    generate_content_config=gen_config(0.0),
    instruction=PROMPT,
    tools=[tools.get_order_status, tools.list_recent_orders],
)


def _parse_order(returns) -> dict | None:
    for r in returns:
        if r.name in ("get_order_status", "list_recent_orders") and r.content.strip():
            if "unknown order" not in r.content.lower():
                return {"raw": r.content.strip()}
    return None


def extract(deps: ShoppingDeps, run_events, before) -> StepResult:
    order = _parse_order(tool_returns(run_events))
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
