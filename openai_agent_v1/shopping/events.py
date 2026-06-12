"""SSE event-vocabulary helpers for the shopping turn.

Unlike agent_v4_1 (which classified a langgraph custom stream), this port's
sub-agent tools emit already-classified UI events directly onto the turn's event
bus (see :mod:`openai_agent_v1.core.subagent`). These helpers remain for parity /
tests: the canonical ``step`` shape and the sub-agent name set.
"""

from __future__ import annotations

from openai_agent_v1.core.step_result import StepResult

# The sub-agent tool names — a routing event targets one of these.
SUBAGENT_NAMES = {"product_rec", "checkout", "order_status"}


def step_event(sr: StepResult) -> dict:
    return {
        "type": "step",
        "sop": sr.sop,
        "summary": sr.summary,
        "asks": list(sr.asks),
        "next_sop": sr.next_sop,
        "details": sr.details,
    }


__all__ = ["SUBAGENT_NAMES", "step_event"]
