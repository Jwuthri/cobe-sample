"""Structured result a sub-agent (leaf) returns to the supervisor.

Each leaf wrapper builds a ``StepResult`` from the inner subagent's
tool calls + final state. The supervisor reads them to decide:
  - what to route to next (``next_sop`` hint or re-classification)
  - whether the turn is done (no more outstanding ``asks`` from
    blocking SOPs)

The writer agent consumes the accumulated list to produce the
single user-facing message.

``sop`` / ``next_sop`` are leaf ids (plain strings — see
:mod:`agent_v4.ids`) rather than a hard-coded enum, so the leaf set
stays data-driven.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StepResult(BaseModel):
    sop: str
    summary: str = ""
    asks: list[str] = Field(default_factory=list)
    next_sop: str | None = None
    cart_diff: dict | None = None
    details: dict | None = None
