"""Structured result a sub-agent returns to the supervisor.

Each ``*_wrapper`` builds a ``StepResult`` from the inner subagent's
tool calls + final state. The supervisor reads them to decide:
  - what to route to next (``next_sop`` hint or re-classification)
  - whether the turn is done (no more outstanding ``asks`` from
    blocking SOPs)

The writer agent consumes the accumulated list to produce the
single user-facing message.
"""

from __future__ import annotations

from agent_v2.supervisor import SOPName
from pydantic import BaseModel, Field


class StepResult(BaseModel):
    sop: SOPName
    summary: str = ""
    asks: list[str] = Field(default_factory=list)
    next_sop: SOPName | None = None
    cart_diff: dict | None = None
    # Structured data the writer should render verbatim — e.g.
    # ``{"products": [{"id": "P-1", "name": "...", "price": "..."}]}``
    # or ``{"order": {"id": "ORD-7", "status": "shipped"}}``.
    details: dict | None = None
