"""A member sub-agent's structured result — the bridge to blocks + grounding.

Each delegation is distilled (from the member's streamed tool calls + the live
cart) into a ``StepResult`` and appended to the turn record. The supervisor's
prose is grounded by the cart snapshot it sees; the rich ``details`` dict feeds
the deterministic block builder so ids/prices are never invented by a model.
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


__all__ = ["StepResult"]
