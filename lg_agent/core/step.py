"""``StepResult`` — the structured outcome of one sub-agent run.

Each sub-agent distills its run into a ``StepResult`` and appends it to the turn
context. Two audiences read it:

  * the orchestrator LLM reads ONLY the terse ``summary`` (so it can't hallucinate
    ids/prices it never saw);
  * the deterministic block builder + the writer read the rich ``details`` (the
    grounded facts that become the user-facing cards).

``sop`` is the name of the sub-agent that produced the result (a.k.a. the step's
"SOP"). It is the shared vocabulary used for routing, block selection, and the
frontend ``step`` event.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StepResult(BaseModel):
    sop: str  # which sub-agent ran (e.g. "product_rec")
    summary: str = ""  # terse line — the ONLY thing the orchestrator reads
    asks: list[str] = Field(default_factory=list)  # what the user still needs to provide
    next_sop: str | None = None  # a hint the orchestrator can follow next
    cart_diff: dict | None = None
    details: dict | None = None  # grounded facts → deterministic blocks + writer
    recall: str | None = None


__all__ = ["StepResult"]
