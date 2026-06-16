"""``StepResult`` — the distilled outcome of one worker (sub-agent) run.

A worker does its job via tool calls, then packs everything anyone downstream needs
into a ``StepResult``. It has two distinct audiences, and that split is the whole
point:

* the **orchestrator** reads ONLY the terse :pyattr:`summary` — so it can route the
  next step without ever seeing (and therefore without being able to hallucinate)
  raw ids/prices;
* the **writer + block builder** read the rich :pyattr:`details` — the grounded
  facts that become the user-facing text and the typed cards.

``sop`` is the name of the worker that produced the result (its "standard operating
procedure"). It is the shared vocabulary used for routing, block selection, and the
frontend ``step`` event.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StepResult(BaseModel):
    sop: str  # which worker ran (e.g. "product_rec")
    summary: str = ""  # terse line — the ONLY thing the orchestrator reads
    asks: list[str] = Field(default_factory=list)  # what the user still needs to provide
    next_sop: str | None = None  # a hint the orchestrator may follow next
    details: dict | None = None  # grounded facts → deterministic blocks + writer
    # Domain-rendered free text the orchestrator should remember NEXT turn to resolve
    # references ("the green one"). Private to the orchestrator — never shown the writer.
    recall: str | None = None
