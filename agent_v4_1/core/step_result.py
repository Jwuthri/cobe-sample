"""A sub-agent's structured result — the bridge to blocks + the writer.

Each sub-agent tool distills its run into a ``StepResult`` and appends it to the
turn context. The orchestrator LLM only ever reads the terse ``summary``; the
rich ``details`` dict feeds the deterministic block builder and the writer (so
ids/prices are never invented by a model).
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
