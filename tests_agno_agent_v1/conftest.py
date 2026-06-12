"""Shared fakes — let the deterministic layer be tested with NO real LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agno.run.agent import RunContentEvent

from agno_agent_v1.agent.context import ShoppingContext, StepResult


@dataclass
class FakeToolCall:
    """Stand-in for an Agno ``ToolExecution`` (tool_name / tool_args / result)."""

    tool_name: str
    result: str = ""
    tool_args: dict = field(default_factory=dict)


class FakeMetrics:
    input_tokens = 11
    output_tokens = 7


@dataclass
class FakeRunOutput:
    """Stand-in for an Agno ``RunOutput`` (only the fields the wrapper reads)."""

    tools: list[FakeToolCall] = field(default_factory=list)
    content: str = "DONE"
    metrics: FakeMetrics = field(default_factory=FakeMetrics)


class FakeOrchestrator:
    """A no-LLM orchestrator: on ``run`` it pushes a canned step + events to ctx.

    Simulates "product_rec added P-1": appends a router/step event and a
    ``StepResult`` to the shared context, and mutates the live cart — exactly what
    the real sub-agent wrapper would do, so the session pipeline can be exercised.
    """

    def __init__(self, product_id: str = "P-1") -> None:
        self.product_id = product_id

    def run(self, _input: Any, *, dependencies: dict) -> FakeRunOutput:
        ctx: ShoppingContext = dependencies["ctx"]
        ctx.cart_service.add_item(self.product_id)
        ctx.events.append({"type": "router", "target": "product_rec", "iteration": 0})
        ctx.events.append({"type": "tool_start", "name": "add_item", "args": {"product_id": self.product_id}})
        ctx.events.append({"type": "tool_end", "name": "add_item", "result": "ok"})
        sr = StepResult(
            sop="product_rec",
            summary=f"added {self.product_id} to cart",
            next_sop="checkout",
            details={"added": [self.product_id]},
        )
        ctx.step_results.append(sr)
        ctx.events.append(
            {"type": "step", "sop": sr.sop, "summary": sr.summary, "asks": [], "next_sop": sr.next_sop, "details": sr.details}
        )
        return FakeRunOutput()


class FakeWriter:
    """A no-LLM writer: ``arun(stream=True)`` yields a few RunContentEvent deltas."""

    def __init__(self, deltas: list[str] | None = None) -> None:
        self.deltas = deltas or ["Added ", "P-1 ", "to your cart."]

    async def arun(self, _input: Any, *, stream: bool = True):
        for d in self.deltas:
            yield RunContentEvent(content=d)
