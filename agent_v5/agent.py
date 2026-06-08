"""``ShoppingAgentV5`` — the per-session orchestrator that runs one turn.

Holds the long-lived session state (the clean user/assistant transcript + the
live cart) and drives the supervisor for one user message. Both variants share
this orchestrator; the only fork is at the end:

  * ``speaking``: the reply is the supervisor's own final message.
  * ``router``:   the supervisor emits ``DONE`` and :func:`agent_v5.writer.compose_reply`
    produces the prose (one extra LLM call).

Blocks are assembled deterministically from the accumulated step results + cart
in BOTH cases (:func:`agent_v5.blocks.build_blocks`), so ids/prices are verbatim.

Each :class:`TurnResult` carries the metrics the eval needs: wall-clock latency,
a per-component token/LLM-call breakdown, which subagents ran, and a cart
snapshot for accuracy/hallucination checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent_v4.checkout.service import CartService
from agent_v4.step_result import StepResult
from agent_v5.blocks import build_blocks
from agent_v5.context import SupervisorContext, add_message_usage
from agent_v5.supervisor import Variant, build_supervisor
from agent_v5.writer import compose_reply
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return ""


def _merge(into: dict[str, int], other: dict[str, int]) -> None:
    for k, v in other.items():
        into[k] = into.get(k, 0) + int(v)


@dataclass
class TurnResult:
    user_message: str
    message: str
    blocks: list[dict]
    sops: list[str]
    step_results: list[StepResult]
    latency_s: float
    usage_total: dict[str, int]
    usage_breakdown: dict[str, dict[str, int]]
    supervisor_raw: str
    cart: dict
    capped: bool = False


@dataclass
class ShoppingAgentV5:
    variant: Variant
    user_id: str = "demo"
    session_id: str = "sess-v5"
    messages: list[BaseMessage] = field(default_factory=list)
    cart_service: CartService = field(default_factory=CartService)
    skills_loaded: list[str] = field(default_factory=list)
    _supervisor: object = None

    def __post_init__(self) -> None:
        self._supervisor = build_supervisor(self.variant)

    def _cart_snapshot(self) -> dict:
        c = self.cart_service.cart
        return {
            "step": c.step.value,
            "items": [{"id": i.product_id, "name": i.name, "qty": i.quantity} for i in c.items],
            "subtotal": str(c.subtotal),
            "grand_total": str(c.grand_total) if c.grand_total is not None else None,
            "confirmed": c.confirmed,
            "receipt_id": c.receipt_id,
        }

    def run_turn(self, user_text: str) -> TurnResult:
        self.messages.append(HumanMessage(content=user_text))
        ctx = SupervisorContext(
            user_id=self.user_id,
            session_id=self.session_id,
            cart_service=self.cart_service,
            skills_loaded=list(self.skills_loaded),
        )

        t0 = time.perf_counter()
        result = self._supervisor.invoke({"messages": self.messages}, context=ctx)
        sup_messages = result["messages"]

        supervisor_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "llm_calls": 0}
        add_message_usage(supervisor_usage, sup_messages)

        supervisor_raw = _last_ai_text(sup_messages)
        writer_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "llm_calls": 0}
        capped = False

        if self.variant == "speaking":
            message = supervisor_raw
            if not message:
                # Loop cap fired before a closing message — fall back to the
                # writer so the user still gets a reply (flagged for the eval).
                capped = True
                message, writer_usage = compose_reply(
                    self.messages, ctx.step_results, self.cart_service.cart
                )
        else:  # router → dedicated writer
            message, writer_usage = compose_reply(
                self.messages, ctx.step_results, self.cart_service.cart
            )

        latency = time.perf_counter() - t0

        blocks = build_blocks(ctx.step_results, self.cart_service.cart)
        self.skills_loaded = ctx.skills_loaded
        # Carry the typed blocks on the stored AIMessage (like v4's emit node) so
        # a serialized transcript — e.g. the webapp's chat history — can render
        # them on reload, not just on the live turn event.
        self.messages.append(
            AIMessage(content=message, additional_kwargs={"blocks": blocks} if blocks else {})
        )

        total = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "llm_calls": 0}
        _merge(total, supervisor_usage)
        _merge(total, ctx.subagent_usage)
        _merge(total, writer_usage)

        return TurnResult(
            user_message=user_text,
            message=message,
            blocks=blocks,
            sops=[sr.sop for sr in ctx.step_results],
            step_results=list(ctx.step_results),
            latency_s=latency,
            usage_total=total,
            usage_breakdown={
                "supervisor": supervisor_usage,
                "subagents": dict(ctx.subagent_usage),
                "writer": writer_usage,
            },
            supervisor_raw=supervisor_raw,
            cart=self._cart_snapshot(),
            capped=capped,
        )


__all__ = ["ShoppingAgentV5", "TurnResult"]
