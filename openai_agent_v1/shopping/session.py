"""``ShoppingSession`` — the per-session, streaming-first turn engine.

This is where the streaming story lands. A turn is:

  1. input guardrails (pre-flight, before any model call) — a refusal is instant,
     a redaction rewrites the user text before it enters the transcript;
  2. the orchestrator routes to sub-agent tools, run in a background task; its
     sub-agent tools push live UI events (router / tool / step / trace) onto the
     turn's **event bus**, which this loop drains as they happen;
  3. the writer — the LAST model call, with nothing after it — streams its tokens
     straight to the client (``{type:"token"}``), retried once if it emits nothing;
  4. deterministic blocks are attached and a final ``{type:"bot"}`` carries the
     authoritative text + blocks.

Because the writer is terminal and only ever sees verified step results + cart
(blocks are built deterministically), there is no post-generation validator to
gate the stream — the grounding happened at construction. See the README.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, AsyncGenerator

from agents import RunConfig, Runner
from openai.types.responses import ResponseTextDeltaEvent

from openai_agent_v1.core.factory import agent_max_turns
from openai_agent_v1.core.guardrails import CompiledInputRule, run_input_guardrails
from openai_agent_v1.core.messages import Msg, ai, human, msgs_to_input, system
from openai_agent_v1.core.trace import render_messages, trace_event
from openai_agent_v1.shopping.agents import BLOCK_BY_SOP, ORCHESTRATOR_AGENT, SUBAGENTS
from openai_agent_v1.shopping.blocks import build_blocks
from openai_agent_v1.shopping.context import ShoppingContext
from openai_agent_v1.shopping.domain import CartService
from openai_agent_v1.shopping.platform import build_orchestrator, build_writer, store
from openai_agent_v1.shopping.prompts import WRITER_SYSTEM
from openai_agent_v1.shopping.writer_payload import build_writer_payload

# How many transcript turns to show in the orchestrator_input trace.
_TRACE_HISTORY = 24

_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"

# Sentinel placed on the bus when the orchestrator run completes.
_DONE = object()

# Tracing to the OpenAI dashboard is disabled (no project wiring for the demo).
_NO_TRACE = RunConfig(tracing_disabled=True)


def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


@dataclass
class ShoppingSession:
    """One conversation: clean transcript + live cart + the compiled agents."""

    user_id: str = "demo"
    session_id: str = "sess-oa1"
    messages: list[Msg] = field(default_factory=list)
    cart_service: CartService = field(default_factory=CartService)
    skills_loaded: list[str] = field(default_factory=list)
    turn: int = 0  # incremented each run_turn_stream — delimits turns in the UI
    # Persisted per-step ``recall`` snippets (keyed by sop), carried to the
    # orchestrator so it can resolve the user's references WITHOUT the sub-agents
    # seeing the chat. Opaque to the session — domain code renders the content.
    routing_notes: dict[str, str] = field(default_factory=dict)

    # Injected for tests; built from the platform otherwise.
    orchestrator: Any = None
    writer: Any = None
    input_rules: list[CompiledInputRule] = field(default_factory=list)
    # If the writer has output-side guardrails, that turn buffers (no token stream).
    writer_buffered: bool = False
    # Emit deep-trace events (exact payloads between orchestrator/sub-agents/writer).
    debug: bool = True

    def __post_init__(self) -> None:
        if self.orchestrator is None:
            self.orchestrator = build_orchestrator()
        if self.writer is None:
            self.writer = build_writer()

    # ----- snapshot (the frontend's AgentSnapshot) -----
    def snapshot(self) -> dict[str, Any]:
        cart = self.cart_service.cart
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "active_sop": None,
            "skills_loaded": list(self.skills_loaded),
            "cart": {
                "step": cart.step.value,
                "cart_id": cart.cart_id,
                "items": [
                    {
                        "id": i.product_id,
                        "name": i.name,
                        "qty": i.quantity,
                        "unit_price": _d(i.unit_price),
                        "line_total": _d(i.line_total),
                        "tags": list(i.tags),
                    }
                    for i in cart.items
                ],
                "customer": cart.customer.model_dump(),
                "address": cart.address.model_dump(),
                "serviceable": cart.serviceable,
                "serviceable_options": list(cart.serviceable_options),
                "delivery_option": cart.delivery_option,
                "shipping": (
                    {"cost": _d(cart.shipping.cost), "eta_hours": cart.shipping.eta_hours}
                    if cart.shipping_is_fresh()
                    else None
                ),
                "tax": (
                    {"amount": _d(cart.tax.amount), "rate": _d(cart.tax.rate)}
                    if cart.tax_is_fresh()
                    else None
                ),
                "promo": (
                    {"code": cart.promo.code, "discount": _d(cart.promo.discount)}
                    if cart.promo
                    else None
                ),
                "payment_method": cart.payment_method,
                "card_token_set": bool(cart.card_token),
                "subtotal": _d(cart.subtotal),
                "grand_total": _d(cart.grand_total) if cart.grand_total is not None else None,
                "blockers": [{"code": b.code, "message": b.message} for b in cart.blockers()],
                "ready_to_confirm": cart.ready_to_confirm(),
                "confirmed": cart.confirmed,
                "receipt_id": cart.receipt_id,
            },
            "messages": [
                {"role": m.role, "content": str(m.content), "blocks": list(m.blocks)}
                for m in self.messages
            ],
            "iteration": 0,
            "done": True,
        }

    # ----- routing context (the orchestrator owns reference resolution) -----
    def _routing_memo(self, ctx: ShoppingContext) -> str | None:
        """Assemble the orchestrator's reference-resolution block (domain-agnostic)."""
        live = ctx.routing_context()
        blocks = [text for text in live.values() if text]
        blocks += [text for key, text in self.routing_notes.items() if key not in live and text]
        if not blocks:
            return None
        return (
            "Context for resolving the user's references (authoritative — do NOT "
            "invent; if a reference is ambiguous, delegate to a sub-agent to look "
            "it up):\n" + "\n".join(blocks)
        )

    def _absorb_recalls(self, ctx: ShoppingContext) -> None:
        """Persist each step's ``recall`` snippet (keyed by sop) for future turns."""
        for sr in ctx.step_results:
            if sr.recall:
                self.routing_notes[sr.sop] = sr.recall

    def _new_context(self, bus: asyncio.Queue) -> ShoppingContext:
        return ShoppingContext(
            user_id=self.user_id,
            session_id=self.session_id,
            cart_service=self.cart_service,
            skills_loaded=list(self.skills_loaded),
            debug=self.debug,
            store=store(),
            bus=bus,
        )

    # ----- the streaming turn -----
    async def run_turn_stream(self, user_text: str) -> AsyncGenerator[dict, None]:
        self.turn += 1
        yield {"type": "user", "content": user_text, "turn": self.turn}
        yield {"type": "state", "snapshot": self.snapshot()}

        # 1. input guardrails (pre-flight, before any model call)
        outcome = run_input_guardrails(self.input_rules, user_text)
        for hit in outcome.triggered:
            yield {"type": "guardrail", "stage": "input", "rule": hit.type, "action": hit.action}
        if not outcome.allowed:
            self.messages.append(human(user_text))
            self.messages.append(ai(outcome.refusal or ""))
            yield {"type": "bot", "content": outcome.refusal or "", "blocks": []}
            yield {"type": "state", "snapshot": self.snapshot()}
            yield {"type": "end"}
            return

        self.messages.append(human(outcome.text))

        bus: asyncio.Queue = asyncio.Queue()
        ctx = self._new_context(bus)

        # The orchestrator is the SOLE reader of the conversation. It also gets a
        # deterministic routing memo (cart + recently shown products) so it can
        # resolve references into concrete ids before delegating — the sub-agents
        # never see the transcript (see core/subagent.py "Context isolation").
        memo = self._routing_memo(ctx)
        orch_input = msgs_to_input(self.messages)
        if memo and orch_input:
            orch_input.insert(len(orch_input) - 1, {"role": "system", "content": memo})

        # trace: exactly what the orchestrator sees (transcript + routing memo)
        if self.debug:
            trace_msgs: list[Msg] = list(self.messages)
            if memo:
                trace_msgs.insert(len(trace_msgs) - 1, system(memo))
            yield trace_event(
                "orchestrator_input",
                "orchestrator",
                "user turn → orchestrator",
                {
                    "system_prompt": ORCHESTRATOR_AGENT.get("system_prompt", ""),
                    "delegates": [s.name for s in SUBAGENTS],
                    "routing_memo": memo,
                    "messages_total": len(self.messages),
                    "conversation_seen": render_messages(trace_msgs[-_TRACE_HISTORY:]),
                    "context": ctx.debug_view(),
                },
            )

        # 2. orchestrator phase — driven in a background task; sub-agent tools push
        #    live events onto the bus, which we drain in order until DONE.
        async def _drive() -> None:
            try:
                await Runner.run(
                    self.orchestrator,
                    orch_input,
                    context=ctx,
                    max_turns=agent_max_turns(self.orchestrator),
                    run_config=_NO_TRACE,
                )
            except Exception as e:  # noqa: BLE001
                bus.put_nowait({"type": "error", "content": str(e)})
            finally:
                bus.put_nowait(_DONE)

        task = asyncio.create_task(_drive())
        errored = False
        while True:
            ev = await bus.get()
            if ev is _DONE:
                break
            if ev.get("type") == "error":
                errored = True
            yield ev
        await task
        if errored:
            yield {"type": "end"}
            return

        # persist this turn's recall snippets so the NEXT turn's orchestrator can
        # resolve references without the sub-agents ever seeing the chat
        self._absorb_recalls(ctx)

        yield {"type": "state", "snapshot": self.snapshot()}

        # 3. writer phase — the terminal model call, streamed token-by-token
        yield {"type": "router", "target": "writer", "iteration": 0}
        payload_json, _mode = build_writer_payload(
            self.messages, ctx.step_results, self.cart_service.cart
        )

        # trace: the exact grounded payload the writer composes its reply from
        if self.debug:
            try:
                parsed_payload = json.loads(payload_json)
            except json.JSONDecodeError:
                parsed_payload = {"raw": payload_json}
            yield trace_event(
                "writer_payload",
                "writer",
                f"writer input (mode={_mode})",
                {"system_prompt": WRITER_SYSTEM, "mode": _mode, "payload": parsed_payload},
            )

        text = ""
        try:
            if self.writer_buffered:
                text = await self._invoke_writer(payload_json)
            else:
                async for ev in self._stream_writer(payload_json):
                    if ev["type"] == "token":
                        yield ev
                    elif ev["type"] == "_final":
                        text = ev["content"]
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "content": str(e)}
            yield {"type": "end"}
            return

        if not text:
            text = _EMPTY_WRITER_FALLBACK

        blocks = build_blocks(ctx.step_results, self.cart_service.cart, BLOCK_BY_SOP)
        self.skills_loaded = ctx.skills_loaded
        self.messages.append(ai(text, blocks=blocks))

        yield {"type": "writer", "draft": text, "blocks": blocks}
        yield {"type": "bot", "content": text, "blocks": blocks}
        yield {"type": "state", "snapshot": self.snapshot()}
        yield {"type": "end"}

    async def _stream_writer(self, payload_json: str) -> AsyncGenerator[dict, None]:
        """Stream writer tokens, retrying once if the first attempt is empty.

        Empty-retry is stream-safe: an empty stream sent zero tokens, so the
        retry is invisible to the client.
        """
        for _attempt in (1, 2):
            parts: list[str] = []
            result = Runner.run_streamed(
                self.writer,
                payload_json,
                max_turns=agent_max_turns(self.writer),
                run_config=_NO_TRACE,
            )
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    delta = event.data.delta or ""
                    if delta:
                        parts.append(delta)
                        yield {"type": "token", "content": delta}
            text = "".join(parts).strip()
            if text:
                yield {"type": "_final", "content": text}
                return
        yield {"type": "_final", "content": ""}

    async def _invoke_writer(self, payload_json: str) -> str:
        """Buffered writer (used when output-side guardrails are configured)."""
        result = await Runner.run(
            self.writer,
            payload_json,
            max_turns=agent_max_turns(self.writer),
            run_config=_NO_TRACE,
        )
        out = result.final_output
        return str(out).strip() if out else ""

    # ----- sync convenience (tests / non-streaming callers) -----
    def run_turn(self, user_text: str) -> dict[str, Any]:
        """Run a turn to completion, returning a summary of the collected events."""

        async def _collect() -> list[dict]:
            return [ev async for ev in self.run_turn_stream(user_text)]

        events = asyncio.run(_collect())
        tokens = [e["content"] for e in events if e["type"] == "token"]
        bot = next((e for e in reversed(events) if e["type"] == "bot"), None)
        return {
            "events": events,
            "message": bot["content"] if bot else "",
            "blocks": bot["blocks"] if bot else [],
            "tokens": tokens,
        }


__all__ = ["ShoppingSession"]
