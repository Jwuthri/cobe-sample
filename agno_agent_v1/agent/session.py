"""``ShoppingSession`` — the per-session, streaming-first turn engine.

A turn is:

  1. input guardrails (pre-flight, before any model call) — a refusal is instant,
     a redaction rewrites the user text before it enters the transcript;
  2. the orchestrator routes to sub-agent tools (its UI events — router / tool /
     step / trace — are buffered on the context and drained here);
  3. the writer — the LAST model call, with nothing after it — streams its tokens
     straight to the client (``{type:"token"}``), retried once if it emits nothing;
  4. deterministic blocks are attached and a final ``{type:"bot"}`` carries the
     authoritative text + blocks.

Because the writer is terminal and only ever sees verified step results + cart
(blocks are built deterministically), there is no post-generation validator to
gate the stream — the grounding happened at construction.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, AsyncGenerator

from agno.models.message import Message
from agno.run.agent import RunContentEvent

from agno_agent_v1.agent.agents import BLOCK_BY_SOP, ROUTER_PROMPT_DELEGATES, build_orchestrator, build_writer
from agno_agent_v1.agent.blocks import build_blocks
from agno_agent_v1.agent.context import ShoppingContext
from agno_agent_v1.agent.events import render_messages, trace_event
from agno_agent_v1.agent.guardrails import CompiledInputRule, run_input_guardrails
from agno_agent_v1.agent.prompts import ROUTER_PROMPT, WRITER_SYSTEM
from agno_agent_v1.agent.writer_payload import build_writer_payload
from agno_agent_v1.domain import CartService, MemoryStore, build_store

_TRACE_HISTORY = 24
_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"


def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


@dataclass
class ShoppingSession:
    """One conversation: plain transcript + live cart/store + the compiled agents."""

    user_id: str = "demo"
    session_id: str = "sess-agno-v1"
    messages: list[dict] = field(default_factory=list)
    cart_service: CartService = field(default_factory=CartService)
    store: MemoryStore = field(default_factory=build_store)
    turn: int = 0
    # Persisted per-step ``recall`` snippets (keyed by sop), carried to the next
    # turn's orchestrator so it can resolve references WITHOUT the sub-agents
    # seeing the chat. Opaque to the session — domain code renders the content.
    routing_notes: dict[str, str] = field(default_factory=dict)

    writer: Any = None
    input_rules: list[CompiledInputRule] = field(default_factory=list)
    debug: bool = True

    def __post_init__(self) -> None:
        if self.writer is None:
            self.writer = build_writer()

    # ----- snapshot (the frontend's AgentSnapshot) -----
    def snapshot(self) -> dict[str, Any]:
        cart = self.cart_service.cart
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "active_sop": None,
            "skills_loaded": [],
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
                {"role": m.get("role"), "content": str(m.get("content", "")), "blocks": m.get("blocks", [])}
                for m in self.messages
            ],
            "iteration": 0,
            "done": True,
        }

    # ----- routing memo (the orchestrator owns reference resolution) -----
    def _routing_memo(self, ctx: ShoppingContext) -> str | None:
        live = ctx.routing_context()
        blocks = [text for text in live.values() if text]
        blocks += [text for key, text in self.routing_notes.items() if key not in live and text]
        if not blocks:
            return None
        return (
            "Routing context for resolving the user's references (authoritative — do "
            "NOT invent; if a reference is ambiguous, delegate to a sub-agent to look "
            "it up):\n" + "\n".join(blocks)
        )

    def _absorb_recalls(self, ctx: ShoppingContext) -> None:
        for sr in ctx.step_results:
            if sr.recall:
                self.routing_notes[sr.sop] = sr.recall

    # ----- orchestrator input (transcript as Agno messages + routing memo) -----
    def _orchestrator_input(self, memo: str | None) -> list[Message]:
        msgs = [Message(role=m["role"], content=str(m["content"])) for m in self.messages]
        if memo and msgs:
            msgs.insert(len(msgs) - 1, Message(role="system", content=memo))
        return msgs

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
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": outcome.refusal or "", "blocks": []})
            yield {"type": "bot", "content": outcome.refusal or "", "blocks": []}
            yield {"type": "state", "snapshot": self.snapshot()}
            yield {"type": "end"}
            return

        self.messages.append({"role": "user", "content": outcome.text})
        ctx = ShoppingContext(
            user_id=self.user_id,
            session_id=self.session_id,
            cart_service=self.cart_service,
            store=self.store,
            debug=self.debug,
        )

        memo = self._routing_memo(ctx)
        orch_input = self._orchestrator_input(memo)

        if self.debug:
            yield trace_event(
                "orchestrator_input",
                "orchestrator",
                "user turn → orchestrator",
                {
                    "system_prompt": ROUTER_PROMPT,
                    "delegates": ROUTER_PROMPT_DELEGATES,
                    "routing_memo": memo,
                    "messages_total": len(self.messages),
                    "conversation_seen": render_messages(orch_input[-_TRACE_HISTORY:]),
                    "context": ctx.debug_view(),
                },
            )

        # 2. orchestrator phase — run to completion in a thread (its sub-agent
        # wrappers buffer UI events onto ctx.events), then drain those events.
        cart_empty = not self.cart_service.cart.items
        orchestrator = build_orchestrator(cart_empty=cart_empty)
        try:
            await asyncio.to_thread(orchestrator.run, orch_input, dependencies={"ctx": ctx})
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "content": str(e)}
            yield {"type": "end"}
            return

        for ev in ctx.events:
            yield ev

        self._absorb_recalls(ctx)
        yield {"type": "state", "snapshot": self.snapshot()}

        # 3. writer phase — the terminal model call, streamed token-by-token
        yield {"type": "router", "target": "writer", "iteration": 0}
        payload_json, mode = build_writer_payload(
            self.messages, ctx.step_results, self.cart_service.cart
        )

        if self.debug:
            try:
                parsed_payload = json.loads(payload_json)
            except json.JSONDecodeError:
                parsed_payload = {"raw": payload_json}
            yield trace_event(
                "writer_payload",
                "writer",
                f"writer input (mode={mode})",
                {"system_prompt": WRITER_SYSTEM, "mode": mode, "payload": parsed_payload},
            )

        text = ""
        try:
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
        self.messages.append({"role": "assistant", "content": text, "blocks": blocks})

        yield {"type": "writer", "draft": text, "blocks": blocks}
        yield {"type": "bot", "content": text, "blocks": blocks}
        yield {"type": "state", "snapshot": self.snapshot()}
        yield {"type": "end"}

    async def _stream_writer(self, payload_json: str) -> AsyncGenerator[dict, None]:
        """Stream writer tokens, retrying once if the first attempt is empty.

        Empty-retry is stream-safe: an empty stream sent zero tokens, so the retry
        is invisible to the client.
        """
        for _attempt in (1, 2):
            parts: list[str] = []
            stream = self.writer.arun(payload_json, stream=True)
            if inspect.iscoroutine(stream):
                stream = await stream
            async for event in stream:
                if isinstance(event, RunContentEvent) and event.content:
                    parts.append(event.content)
                    yield {"type": "token", "content": event.content}
            text = "".join(parts).strip()
            if text:
                yield {"type": "_final", "content": text}
                return
        yield {"type": "_final", "content": ""}

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
