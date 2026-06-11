"""``ShoppingSession`` — the per-session, streaming-first turn engine (Agno).

A turn is:

  1. input guardrails (pre-flight, before the team runs) — a refusal is instant,
     a redaction rewrites the user text before it reaches any model;
  2. the coordinate-mode supervisor TEAM runs, streamed: the leader delegates to
     members (``router`` / ``tool_start`` / ``tool_end`` / ``step`` events as they
     happen), then authors the single user-facing reply — whose tokens stream
     straight to the client as ``{type:"token"}`` (the leader's ``TeamRunContent``);
  3. deterministic blocks are attached from the verified ``StepResult`` s + cart,
     and a final ``{type:"bot"}`` carries the authoritative text + blocks.

The streaming-safety story is preserved from agent_v4_1: the leader's reply is the
LAST thing in the turn (nothing validates after it), the cart invariant gates
confirmation (not model prose), and structured cards are built deterministically
(ids/prices verbatim) so the streamed prose can never invent them.

Discrimination between the leader's user-facing tokens and a member's internal
chatter is purely on the Agno ``.event`` string (``TeamRunContent`` vs
``RunContent``); see :mod:`agent_agno_v1.core.events`.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, AsyncGenerator

from agno.db.in_memory import InMemoryDb

from agent_agno_v1.core.context import ToolEvent, TurnContext
from agent_agno_v1.core.events import (
    DELEGATE_TOOLS,
    LEADER_TOKEN,
    LEADER_TOOL_END,
    LEADER_TOOL_START,
    MEMBER_TOOL_END,
    MEMBER_TOOL_START,
    TEAM_RUN_COMPLETED,
    agent_event,
    canonical_member,
    delegate_target,
    ev_content,
    ev_name,
    ev_tool,
    is_member_not_found,
    router_event,
    step_event,
    token_event,
    tool_args,
    tool_end_event,
    tool_name,
    tool_result,
    tool_start_event,
)
from agent_agno_v1.core.guardrails import CompiledInputRule, run_input_guardrails
from agent_agno_v1.shopping.agents import BLOCK_BY_SOP
from agent_agno_v1.shopping.blocks import build_blocks
from agent_agno_v1.shopping.domain import CartService, MemoryStore, build_store
from agent_agno_v1.shopping.extractors import EXTRACTORS, cart_quantities, checkout_anchor_text
from agent_agno_v1.shopping.platform import build_supervisor

_EMPTY_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"


def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


@dataclass
class ShoppingSession:
    """One conversation: a clean transcript + a live cart + the compiled team."""

    user_id: str = "demo"
    session_id: str = "sess-agno1"
    messages: list[dict] = field(default_factory=list)
    cart_service: CartService = field(default_factory=CartService)
    store: MemoryStore = field(default_factory=build_store)
    input_rules: list[CompiledInputRule] = field(default_factory=list)

    # Injected for tests; built from the platform otherwise.
    team: Any = None
    db: Any = None

    def __post_init__(self) -> None:
        if self.db is None:
            self.db = InMemoryDb()
        if self.team is None:
            self.team = build_supervisor(self.cart_service, self.store, db=self.db)

    # ----- grounding state injected into the team each turn -----
    def _grounding_state(self) -> dict[str, Any]:
        cart = self.cart_service.cart
        return {
            "checkout_progress": checkout_anchor_text(cart),
            "cart": {
                "step": cart.step.value,
                "items": [
                    {"id": i.product_id, "name": i.name, "qty": i.quantity, "price": _d(i.unit_price)}
                    for i in cart.items
                ],
                "subtotal": _d(cart.subtotal),
                "grand_total": _d(cart.grand_total) if cart.grand_total is not None else None,
                "customer": cart.customer.model_dump(),
                "address": cart.address.model_dump(),
                "serviceable": cart.serviceable,
                "serviceable_options": list(cart.serviceable_options),
                "delivery_option": cart.delivery_option,
                "payment_method": cart.payment_method,
                "confirmed": cart.confirmed,
                "receipt_id": cart.receipt_id,
                "ready_to_confirm": cart.ready_to_confirm(),
                "blockers": [{"code": b.code, "message": b.message} for b in cart.blockers()],
            },
        }

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
            "messages": list(self.messages),
            "iteration": 0,
            "done": True,
        }

    # ----- the streaming turn -----
    async def run_turn_stream(self, user_text: str) -> AsyncGenerator[dict, None]:
        yield {"type": "user", "content": user_text}
        yield {"type": "state", "snapshot": self.snapshot()}

        # 1. input guardrails (pre-flight, before any model call)
        outcome = run_input_guardrails(self.input_rules, user_text)
        for hit in outcome.triggered:
            yield {"type": "guardrail", "stage": "input", "rule": hit.type, "action": hit.action}
        if not outcome.allowed:
            self.messages.append({"role": "human", "content": user_text})
            self.messages.append({"role": "ai", "content": outcome.refusal or "", "blocks": []})
            yield {"type": "bot", "content": outcome.refusal or "", "blocks": []}
            yield {"type": "state", "snapshot": self.snapshot()}
            yield {"type": "end"}
            return

        clean_text = outcome.text
        self.messages.append({"role": "human", "content": clean_text})

        # 2. stream the coordinate-mode supervisor team
        turn = TurnContext(user_id=self.user_id, session_id=self.session_id)
        before_qty: dict[str, int] = {}
        emitted_steps = 0
        streamed_text = ""
        completed_content = ""
        first_token = True

        try:
            # arun(stream=True) returns an async generator directly in agno 2.6.x;
            # guard for a coroutine-returning variant for forward-compatibility.
            stream = self.team.arun(
                clean_text,
                stream=True,
                stream_events=True,
                session_id=self.session_id,
                user_id=self.user_id,
                session_state=self._grounding_state(),
            )
            if inspect.iscoroutine(stream):
                stream = await stream
            async for ev in stream:
                name = ev_name(ev)

                if name == LEADER_TOOL_START:
                    tool = ev_tool(ev)
                    if tool_name(tool) in DELEGATE_TOOLS:
                        member = canonical_member(delegate_target(tool))
                        turn.current_member = member
                        before_qty = cart_quantities(self.cart_service.cart)
                        yield router_event(member)
                    elif tool is not None:
                        yield tool_start_event(tool)

                elif name == LEADER_TOOL_END:
                    tool = ev_tool(ev)
                    if tool_name(tool) in DELEGATE_TOOLS:
                        member = turn.current_member or canonical_member(delegate_target(tool))
                        turn.current_member = None
                        if is_member_not_found(tool_result(tool)):
                            continue  # a mis-addressed delegation; the leader will retry
                        extractor = EXTRACTORS.get(member)
                        if extractor is not None:
                            sr = extractor(
                                self.cart_service.cart, turn.member_tool_events(member), before_qty
                            )
                            turn.step_results.append(sr)
                        yield agent_event(member)
                        while emitted_steps < len(turn.step_results):
                            yield step_event(turn.step_results[emitted_steps])
                            emitted_steps += 1

                elif name == MEMBER_TOOL_START:
                    tool = ev_tool(ev)
                    if tool is not None:
                        yield tool_start_event(tool)

                elif name == MEMBER_TOOL_END:
                    tool = ev_tool(ev)
                    if tool is not None:
                        turn.tool_events.append(
                            ToolEvent(
                                sop=turn.current_member or "?",
                                name=tool_name(tool),
                                args=tool_args(tool),
                                result=tool_result(tool),
                            )
                        )
                        yield tool_end_event(tool)

                elif name == LEADER_TOKEN:
                    delta = ev_content(ev)
                    if delta:
                        if first_token:
                            first_token = False
                            yield {"type": "state", "snapshot": self.snapshot()}
                        streamed_text += delta
                        yield token_event(delta)

                elif name == TEAM_RUN_COMPLETED:
                    content = getattr(ev, "content", None)
                    if isinstance(content, str) and content.strip():
                        completed_content = content
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "content": str(e)}
            yield {"type": "end"}
            return

        # drain any tail step events
        while emitted_steps < len(turn.step_results):
            yield step_event(turn.step_results[emitted_steps])
            emitted_steps += 1

        # 3. blocks + finalize
        text = streamed_text.strip() or completed_content.strip() or _EMPTY_FALLBACK
        blocks = build_blocks(turn.step_results, self.cart_service.cart, BLOCK_BY_SOP)
        self.messages.append({"role": "ai", "content": text, "blocks": blocks})

        yield {"type": "writer", "draft": text, "blocks": blocks}
        yield {"type": "bot", "content": text, "blocks": blocks}
        yield {"type": "state", "snapshot": self.snapshot()}
        yield {"type": "end"}

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
