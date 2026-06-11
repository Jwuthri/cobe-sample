"""``ShoppingSession`` — the per-session, streaming turn engine (Agno edition).

A turn is:

  1. the **Team** routes the message to its member sub-agents (awaited), each
     mutating the shared cart via ``dependencies``; we replay their work as
     router/tool/agent events and extract one ``StepResult`` per member;
  2. the **writer** — a separate, terminal Agent — streams its tokens straight to
     the client (``{type:"token"}``), retried once if it emits nothing;
  3. deterministic **blocks** are assembled from the StepResults + cart (ids and
     prices verbatim — the hallucination firewall) and a final ``{type:"bot"}``
     carries the authoritative text + blocks.

Because the writer is the last model call and only ever sees grounded step
results + the cart, there is no post-generation validator gating the stream.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, AsyncGenerator

from agno.run.agent import RunContentEvent

from agent_agno_v4_1.agents import build_team, build_writer
from agent_agno_v4_1.context import ShoppingContext
from agent_agno_v4_1.events import member_events, step_event
from agent_agno_v4_1.extractors import BLOCK_BY_SOP, EXTRACTORS
from agent_agno_v4_1.writer_payload import build_writer_payload
from agent_v4_1.shopping.blocks import build_blocks
from agent_v4_1.shopping.domain import CartService
from agent_v4_1.shopping.domain.memory import build_store

_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"


def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


async def _aiter(maybe):
    """Normalize ``arun(stream=True)`` — async-iterator or coroutine-of-iterator."""
    if inspect.iscoroutine(maybe):
        maybe = await maybe
    async for ev in maybe:
        yield ev


@dataclass
class ShoppingSession:
    """One conversation: clean transcript + live cart + the compiled agents."""

    user_id: str = "demo"
    session_id: str = "sess-agno-v41"
    messages: list[dict] = field(default_factory=list)
    cart_service: CartService = field(default_factory=CartService)
    store: Any = None

    team: Any = None
    writer: Any = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = build_store()
        if self.team is None:
            self.team = build_team()
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
            "messages": list(self.messages),
            "iteration": 0,
            "done": True,
        }

    # ----- the streaming turn -----
    async def run_turn_stream(self, user_text: str) -> AsyncGenerator[dict, None]:
        yield {"type": "user", "content": user_text}
        yield {"type": "state", "snapshot": self.snapshot()}

        self.messages.append({"role": "user", "content": user_text})
        ctx = ShoppingContext(
            user_id=self.user_id,
            session_id=self.session_id,
            cart_service=self.cart_service,
            store=self.store,
        )
        before = {i.product_id: i.quantity for i in self.cart_service.cart.items}

        # 1. team phase — route to member sub-agents, then replay their work.
        try:
            team_out = await self.team.arun(user_text, dependencies=ctx.as_dependencies())
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "content": str(e)}
            yield {"type": "end"}
            return

        for mr in team_out.member_responses or []:
            name = getattr(mr, "agent_name", None) or getattr(mr, "team_name", None)
            extractor = EXTRACTORS.get(name)
            if extractor is None:
                continue
            tools = getattr(mr, "tools", None) or []
            for ev in member_events(name, tools):
                yield ev
            sr = extractor(ctx, tools, before)
            ctx.step_results.append(sr)
            yield step_event(sr)

        yield {"type": "state", "snapshot": self.snapshot()}

        # 2. writer phase — the terminal model call, streamed token-by-token.
        yield {"type": "router", "target": "writer", "iteration": 0}
        payload_json, _mode = build_writer_payload(
            self.messages, ctx.step_results, self.cart_service.cart
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

        Empty-retry is stream-safe: an empty stream sent zero tokens, so the
        retry is invisible to the client.
        """
        for _attempt in (1, 2):
            parts: list[str] = []
            async for ev in _aiter(self.writer.arun(payload_json, stream=True)):
                if isinstance(ev, RunContentEvent) and ev.content:
                    parts.append(ev.content)
                    yield {"type": "token", "content": ev.content}
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
