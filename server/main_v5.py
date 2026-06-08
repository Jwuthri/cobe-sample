"""FastAPI bridge between the ``agent_v5`` (agent-as-tool) assistant and the
Next.js debug console — the SAME web UI that v2/v4 use.

It speaks the exact SSE/JSON contract ``server/main_v4.py`` defines, so the
existing frontend works unchanged: point the web app's backend at this server
(it proxies ``/api/*`` → ``http://localhost:8001`` by default) and you're using
v5 instead of v4.

Pick which architecture to demo with the ``AGENT_V5_VARIANT`` env var:

    AGENT_V5_VARIANT=router   uvicorn server.main_v5:app --reload --port 8001  # with writer (default)
    AGENT_V5_VARIANT=speaking uvicorn server.main_v5:app --reload --port 8001  # no writer

You can also override per session: ``POST /api/session?variant=speaking``.
``GET /api/health`` reports the active default so you can confirm which is live.

Unlike v4 (one LangGraph streamed node-by-node), v5 runs a supervisor whose
subagents are invoked *inside* tool calls, so events are emitted at TURN
granularity (routing + step summaries + the final reply) rather than streamed
token-by-token. Chat, the live cart panel, and rich-reply blocks all work.

Sessions live in process memory (restart = clean slate), same as v4.
"""

from __future__ import annotations

import json
import os
import uuid
from decimal import Decimal
from typing import Any, AsyncGenerator

import anyio
from agent_v5.agent import ShoppingAgentV5, TurnResult
from agent_v5.supervisor import Variant
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

_DEFAULT_VARIANT: Variant = (
    "speaking" if os.environ.get("AGENT_V5_VARIANT", "router").lower() == "speaking" else "router"
)

app = FastAPI(title="agent_v5 (agent-as-tool) debug API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

SESSIONS: dict[str, ShoppingAgentV5] = {}


class NewSessionResponse(BaseModel):
    session_id: str
    variant: str


class TurnRequest(BaseModel):
    message: str


def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def _resolve_variant(raw: str | None) -> Variant:
    if raw and raw.lower() == "speaking":
        return "speaking"
    if raw and raw.lower() == "router":
        return "router"
    return _DEFAULT_VARIANT


def _get_or_create(session_id: str, variant: Variant | None = None) -> ShoppingAgentV5:
    agent = SESSIONS.get(session_id)
    if agent is None:
        agent = ShoppingAgentV5(
            variant=variant or _DEFAULT_VARIANT, user_id="demo", session_id=session_id
        )
        SESSIONS[session_id] = agent
    return agent


def serialize_state(agent: ShoppingAgentV5) -> dict[str, Any]:
    """Project the agent's live cart + transcript into the frontend's AgentSnapshot."""
    cart = agent.cart_service.cart
    return {
        "user_id": agent.user_id,
        "session_id": agent.session_id,
        # v5 has no persistent active_sop; surface the variant here for visibility.
        "active_sop": None,
        "variant": agent.variant,
        "skills_loaded": list(agent.skills_loaded),
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
            {
                "role": getattr(m, "type", "?"),
                "content": str(m.content),
                "blocks": (getattr(m, "additional_kwargs", {}) or {}).get("blocks", []),
            }
            for m in agent.messages
        ],
        "iteration": 0,
        "done": True,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "agent_v5", "variant": _DEFAULT_VARIANT}


@app.post("/api/session", response_model=NewSessionResponse)
def new_session(variant: str | None = None) -> NewSessionResponse:
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    v = _resolve_variant(variant)
    SESSIONS[sid] = ShoppingAgentV5(variant=v, user_id="demo", session_id=sid)
    return NewSessionResponse(session_id=sid, variant=v)


@app.get("/api/state/{session_id}")
def get_state(session_id: str) -> dict[str, Any]:
    agent = SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(404, f"unknown session: {session_id}")
    return serialize_state(agent)


@app.post("/api/turn/{session_id}")
async def turn(session_id: str, req: TurnRequest, variant: str | None = None) -> StreamingResponse:
    """Run one turn; stream turn-granular events as SSE (same contract as v4)."""
    agent = _get_or_create(session_id, _resolve_variant(variant) if variant else None)

    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse({"type": "user", "content": req.message})
        yield _sse({"type": "state", "snapshot": serialize_state(agent)})
        try:
            # Offload the blocking turn so the events above flush immediately.
            result: TurnResult = await anyio.to_thread.run_sync(agent.run_turn, req.message)
            for ev in _events_for_turn(agent.variant, result):
                yield _sse(ev)
            yield _sse({"type": "state", "snapshot": serialize_state(agent)})
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "content": str(e)})
        yield _sse({"type": "end"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


def _events_for_turn(variant: str, result: TurnResult) -> list[dict]:
    """Map a completed turn into the frontend's event vocabulary.

    One (router → agent → step) triple per subagent that ran, then the writer
    line (router variant only — it reflects the real extra LLM call), then the
    final bot message carrying the typed blocks.
    """
    out: list[dict] = []
    for i, sr in enumerate(result.step_results, 1):
        out.append({"type": "router", "target": sr.sop, "iteration": i})
        out.append({"type": "agent", "node": f"{sr.sop}_wrapper"})
        out.append(
            {
                "type": "step",
                "sop": sr.sop,
                "summary": sr.summary,
                "asks": list(sr.asks),
                "next_sop": sr.next_sop,
                "details": sr.details,
            }
        )
    if variant == "router":
        out.append({"type": "router", "target": "writer", "iteration": 0})
        out.append({"type": "writer", "draft": result.message, "blocks": result.blocks})
    out.append({"type": "bot", "content": result.message, "blocks": result.blocks})
    return out
