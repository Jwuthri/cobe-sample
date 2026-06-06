"""FastAPI bridge between the ``agent_v3`` (Agno) workflow and the Next.js
debug console — a drop-in replacement for ``server/main.py``.

Same endpoints + identical SSE event contract as the v2 server, so the
existing Next.js frontend works unchanged. Run EITHER server on :8001:

    uvicorn server.main_v3:app --reload --port 8001   # Agno (v3)
    uvicorn server.main:app    --reload --port 8001   # LangGraph (v2)

Sessions live in process memory (same trade-off as v2). The agent state
is an Agno ``session_state`` dict (not a Pydantic AgentState); the live
cart for a turn is rebuilt from its serialized snapshot each turn inside
``agent_v3.workflow``.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_v3.state import fresh_state, load_cart
from agent_v3.workflow import stream_turn

app = FastAPI(title="agent_v3 (Agno) debug API", version="0.1.0")

# Loose CORS for local dev; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, dict[str, Any]] = {}

# Events after which the cart/state may have changed -> push a fresh snapshot.
_STATE_DIRTY_EVENTS = {"agent", "step", "gate", "validator", "bot"}


class NewSessionResponse(BaseModel):
    session_id: str


class TurnRequest(BaseModel):
    message: str


def _d(value: Any) -> Any:
    """Decimal → str for JSON."""
    return str(value) if isinstance(value, Decimal) else value


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Match the v2 server's AgentSnapshot shape, sourced from session_state."""
    cart = load_cart(state)
    return {
        "user_id": state.get("user_id", "demo"),
        "session_id": state.get("session_id", ""),
        "active_sop": state.get("active_sop"),
        "skills_loaded": list(state.get("skills_loaded", [])),
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
            {"role": m.get("role", "?"), "content": str(m.get("content", ""))}
            for m in state.get("messages", [])
        ],
        "iteration": state.get("iteration", 0),
        "done": state.get("done", False),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "agno-agent_v3"}


@app.post("/api/session", response_model=NewSessionResponse)
def new_session() -> NewSessionResponse:
    state = fresh_state(user_id="demo")
    sid = state["session_id"]
    SESSIONS[sid] = state
    return NewSessionResponse(session_id=sid)


@app.get("/api/state/{session_id}")
def get_state(session_id: str) -> dict[str, Any]:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(404, f"unknown session: {session_id}")
    return serialize_state(state)


@app.post("/api/turn/{session_id}")
async def turn(session_id: str, req: TurnRequest) -> StreamingResponse:
    """Run one turn, stream events as SSE (identical contract to the v2 server)."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = fresh_state(user_id="demo", session_id=session_id)
    state = SESSIONS[session_id]

    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse({"type": "user", "content": req.message})
        yield _sse({"type": "state", "snapshot": serialize_state(state)})
        try:
            for ev in stream_turn(state, req.message):
                yield _sse(ev)
                if ev.get("type") in _STATE_DIRTY_EVENTS:
                    yield _sse({"type": "state", "snapshot": serialize_state(state)})
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "content": str(e)})
        yield _sse({"type": "state", "snapshot": serialize_state(state)})
        yield _sse({"type": "end"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"
