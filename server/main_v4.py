"""FastAPI bridge between the ``agent_v4`` graph and the Next.js debug
console.

Endpoints:
  - POST /api/session             → create a fresh session
  - GET  /api/state/{session_id}  → current state snapshot
  - POST /api/turn/{session_id}   → run a turn, stream events as SSE
  - GET  /api/health              → liveness

Sessions live in process memory. Restart = clean slate (same trade-off
as the CLI). Production would swap ``SESSIONS`` for a checkpointer-
backed store.

Run:
    uvicorn server.main_v4:app --reload --port 8001
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import Any, AsyncGenerator

from agent_v4 import debug_log
from agent_v4.config import setup_debug_logging
from agent_v4.graph import build_graph
from agent_v4.state import AgentState
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

setup_debug_logging()

app = FastAPI(title="agent_v4 debug API", version="0.1.0")

# Loose CORS for local dev; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH = build_graph()
SESSIONS: dict[str, AgentState] = {}


# ----- request / response models -----
class NewSessionResponse(BaseModel):
    session_id: str


class TurnRequest(BaseModel):
    message: str


# ----- serialization -----
def _d(value: Any) -> Any:
    """Decimal → str for JSON."""
    return str(value) if isinstance(value, Decimal) else value


def serialize_state(state: AgentState) -> dict[str, Any]:
    cart = state.cart
    return {
        "user_id": state.user_id,
        "session_id": state.session_id,
        "active_sop": state.active_sop.value if state.active_sop else None,
        "skills_loaded": list(state.skills_loaded),
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
                {
                    "cost": _d(cart.shipping.cost),
                    "eta_hours": cart.shipping.eta_hours,
                }
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
            {"role": m.type if hasattr(m, "type") else "?", "content": str(m.content)}
            for m in state.messages
        ],
        "iteration": state.iteration,
        "done": state.done,
    }


# ----- endpoints -----
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session", response_model=NewSessionResponse)
def new_session() -> NewSessionResponse:
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    SESSIONS[sid] = AgentState(user_id="demo", session_id=sid)
    return NewSessionResponse(session_id=sid)


@app.get("/api/state/{session_id}")
def get_state(session_id: str) -> dict[str, Any]:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(404, f"unknown session: {session_id}")
    return serialize_state(state)


@app.post("/api/turn/{session_id}")
async def turn(session_id: str, req: TurnRequest) -> StreamingResponse:
    """Run one turn, stream events to the client as Server-Sent Events.

    Event stream contract (each line is `data: <json>\\n\\n`):

      {type:"user", content:"..."}                  initial echo
      {type:"router", target:"<sop>|writer", iteration:n}
      {type:"agent", node:"checkout_wrapper"}
      {type:"skill", name:"collect_address"}
      {type:"tool_start", name:"...", args:{...}}
      {type:"tool_end", name:"...", result:"..."}
      {type:"step", sop:"checkout", summary:"...", asks:[...], next_sop:null}
      {type:"writer", draft:"..."}
      {type:"gate", rejected:true, errors:[...]}
      {type:"validator", errors:[...]}
      {type:"bot", content:"final assistant message"}
      {type:"state", snapshot:{...full cart+state...}}
      {type:"end"}
      {type:"error", content:"..."}
    """
    if session_id not in SESSIONS:
        SESSIONS[session_id] = AgentState(user_id="demo", session_id=session_id)

    base = SESSIONS[session_id]
    state = base.model_copy(
        update={
            "messages": base.messages + [HumanMessage(content=req.message)],
            "draft_response": None,
            "validation_errors": [],
            "response_attempts": 0,
            "step_results": [],
            "iteration": 0,
            "done": False,
        }
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal state
        yield _sse({"type": "user", "content": req.message})
        yield _sse({"type": "state", "snapshot": serialize_state(state)})

        try:
            for chunk in GRAPH.stream(state, stream_mode=["updates", "custom"]):
                stream_mode, payload = chunk
                for ev in _classify_chunk(stream_mode, payload):
                    debug_log.sse(ev)
                    yield _sse(ev)
                if stream_mode == "updates":
                    for _node, update in payload.items():
                        if isinstance(update, dict):
                            state = _merge(state, update)
                    yield _sse({"type": "state", "snapshot": serialize_state(state)})
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "content": str(e)})

        SESSIONS[session_id] = state
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


# ----- helpers -----
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


def _classify_chunk(mode: str, payload: Any) -> list[dict]:
    """Turn a LangGraph stream chunk into 0+ UI events."""
    out: list[dict] = []
    if mode == "custom":
        ev = payload.get("event") if isinstance(payload, dict) else None
        if ev == "tool_start":
            name = payload.get("tool")
            args = {k: v for k, v in (payload.get("args") or {}).items() if k != "tool_call_id"}
            if name == "load_skill":
                out.append({"type": "skill", "name": args.get("skill_name")})
            else:
                out.append({"type": "tool_start", "name": name, "args": args})
        elif ev == "tool_end":
            out.append(
                {
                    "type": "tool_end",
                    "name": payload.get("tool"),
                    "result": payload.get("result", ""),
                }
            )
        return out

    if mode == "updates" and isinstance(payload, dict):
        for node, update in payload.items():
            if not isinstance(update, dict):
                continue
            if node == "supervisor":
                sop = update.get("active_sop")
                iteration = update.get("iteration", 0)
                if sop is not None:
                    out.append(
                        {
                            "type": "router",
                            "target": sop.value if hasattr(sop, "value") else str(sop),
                            "iteration": iteration,
                        }
                    )
                elif iteration == 0:
                    out.append({"type": "router", "target": "writer", "iteration": 0})
            elif node.endswith("_wrapper"):
                out.append({"type": "agent", "node": node})
                for sr in update.get("step_results", []) or []:
                    out.append(
                        {
                            "type": "step",
                            "sop": sr.sop.value,
                            "summary": sr.summary,
                            "asks": list(sr.asks),
                            "next_sop": sr.next_sop.value if sr.next_sop else None,
                            "details": sr.details,
                        }
                    )
            elif node == "writer":
                draft = update.get("draft_response", "")
                if draft:
                    out.append({"type": "writer", "draft": draft})
            elif node == "checkout_gate":
                errs = update.get("validation_errors") or []
                if errs:
                    out.append(
                        {"type": "gate", "rejected": True, "errors": [e.detail for e in errs]}
                    )
            elif node == "validator":
                errs = update.get("validation_errors") or []
                if errs:
                    out.append({"type": "validator", "errors": [e.code for e in errs]})
            elif node == "emit":
                for m in update.get("messages") or []:
                    if isinstance(m, AIMessage):
                        out.append({"type": "bot", "content": str(m.content)})
    return out


def _merge(state: AgentState, update: dict) -> AgentState:
    """Apply a partial update dict to the Pydantic state.

    Mirrors what LangGraph itself does internally for the channel
    schema, so our serialized snapshot reflects the latest state
    without a second graph query. The ``messages`` field uses the
    ``add_messages`` reducer (append, not replace), so when a node
    emits ``{"messages": [AIMessage(...)]}`` we must APPEND to the
    current list — otherwise the AI's reply silently drops on the
    floor and the chat tab never shows bot answers.
    """
    appended = update.get("messages") or []
    safe = {k: v for k, v in update.items() if k != "messages"}
    if appended:
        safe["messages"] = list(state.messages) + list(appended)
    if not safe:
        return state
    try:
        return state.model_copy(update=safe)
    except Exception:
        # Schema-mismatched fields (e.g. an internal _-prefixed field)
        # are ignored. We never crash the stream on serialization issues.
        return state
