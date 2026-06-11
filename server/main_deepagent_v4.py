"""FastAPI bridge for agent_deepagent_v4 — wired to the existing `web/` frontend.

This speaks the SAME contract the Next.js console already uses for agent_v4
(see ``web/lib/api.ts`` + ``web/lib/types.ts``), so the browser UI works with no
frontend changes:

  POST /api/session                 → {session_id}
  GET  /api/state/{session_id}      → AgentSnapshot
  POST /api/turn/{session_id}       → SSE stream of ServerEvents (user, router,
                                       agent, tool_start/end, step, writer, bot,
                                       state, end)

The deep agent isn't a fixed graph, so we run the turn and then reconstruct the
SSE events from the orchestrator's message trace (each `task` tool call → which
subagent ran). The chat + cart panels render live; the event panel shows the
delegation trace.

**Safe checkout over the existing UI.** The frontend has no approval widget, so
the human-in-the-loop pause is surfaced *conversationally*: when the order is
ready, the bot replies with the order summary and asks the customer to reply
"yes". That next "yes"/"no" resumes the ``interrupt()`` — so the order still only
places on an explicit approval (the safety gate is preserved, NOT auto-approved).

Run (frontend rewrites /api/* → AGENT_V2_API_URL, default http://localhost:8001):
    uvicorn server.main_deepagent_v4:app --reload --port 8001
  or keep 8002 and start the web app with AGENT_V2_API_URL=http://localhost:8002
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from agent_deepagent_v4.config import load_env
from agent_deepagent_v4.domain.cart import Cart
from agent_deepagent_v4.messages import text_of
from agent_deepagent_v4.runtime import cart_service_for, reset_session, resume_turn, run_turn

load_env()

app = FastAPI(title="agent_deepagent_v4 API (frontend-wired)", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Per-session UI state the frontend needs but the agent doesn't track itself.
TRANSCRIPTS: dict[str, list[dict[str, Any]]] = {}  # clean role/content message log
PENDING_APPROVAL: dict[str, dict[str, Any]] = {}  # session_id -> interrupt payload

# Map our subagent names to the SOPName vocabulary the frontend's panels expect.
SUBAGENT_TO_SOP = {
    "product-agent": "product_rec",
    "checkout-agent": "checkout",
    "order-status-agent": "order_status",
    "writer-agent": None,
}


# ----- request models -----
class NewSessionResponse(BaseModel):
    session_id: str


class TurnRequest(BaseModel):
    message: str
    user_id: str = "demo"


# ----- serialization (matches web/lib/types.ts) -----
def _d(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def _fe_cart(cart: Cart) -> dict[str, Any]:
    return {
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
            {"amount": _d(cart.tax.amount), "rate": _d(cart.tax.rate)} if cart.tax_is_fresh() else None
        ),
        "promo": ({"code": cart.promo.code, "discount": _d(cart.promo.discount)} if cart.promo else None),
        "payment_method": cart.payment_method,
        "card_token_set": bool(cart.card_token),
        "subtotal": _d(cart.subtotal),
        "grand_total": _d(cart.grand_total) if cart.grand_total is not None else None,
        "blockers": [{"code": b.code, "message": b.message} for b in cart.blockers()],
        "ready_to_confirm": cart.ready_to_confirm(),
        "confirmed": cart.confirmed,
        "receipt_id": cart.receipt_id,
    }


def _snapshot(session_id: str, *, active_sop: str | None = None) -> dict[str, Any]:
    cart = cart_service_for(session_id).cart
    return {
        "user_id": "demo",
        "session_id": session_id,
        "active_sop": active_sop,
        "skills_loaded": [],
        "cart": _fe_cart(cart),
        "messages": list(TRANSCRIPTS.get(session_id, [])),
        "iteration": 0,
        "done": session_id not in PENDING_APPROVAL,
    }


# ----- event reconstruction from the orchestrator trace -----
def _new_messages(raw: dict[str, Any]) -> list[Any]:
    """The messages produced THIS turn (everything after the last user message)."""
    messages = raw.get("messages") or []
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[last_human + 1 :]


def _delegation_events(new_messages: list[Any]) -> list[dict[str, Any]]:
    """Turn the `task` tool calls + results into router/tool/agent/step events."""
    events: list[dict[str, Any]] = []
    call_subagent: dict[str, str] = {}  # tool_call_id -> subagent_type
    iteration = 0
    for m in new_messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.get("name") != "task":
                    continue
                args = tc.get("args") or {}
                sub = args.get("subagent_type", "?")
                call_subagent[tc.get("id", "")] = sub
                iteration += 1
                events.append({"type": "router", "target": sub, "iteration": iteration})
                events.append(
                    {
                        "type": "tool_start",
                        "name": "task",
                        "args": {"subagent_type": sub, "description": args.get("description", "")},
                    }
                )
        elif isinstance(m, ToolMessage) and getattr(m, "name", None) == "task":
            sub = call_subagent.get(getattr(m, "tool_call_id", ""), "?")
            result = text_of(m)
            events.append({"type": "tool_end", "name": "task", "result": result})
            events.append({"type": "agent", "node": sub})
            if sub != "writer-agent":
                events.append(
                    {
                        "type": "step",
                        "sop": SUBAGENT_TO_SOP.get(sub) or sub,
                        "summary": result[:240],
                        "asks": [],
                        "next_sop": None,
                        "details": None,
                    }
                )
    return events


def _last_worker_sop(new_messages: list[Any]) -> str | None:
    sop = None
    for m in new_messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.get("name") == "task":
                    mapped = SUBAGENT_TO_SOP.get((tc.get("args") or {}).get("subagent_type", ""))
                    if mapped:
                        sop = mapped
    return sop


# ----- approval (conversational HITL) -----
_YES_WORDS = ("yes", "y", "yeah", "yep", "confirm", "place", "go ahead", "do it", "approve", "sure")


def _parse_decision(message: str) -> dict[str, Any]:
    m = message.strip().lower()
    if m.startswith(_YES_WORDS) or m in _YES_WORDS:
        return {"approved": True}
    return {"approved": False, "reason": message.strip()}


def _format_approval(payload: dict[str, Any]) -> str:
    s = (payload or {}).get("summary", {}) or {}
    items = ", ".join(f"{i.get('name')} ×{i.get('qty')}" for i in s.get("items", [])) or "your items"
    total = s.get("grand_total")
    pay = s.get("payment_method")
    ship = s.get("ship_to")
    lines = [
        "Please confirm your order before I place it:",
        f"  • {items}",
    ]
    if total:
        lines.append(f"  • Total: ${total}")
    if pay or ship:
        lines.append(f"  • Paying by {pay or '—'}, shipping to {ship or '—'}")
    lines.append("")
    lines.append("Reply 'yes' to place the order, or 'no' to cancel.")
    return "\n".join(lines)


# ----- endpoints -----
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session", response_model=NewSessionResponse)
def new_session() -> NewSessionResponse:
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    TRANSCRIPTS[sid] = []
    PENDING_APPROVAL.pop(sid, None)
    cart_service_for(sid)
    return NewSessionResponse(session_id=sid)


@app.get("/api/state/{session_id}")
def get_state(session_id: str) -> dict[str, Any]:
    if session_id not in TRANSCRIPTS:
        # Be lenient: the frontend may call this for a session it created elsewhere.
        TRANSCRIPTS.setdefault(session_id, [])
    return _snapshot(session_id)


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


@app.post("/api/turn/{session_id}")
async def turn(session_id: str, req: TurnRequest) -> StreamingResponse:
    TRANSCRIPTS.setdefault(session_id, [])

    async def event_stream() -> AsyncGenerator[str, None]:
        transcript = TRANSCRIPTS[session_id]
        transcript.append({"role": "user", "content": req.message})
        yield _sse({"type": "user", "content": req.message})
        yield _sse({"type": "state", "snapshot": _snapshot(session_id)})

        try:
            pending = PENDING_APPROVAL.pop(session_id, None)
            if pending is not None:
                # This message is the answer to a pending safe-checkout approval.
                decision = _parse_decision(req.message)
                yield _sse(
                    {
                        "type": "agent",
                        "node": "checkout-agent" if decision["approved"] else "checkout-agent (declined)",
                    }
                )
                result = resume_turn(session_id, decision, user_id=req.user_id)
            else:
                result = run_turn(session_id, req.message, user_id=req.user_id)

            new_messages = _new_messages(result.raw)
            for ev in _delegation_events(new_messages):
                yield _sse(ev)

            if result.needs_approval:
                bot_text = _format_approval(result.interrupt)
                PENDING_APPROVAL[session_id] = result.interrupt or {}
            else:
                bot_text = result.reply or "(no response)"

            yield _sse({"type": "writer", "draft": bot_text})
            transcript.append({"role": "assistant", "content": bot_text})
            yield _sse({"type": "bot", "content": bot_text})
            yield _sse(
                {"type": "state", "snapshot": _snapshot(session_id, active_sop=_last_worker_sop(new_messages))}
            )
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "content": str(e)})

        yield _sse({"type": "end"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/session/{session_id}/reset")
def reset(session_id: str) -> dict[str, str]:
    reset_session(session_id)
    TRANSCRIPTS.pop(session_id, None)
    PENDING_APPROVAL.pop(session_id, None)
    return {"status": "reset"}
