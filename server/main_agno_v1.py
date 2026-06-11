"""FastAPI bridge for ``agent_agno_v1`` — the SAME web UI that v2/v4/v5 use.

It speaks the v4/v5 SSE/JSON contract, plus the two streaming additions:

  * ``{type:"token", content}``     — one per leader delta (live token streaming);
  * ``{type:"guardrail", stage, rule, action}`` — an input-guardrail trip.

``run_turn_stream`` is async end-to-end: it drives the Agno coordinate-mode team
with ``arun(stream=True)`` and yields events as they happen, so the supervisor's
reply tokens reach the client as it generates them. The existing frontend keeps
working unchanged (it ignores unknown event types).

Run:
    uvicorn server.main_agno_v1:app --reload --port 8001
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_agno_v1.shopping.session import ShoppingSession

app = FastAPI(title="agent_agno_v1 (Agno streaming) debug API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict[str, ShoppingSession] = {}
_LOCKS: dict[str, asyncio.Lock] = {}


class NewSessionResponse(BaseModel):
    session_id: str


class TurnRequest(BaseModel):
    message: str


def _get_or_create(session_id: str) -> ShoppingSession:
    session = SESSIONS.get(session_id)
    if session is None:
        session = ShoppingSession(user_id="demo", session_id=session_id)
        SESSIONS[session_id] = session
        _LOCKS[session_id] = asyncio.Lock()
    return session


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "agent_agno_v1"}


@app.post("/api/session", response_model=NewSessionResponse)
def new_session() -> NewSessionResponse:
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    SESSIONS[sid] = ShoppingSession(user_id="demo", session_id=sid)
    _LOCKS[sid] = asyncio.Lock()
    return NewSessionResponse(session_id=sid)


@app.get("/api/state/{session_id}")
def get_state(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session: {session_id}")
    return session.snapshot()


@app.post("/api/turn/{session_id}")
async def turn(session_id: str, req: TurnRequest) -> StreamingResponse:
    """Run one turn; stream events (incl. live leader tokens) as SSE."""
    session = _get_or_create(session_id)
    lock = _LOCKS.setdefault(session_id, asyncio.Lock())

    async def event_stream() -> AsyncGenerator[str, None]:
        async with lock:  # one turn at a time per session
            try:
                async for ev in session.run_turn_stream(req.message):
                    yield _sse(ev)
            except Exception as e:  # noqa: BLE001
                yield _sse({"type": "error", "content": str(e)})
                yield _sse({"type": "end"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
