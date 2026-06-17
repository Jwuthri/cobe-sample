"""FastAPI bridge for ``lg_agent_v3`` — speaks the same SSE/JSON contract as the other
engines, so the existing web UI works unchanged.

Endpoints:
  * ``POST /api/session``      — create a session, returns ``{session_id}``;
  * ``GET  /api/state/{id}``   — the current ``AgentSnapshot`` (cart + messages);
  * ``POST /api/turn/{id}``    — run one turn; stream events (incl. live writer
                                 tokens) as Server-Sent Events.

Run:
    uvicorn server.main_lg_agent_v3:app --reload --port 8007

Point the web UI at it with ``AGENT_V2_API_URL=http://localhost:8007`` (or run it on
whatever port your frontend already targets).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lg_agent.core.event_store import SQLiteEventStore, now_iso
from lg_agent_v3 import ShoppingSession

app = FastAPI(title="lg_agent_v3 (streaming) debug API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Tee every turn's events + snapshots to SQLite (override path with LG_AGENT_V3_EVENTS_DB).
EVENTS = SQLiteEventStore(os.environ.get("LG_AGENT_V3_EVENTS_DB", "lg_agent_v3_events.db"))

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
    return {"status": "ok", "engine": "lg_agent_v3"}


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


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    """All stored sessions (for the 'load previous session' picker); ``live`` = in memory."""
    rows = EVENTS.list_sessions()
    for r in rows:
        r["live"] = r["session_id"] in SESSIONS
    return {"sessions": rows}


@app.get("/api/events/{session_id}")
def get_events(session_id: str) -> dict[str, Any]:
    """Every persisted event in order (``data`` = the original event) — replayed by the UI."""
    return {"session_id": session_id, "events": EVENTS.read_events(session_id)}


@app.get("/api/snapshots/{session_id}")
def get_snapshots(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "snapshots": EVENTS.read_snapshots(session_id)}


@app.post("/api/turn/{session_id}")
async def turn(session_id: str, req: TurnRequest) -> StreamingResponse:
    """Run one turn; stream events (incl. live writer tokens) as SSE, teeing to SQLite."""
    session = _get_or_create(session_id)
    lock = _LOCKS.setdefault(session_id, asyncio.Lock())

    async def event_stream() -> AsyncGenerator[str, None]:
        async with lock:  # one turn at a time per session
            rows: list[tuple[str, dict]] = []
            try:
                async for ev in session.run_turn_stream(req.message):
                    rows.append((now_iso(), ev))
                    yield _sse(ev)
            except Exception as e:  # noqa: BLE001  (belt-and-braces; the session handles its own)
                for extra in ({"type": "error", "content": str(e)}, {"type": "end"}):
                    rows.append((now_iso(), extra))
                    yield _sse(extra)
            try:  # persist after the stream; never let logging break the turn
                await asyncio.to_thread(
                    EVENTS.record_turn, session_id, session.user_id, session.turn, rows
                )
            except Exception:  # noqa: BLE001
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
