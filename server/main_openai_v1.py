"""FastAPI bridge for ``openai_agent_v1`` — the SAME web UI that v2/v4/v4_1 use.

It speaks the v4/v4_1 SSE/JSON contract, including the streaming additions:

  * ``{type:"token", content}``     — one per writer delta (live token streaming);
  * ``{type:"guardrail", stage, rule, action}`` — an input-guardrail trip;
  * ``{type:"trace", ...}``         — deep-trace frames (when the session's debug=True).

Like agent_v4_1 it is async end-to-end: ``run_turn_stream`` yields events as they
happen (the orchestrator runs in a background task, pushing live events onto a bus),
and the writer's tokens reach the client as the OpenAI Agents SDK generates them.
The existing frontend keeps working even without the optional token-rendering patch
(it ignores unknown event types).

Run:
    uvicorn server.main_openai_v1:app --reload --port 8002
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

from openai_agent_v1.shopping.session import ShoppingSession

app = FastAPI(title="openai_agent_v1 (streaming) debug API", version="0.1.0")
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
    return {"status": "ok", "engine": "openai_agent_v1"}


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
    """Run one turn; stream events (incl. live writer tokens) as SSE."""
    session = _get_or_create(session_id)
    lock = _LOCKS.setdefault(session_id, asyncio.Lock())

    async def event_stream() -> AsyncGenerator[str, None]:
        async with lock:  # one turn at a time per session
            try:
                async for ev in session.run_turn_stream(req.message):
                    yield _sse(ev)
            except Exception as e:  # noqa: BLE001  (run_turn_stream handles its own; belt-and-braces)
                yield _sse({"type": "error", "content": str(e)})
                yield _sse({"type": "end"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
