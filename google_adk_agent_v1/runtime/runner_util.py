"""Thin helpers for *running* an ADK agent — the moving part Pydantic AI hid behind
``agent.run`` / ``agent.run_stream``.

Google ADK runs an agent through a :class:`~google.adk.runners.Runner` bound to a
:class:`~google.adk.sessions.BaseSessionService`. We keep ONE in-memory session
service and one cached Runner per agent, and give each *call* its own throwaway
session id. Two consequences fall straight out of that:

* **Worker isolation** — a delegated worker runs in a brand-new session whose only
  message is the orchestrator's ``query``. It never sees the conversation.
* **A clean orchestrator history** — the orchestrator's session is seeded each turn
  with text-only history events (no tool-call noise), mirroring the Pydantic build's
  curated ``message_history``.

The shared live state is *not* in session state (ADK deep-copies that); only a string
key is, which every tool resolves back through :mod:`registry`.
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from google_adk_agent_v1.runtime.registry import RUNTIME_KEY

USER_ID = "demo"

_SESSIONS = InMemorySessionService()
_RUNNERS: dict[int, Runner] = {}


def _runner(agent) -> Runner:
    """One cached Runner per agent (reused across turns; picks up model overrides)."""
    cached = _RUNNERS.get(id(agent))
    if cached is None:
        cached = Runner(app_name=f"adk-{agent.name}", agent=agent, session_service=_SESSIONS)
        _RUNNERS[id(agent)] = cached
    return cached


def message(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def event_text(content: types.Content | None) -> str:
    """Concatenate the plain-text parts of a content (ignores function-call parts)."""
    if not content or not content.parts:
        return ""
    return "".join(p.text or "" for p in content.parts if getattr(p, "text", None))


def history_events(agent_name: str, transcript: list[dict]) -> list[Event]:
    """Render a clean transcript into ADK history events for one agent's session.

    User turns are authored ``"user"`` (→ user role); assistant turns are authored as
    the agent itself (→ model role) so the model reads them as its own prior replies.
    Tool-call noise never enters here — this is the curated text-only history.
    """
    events: list[Event] = []
    for m in transcript:
        content = str(m.get("content", ""))
        if m.get("role") == "user":
            events.append(Event(author="user", content=types.Content(role="user", parts=[types.Part(text=content)])))
        elif m.get("role") == "assistant" and content.strip():
            events.append(
                Event(author=agent_name, content=types.Content(role="model", parts=[types.Part(text=content)]))
            )
    return events


async def _new_session(runner: Runner, runtime_key: str | None, history: list[Event] | None):
    sid = uuid.uuid4().hex
    state = {RUNTIME_KEY: runtime_key} if runtime_key else {}
    session = await _SESSIONS.create_session(
        app_name=runner.app_name, user_id=USER_ID, session_id=sid, state=state
    )
    for ev in history or []:
        await _SESSIONS.append_event(session, ev)
    return sid


async def run_collect(
    agent, *, text: str, runtime_key: str | None = None, history: list[Event] | None = None
) -> list[Event]:
    """Run ``agent`` on a single message to completion; return every event it emitted."""
    runner = _runner(agent)
    sid = await _new_session(runner, runtime_key, history)
    out: list[Event] = []
    async for ev in runner.run_async(user_id=USER_ID, session_id=sid, new_message=message(text)):
        out.append(ev)
    return out


async def run_stream(
    agent, *, text: str, runtime_key: str | None = None, history: list[Event] | None = None
) -> AsyncGenerator[Event, None]:
    """Run ``agent`` in SSE mode, yielding events live (partial text deltas + final)."""
    runner = _runner(agent)
    sid = await _new_session(runner, runtime_key, history)
    async for ev in runner.run_async(
        user_id=USER_ID,
        session_id=sid,
        new_message=message(text),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        yield ev
