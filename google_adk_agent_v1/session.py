"""``ShoppingSession`` — the per-conversation, streaming-first turn engine.

A turn is four phases:

  1. **input guardrails** (pre-flight, before any model call) — a refusal is instant;
     a redaction rewrites the user text before it enters the transcript;
  2. **orchestrator** — routes to worker tools, run in a background task while its
     router / tool / step / trace events stream to the client live (via an event bus);
  3. **writer** — the LAST model call, with nothing after it, so its tokens stream
     straight to the client (``{type:"token"}``), retried once if it emits nothing;
  4. **blocks + bot** — deterministic typed cards are attached and a final
     ``{type:"bot"}`` carries the authoritative text + blocks.

Because the writer is terminal and only ever sees verified step results + cart (the
cards are built deterministically), there is no post-generation validator gating the
stream — the grounding happened at construction.

ADK specifics: the shared :class:`ShoppingDeps` is published to a process-level
registry for the turn (tools resolve it from a string key in session state, since ADK
deep-copies state); the orchestrator runs against a freshly-seeded, text-only history;
the writer streams via ADK's SSE mode (partial-text deltas).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from google_adk_agent_v1.agents import (
    BLOCK_BY_SOP,
    ROUTER_PROMPT,
    WORKERS,
    WRITER_SYSTEM,
    absorb_recalls,
    build_blocks,
    build_memo,
    build_writer_payload,
    orchestrator,
    writer,
)
from google_adk_agent_v1.domain import CartService, MemoryStore
from google_adk_agent_v1.runtime import ShoppingDeps
from google_adk_agent_v1.runtime import events, registry, trace
from google_adk_agent_v1.runtime.guardrails import InputRule, run_input_guardrails
from google_adk_agent_v1.runtime.runner_util import event_text, history_events, run_collect, run_stream
from google_adk_agent_v1.snapshot import build_snapshot

_TRACE_HISTORY = 24  # how many transcript turns to show in the orchestrator_input trace
_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"
_DONE = object()  # bus sentinel: the orchestrator phase has finished


@dataclass
class ShoppingSession:
    """One conversation: a clean transcript + a live cart + the shared agents."""

    user_id: str = "demo"
    session_id: str = "sess-adk"
    cart_service: CartService = field(default_factory=CartService)
    store: MemoryStore = field(default_factory=MemoryStore)
    # the clean transcript: [{role: "user"|"assistant", content: str, blocks: [...]}]
    transcript: list[dict] = field(default_factory=list)
    # the orchestrator's cross-turn memory (per-step recalls), keyed by worker name.
    routing_notes: dict[str, str] = field(default_factory=dict)
    turn: int = 0
    input_rules: list[InputRule] = field(default_factory=list)
    debug: bool = True

    # injectable for tests (default to the module-level singletons)
    orchestrator_agent: Any = field(default_factory=lambda: orchestrator)
    writer_agent: Any = field(default_factory=lambda: writer)

    # ----- snapshot -----
    def snapshot(self) -> dict[str, Any]:
        return build_snapshot(
            user_id=self.user_id,
            session_id=self.session_id,
            cart=self.cart_service.cart,
            transcript=self.transcript,
        )

    # ----- the streaming turn -----
    async def run_turn_stream(self, user_text: str) -> AsyncGenerator[dict, None]:
        self.turn += 1
        yield events.user(user_text, self.turn)
        yield events.state(self.snapshot())

        # 1. input guardrails (pre-flight, before any model call)
        outcome = run_input_guardrails(self.input_rules, user_text)
        for hit in outcome.triggered:
            yield events.guardrail("input", hit.type, hit.action)
        if not outcome.allowed:
            self.transcript.append({"role": "user", "content": user_text})
            self.transcript.append({"role": "assistant", "content": outcome.refusal or "", "blocks": []})
            yield events.bot(outcome.refusal or "", [])
            yield events.state(self.snapshot())
            yield events.end()
            return

        # The new user turn enters the transcript; prior turns become the
        # orchestrator's message history (the workers never see any of it).
        prior = list(self.transcript)
        self.transcript.append({"role": "user", "content": outcome.text})

        bus: asyncio.Queue = asyncio.Queue()
        deps = ShoppingDeps(
            cart_service=self.cart_service,
            store=self.store,
            user_id=self.user_id,
            bus=bus,
            steps=[],
            routing_notes=self.routing_notes,
            debug=self.debug,
        )

        # Publish the live deps for this turn so the agents' tools (which only carry a
        # string key through ADK session state) can resolve it.
        runtime_key = f"{self.session_id}:{self.turn}:{uuid.uuid4().hex[:8]}"
        registry.register(runtime_key, deps)
        try:
            if self.debug:
                yield trace.frame(
                    "orchestrator_input",
                    "orchestrator",
                    "user turn → orchestrator",
                    {
                        "system_prompt": ROUTER_PROMPT,
                        "delegates": [w.name for w in WORKERS],
                        "routing_memo": build_memo(deps),
                        "conversation_seen": _render_transcript(prior[-_TRACE_HISTORY:] + [{"role": "user", "content": outcome.text}]),
                        "context": deps.debug_view(),
                    },
                )

            # 2. orchestrator phase — run in the background, stream its events live
            history = history_events(self.orchestrator_agent.name, prior)
            task = asyncio.create_task(
                self._run_orchestrator(outcome.text, runtime_key, history, bus)
            )
            while True:
                ev = await bus.get()
                if ev is _DONE:
                    break
                yield ev
            await task  # surface any non-event exception

            absorb_recalls(deps)  # remember this turn's recalls for the next one
            yield events.state(self.snapshot())

            # 3. writer phase — the terminal model call, streamed token by token
            yield events.router("writer", 0)
            payload_json, mode = build_writer_payload(self.transcript, deps.steps, self.cart_service.cart)
            if self.debug:
                yield trace.frame(
                    "writer_payload",
                    "writer",
                    f"writer input (mode={mode})",
                    {"system_prompt": WRITER_SYSTEM, "mode": mode, "payload": json.loads(payload_json)},
                )

            text = ""
            try:
                async for ev in self._stream_writer(payload_json):
                    if ev["type"] == "token":
                        yield ev
                    elif ev["type"] == "_final":
                        text = ev["content"]
            except Exception as e:  # noqa: BLE001
                yield events.error(str(e))
                yield events.end()
                return
            text = text or _EMPTY_WRITER_FALLBACK

            # 4. deterministic blocks + the authoritative bot reply
            blocks = build_blocks(deps.steps, self.cart_service.cart, BLOCK_BY_SOP)
            self.transcript.append({"role": "assistant", "content": text, "blocks": blocks})
            yield events.writer(text, blocks)
            yield events.bot(text, blocks)
            yield events.state(self.snapshot())
            yield events.end()
        finally:
            registry.unregister(runtime_key)

    # ----- phase helpers -----
    async def _run_orchestrator(
        self, user_text: str, runtime_key: str, history: list, bus: asyncio.Queue
    ) -> None:
        """Run the router; its tool wrappers push UI events onto the bus as they go."""
        try:
            await run_collect(
                self.orchestrator_agent, text=user_text, runtime_key=runtime_key, history=history
            )
        except Exception as e:  # noqa: BLE001
            bus.put_nowait(events.error(str(e)))
        finally:
            bus.put_nowait(_DONE)

    async def _stream_writer(self, payload_json: str) -> AsyncGenerator[dict, None]:
        """Stream writer tokens, retrying once if the first attempt is empty.

        ADK SSE mode yields partial-text events (incremental deltas, emitted as
        ``token`` events) followed by a final aggregated event (the authoritative
        text, NOT re-emitted). Empty-retry is stream-safe: an empty stream sent zero
        tokens, so the retry is invisible to the client.
        """
        for _attempt in (1, 2):
            parts: list[str] = []
            final = ""
            async for ev in run_stream(self.writer_agent, text=payload_json):
                content = getattr(ev, "content", None)
                if not content or not content.parts:
                    continue
                if any(getattr(p, "function_call", None) for p in content.parts):
                    continue  # never display function-call streaming chunks
                chunk = event_text(content)
                if not chunk:
                    continue
                if ev.partial:
                    parts.append(chunk)
                    yield events.token(chunk)
                else:
                    final = chunk  # the aggregated, complete text
            text = (final or "".join(parts)).strip()
            if text:
                yield {"type": "_final", "content": text}
                return
        yield {"type": "_final", "content": ""}

    # ----- sync convenience (tests / non-streaming callers) -----
    def run_turn(self, user_text: str) -> dict[str, Any]:
        """Run a turn to completion, returning a summary of the collected events."""

        async def _collect() -> list[dict]:
            return [ev async for ev in self.run_turn_stream(user_text)]

        evs = asyncio.run(_collect())
        tokens = [e["content"] for e in evs if e["type"] == "token"]
        bot = next((e for e in reversed(evs) if e["type"] == "bot"), None)
        return {
            "events": evs,
            "message": bot["content"] if bot else "",
            "blocks": bot["blocks"] if bot else [],
            "tokens": tokens,
        }


# --------------------------------------------------------------------------- #
# transcript rendering for the debug trace
# --------------------------------------------------------------------------- #
def _render_transcript(transcript: list[dict]) -> list[dict]:
    return [{"role": m.get("role"), "content": str(m.get("content", ""))} for m in transcript]


__all__ = ["ShoppingSession"]
