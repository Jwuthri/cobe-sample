"""``ShoppingSession`` — the per-conversation, streaming-first turn engine.

A turn is four phases:

  1. **input guardrails** (pre-flight, before any model call) — a refusal is instant; a
     redaction rewrites the user text before it enters the transcript;
  2. **orchestrator** — routes to worker tools, run in a background task while its
     router / tool / step / trace events stream to the client live (via an event bus);
  3. **writer** — the LAST model call, with nothing after it, so its tokens stream
     straight to the client (``{type:"token"}``), retried once if it emits nothing;
  4. **blocks + bot** — deterministic typed cards are attached and a final
     ``{type:"bot"}`` carries the authoritative text + blocks.

Because the writer is terminal and only ever sees verified step results + cart (the
cards are built deterministically), there is no post-generation validator gating the
stream — the grounding happened at construction.

This is a line-for-line port of ``pydantic_agent_v1.session``: the bus + background
task + four phases are identical. Only the model calls change — the orchestrator runs
as a compiled LangChain graph via ``ainvoke(..., context=deps)`` (its tools push events
onto the bus), and the writer streams via ``astream(stream_mode="messages")``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from lg_agent_v2.agents import (
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
from lg_agent_v2.domain import CartService, MemoryStore
from lg_agent_v2.runtime import ShoppingDeps
from lg_agent_v2.runtime import events, trace
from lg_agent_v2.runtime.guardrails import InputRule, run_input_guardrails
from lg_agent_v2.snapshot import build_snapshot

_TRACE_HISTORY = 24  # how many transcript turns to show in the orchestrator_input trace
_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"
_DONE = object()  # bus sentinel: the orchestrator phase has finished


def _chunk_text(chunk: Any) -> str:
    """Extract the text delta from a streamed message chunk (str or list content)."""
    if not isinstance(chunk, AIMessageChunk):
        return ""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


@dataclass
class ShoppingSession:
    """One conversation: a clean transcript + a live cart + the shared agents."""

    user_id: str = "demo"
    session_id: str = "sess-lgv2"
    cart_service: CartService = field(default_factory=CartService)
    store: MemoryStore = field(default_factory=MemoryStore)
    # the clean transcript: [{role: "user"|"assistant", content: str, blocks: [...]}]
    transcript: list[dict] = field(default_factory=list)
    # the orchestrator's cross-turn memory (per-step recalls), keyed by worker name.
    routing_notes: dict[str, str] = field(default_factory=dict)
    turn: int = 0
    input_rules: list[InputRule] = field(default_factory=list)
    debug: bool = True

    # injectable for tests (default to the module-level compiled graphs)
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
        messages = _to_message_history(prior) + [HumanMessage(content=outcome.text)]
        task = asyncio.create_task(self._run_orchestrator(deps, messages, bus))
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

    # ----- phase helpers -----
    async def _run_orchestrator(
        self, deps: ShoppingDeps, messages: list[BaseMessage], bus: asyncio.Queue
    ) -> None:
        """Run the router; its tool wrappers push UI events onto the bus as they go."""
        try:
            await self.orchestrator_agent.ainvoke({"messages": messages}, context=deps)
        except Exception as e:  # noqa: BLE001
            bus.put_nowait(events.error(str(e)))
        finally:
            bus.put_nowait(_DONE)

    async def _stream_writer(self, payload_json: str) -> AsyncGenerator[dict, None]:
        """Stream writer tokens, retrying once if the first attempt is empty.

        Empty-retry is stream-safe: an empty stream sent zero tokens, so the retry is
        invisible to the client.
        """
        for _attempt in (1, 2):
            parts: list[str] = []
            async for chunk, meta in self.writer_agent.astream(
                {"messages": [HumanMessage(content=payload_json)]}, stream_mode="messages"
            ):
                if not isinstance(chunk, AIMessageChunk) or meta.get("langgraph_node") != "model":
                    continue
                t = _chunk_text(chunk)
                if t:
                    parts.append(t)
                    yield events.token(t)
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
# transcript → LangChain message history (clean text only; no tool-call noise)
# --------------------------------------------------------------------------- #
def _to_message_history(transcript: list[dict]) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for m in transcript:
        content = str(m.get("content", ""))
        if m.get("role") == "user":
            msgs.append(HumanMessage(content=content))
        elif m.get("role") == "assistant" and content.strip():
            msgs.append(AIMessage(content=content))
    return msgs


def _render_transcript(transcript: list[dict]) -> list[dict]:
    return [{"role": m.get("role"), "content": str(m.get("content", ""))} for m in transcript]


__all__ = ["ShoppingSession"]
