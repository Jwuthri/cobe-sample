"""``ShoppingSession`` — the per-conversation, streaming-first turn engine.

A turn is four phases:

  1. **input guardrails** (pre-flight, before any model call) — a refusal is instant;
     a redaction rewrites the user text before it enters the transcript;
  2. **orchestrator** — ``Runner.run_streamed`` routes to worker tools; the SDK's
     ``stream_events()`` surfaces ``tool_called`` / ``tool_output`` for each
     delegate call, and each worker's ``custom_output_extractor`` stashes the
     INNER tool calls + the ``StepResult`` on ``ctx.pending_events`` for the
     session to drain inline. **No event bus, no background task.**
  3. **writer** — the LAST model call, with nothing after it, so its raw text
     deltas (``ResponseTextDeltaEvent``) become ``{type:"token"}`` events
     straight to the client; retried once if it emits nothing;
  4. **blocks + bot** — deterministic typed cards are attached and a final
     ``{type:"bot"}`` carries the authoritative text + blocks.

Because the writer is terminal and only ever sees verified step results + cart (the
cards are built deterministically), there is no post-generation validator gating the
stream — the grounding happened at construction.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from agents import Runner
from agents.items import MessageOutputItem
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import ResponseTextDeltaEvent

from agent_openai_sdk_v1.agents import (
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
from agent_openai_sdk_v1.domain import CartService, MemoryStore
from agent_openai_sdk_v1.runtime import ShoppingContext, events, trace
from agent_openai_sdk_v1.runtime.guardrails import InputRule, run_input_guardrails
from agent_openai_sdk_v1.snapshot import build_snapshot

_TRACE_HISTORY = 24  # how many transcript turns to show in the orchestrator_input trace
_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"
_WORKER_NAMES = {w.name for w in WORKERS}


@dataclass
class ShoppingSession:
    """One conversation: a clean transcript + a live cart + the shared agents."""

    user_id: str = "demo"
    session_id: str = "sess-oai-sdk"
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

        # The new user turn enters the transcript.
        prior = list(self.transcript)
        self.transcript.append({"role": "user", "content": outcome.text})

        ctx = ShoppingContext(
            cart_service=self.cart_service,
            store=self.store,
            user_id=self.user_id,
            steps=[],
            routing_notes=self.routing_notes,
            debug=self.debug,
            pending_events=[],
        )

        if self.debug:
            yield trace.frame(
                "orchestrator_input",
                "orchestrator",
                "user turn → orchestrator",
                {
                    "system_prompt": ROUTER_PROMPT,
                    "delegates": [w.name for w in WORKERS],
                    "routing_memo": build_memo(ctx),
                    "conversation_seen": _render_transcript(prior[-_TRACE_HISTORY:] + [{"role": "user", "content": outcome.text}]),
                    "context": ctx.debug_view(),
                },
            )

        # 2. orchestrator phase — stream the SDK's events, drain pending events inline
        orchestrator_input = _to_input_items(prior, outcome.text)
        try:
            result = Runner.run_streamed(
                starting_agent=self.orchestrator_agent,
                input=orchestrator_input,
                context=ctx,
            )
            async for ev in result.stream_events():
                async for out in self._on_orchestrator_event(ev, ctx):
                    yield out
            # final drain (in case the last event was a tool_output → still in queue)
            for queued in ctx.pending_events:
                yield queued
            ctx.pending_events.clear()
        except Exception as e:  # noqa: BLE001
            yield events.error(str(e))
            yield events.end()
            return

        absorb_recalls(ctx)
        yield events.state(self.snapshot())

        # 3. writer phase — terminal model call, streamed token by token
        yield events.router("writer", 0)
        payload_json, mode = build_writer_payload(self.transcript, ctx.steps, self.cart_service.cart)
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
        blocks = build_blocks(ctx.steps, self.cart_service.cart, BLOCK_BY_SOP)
        self.transcript.append({"role": "assistant", "content": text, "blocks": blocks})
        yield events.writer(text, blocks)
        yield events.bot(text, blocks)
        yield events.state(self.snapshot())
        yield events.end()

    # ----- orchestrator stream → wire events -----
    async def _on_orchestrator_event(
        self, ev: Any, ctx: ShoppingContext
    ) -> AsyncGenerator[dict, None]:
        """Map ONE SDK ``StreamEvent`` to zero-or-more wire events.

        Worker delegations surface here as ``tool_called`` / ``tool_output`` items
        at the orchestrator level. We translate them:

          * ``tool_called`` (worker) → ``router(worker_name)``;
          * ``tool_output`` (worker) → drain ``ctx.pending_events`` (inner
            tool_start/tool_end pairs + ``step`` event the extractor stashed).

        Non-worker tool calls at the orchestrator level (none in this build) are
        forwarded as plain tool_start/tool_end.
        """
        if not isinstance(ev, RunItemStreamEvent):
            return
        item = ev.item
        if ev.name == "tool_called":
            from agents.items import ToolCallItem

            if isinstance(item, ToolCallItem):
                name = item.tool_name or ""
                if name in _WORKER_NAMES:
                    yield events.router(name)
                # non-worker tool: ignore (the inner ones are drained on tool_output)
        elif ev.name == "tool_output":
            # The worker's run completed and the extractor populated pending_events.
            for queued in ctx.pending_events:
                yield queued
            ctx.pending_events.clear()
            if self.debug:
                yield trace.frame(
                    "context", "orchestrator", "context after worker", ctx.debug_view()
                )

    # ----- writer streaming -----
    async def _stream_writer(self, payload_json: str) -> AsyncGenerator[dict, None]:
        """Stream writer tokens, retrying once if the first attempt is empty.

        Empty-retry is stream-safe: an empty stream sent zero tokens, so the retry
        is invisible to the client.
        """
        for _attempt in (1, 2):
            parts: list[str] = []
            result = Runner.run_streamed(starting_agent=self.writer_agent, input=payload_json)
            async for ev in result.stream_events():
                if isinstance(ev, RawResponsesStreamEvent) and isinstance(
                    ev.data, ResponseTextDeltaEvent
                ):
                    delta = ev.data.delta or ""
                    if delta:
                        parts.append(delta)
                        yield events.token(delta)
                elif isinstance(ev, RunItemStreamEvent) and isinstance(ev.item, MessageOutputItem):
                    # Capture the final completed text in case the deltas missed any.
                    from agents.items import ItemHelpers

                    final_text = ItemHelpers.text_message_output(ev.item)
                    if not parts and final_text:
                        # No deltas arrived; fall back to the message text. Emit it as
                        # one token so the client at least sees text on the wire.
                        parts.append(final_text)
                        yield events.token(final_text)
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
# transcript → SDK input items (clean text only; no tool-call noise)
# --------------------------------------------------------------------------- #
def _to_input_items(prior: list[dict], current_user: str) -> list[dict]:
    """Render the prior transcript + the current user turn as SDK input items.

    The SDK accepts a list of message dicts ``{"role": "user"|"assistant",
    "content": str}`` (same shape as the Responses API). We strip the assistant's
    block payloads and drop empty messages so the orchestrator's history is the
    clean conversation only.
    """
    items: list[dict] = []
    for m in prior:
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        role = m.get("role")
        if role == "user":
            items.append({"role": "user", "content": content})
        elif role == "assistant":
            items.append({"role": "assistant", "content": content})
    items.append({"role": "user", "content": current_user})
    return items


def _render_transcript(transcript: list[dict]) -> list[dict]:
    return [{"role": m.get("role"), "content": str(m.get("content", ""))} for m in transcript]


__all__ = ["ShoppingSession"]
