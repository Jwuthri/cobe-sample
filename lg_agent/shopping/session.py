"""``ShoppingSession`` — the per-session, streaming-first turn engine.

A turn is four phases:

  1. **input guardrails** (pre-flight, before any model call) — a refusal is
     instant; a redaction rewrites the user text before it enters the transcript;
  2. **orchestrator** — routes to sub-agent tools, streamed live (router / tool /
     step events as they happen);
  3. **writer** — the LAST model call, with nothing after it, so its tokens stream
     straight to the client (``{type:"token"}``), retried once if it emits nothing;
  4. **blocks + bot** — deterministic typed cards are attached and a final
     ``{type:"bot"}`` carries the authoritative text + blocks.

Because the writer is terminal and only ever sees verified step results + cart
(blocks are built deterministically), there is no post-generation validator gating
the stream — the grounding happened at construction.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from lg_agent.core import trace
from lg_agent.core.guardrails import CompiledInputRule, run_input_guardrails
from lg_agent.shopping import setup
from lg_agent.shopping.agents.orchestrator import CONFIG as ORCHESTRATOR_CONFIG
from lg_agent.shopping.agents.orchestrator import build_orchestrator
from lg_agent.shopping.agents.orchestrator.routing import absorb_recalls, build_memo
from lg_agent.shopping.agents.subagents import BLOCK_BY_SOP, SUBAGENTS
from lg_agent.shopping.agents.writer import CONFIG as WRITER_CONFIG
from lg_agent.shopping.agents.writer import build_blocks, build_writer, build_writer_payload
from lg_agent.shopping.context import ShoppingContext
from lg_agent.shopping.domain import CartService
from lg_agent.shopping.events import classify_custom, is_subagent_tool_end, step_event
from lg_agent.shopping.snapshot import build_snapshot

# How many transcript turns to show in the orchestrator_input trace (the
# orchestrator itself sees the full history; this only bounds the debug view).
_TRACE_HISTORY = 24

_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"


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
    """One conversation: clean transcript + live cart + the compiled agents."""

    user_id: str = "demo"
    session_id: str = "sess-lg"
    messages: list[BaseMessage] = field(default_factory=list)
    cart_service: CartService = field(default_factory=CartService)
    skills_loaded: list[str] = field(default_factory=list)
    turn: int = 0  # incremented each run_turn_stream — delimits turns in the UI
    # Persisted per-step ``recall`` snippets (keyed by sop), carried to the
    # orchestrator so it can resolve references WITHOUT the sub-agents seeing the
    # chat. Opaque to the session — the domain renders the content.
    routing_notes: dict[str, str] = field(default_factory=dict)

    # Injected for tests; built from the platform otherwise.
    orchestrator: Any = None
    writer: Any = None
    input_rules: list[CompiledInputRule] = field(default_factory=list)
    # If the writer has output-side guardrails, that turn buffers (no token stream).
    writer_buffered: bool = False
    # Emit deep-trace events (exact payloads between orchestrator/sub-agents/writer).
    debug: bool = True

    def __post_init__(self) -> None:
        if self.orchestrator is None:
            self.orchestrator = build_orchestrator(store=setup.store())
        if self.writer is None:
            self.writer = build_writer()

    # ----- snapshot (the frontend's AgentSnapshot) -----
    def snapshot(self) -> dict[str, Any]:
        return build_snapshot(
            user_id=self.user_id,
            session_id=self.session_id,
            cart=self.cart_service.cart,
            messages=self.messages,
            skills_loaded=self.skills_loaded,
        )

    # ----- the streaming turn -----
    async def run_turn_stream(self, user_text: str) -> AsyncGenerator[dict, None]:
        self.turn += 1
        yield {"type": "user", "content": user_text, "turn": self.turn}
        yield {"type": "state", "snapshot": self.snapshot()}

        # 1. input guardrails (pre-flight, before any model call)
        outcome = run_input_guardrails(self.input_rules, user_text)
        for hit in outcome.triggered:
            yield {"type": "guardrail", "stage": "input", "rule": hit.type, "action": hit.action}
        if not outcome.allowed:
            self.messages.append(HumanMessage(content=user_text))
            self.messages.append(AIMessage(content=outcome.refusal or ""))
            yield {"type": "bot", "content": outcome.refusal or "", "blocks": []}
            yield {"type": "state", "snapshot": self.snapshot()}
            yield {"type": "end"}
            return

        self.messages.append(HumanMessage(content=outcome.text))
        ctx = ShoppingContext(
            user_id=self.user_id,
            session_id=self.session_id,
            cart_service=self.cart_service,
            skills_loaded=list(self.skills_loaded),
            debug=self.debug,
        )

        # The orchestrator is the SOLE reader of the conversation. It also gets a
        # deterministic routing memo (cart + recently shown products) so it can
        # resolve references into concrete ids before delegating — the sub-agents
        # never see the transcript (see core/subagent.py "Context isolation").
        memo = build_memo(ctx, self.routing_notes)
        orch_messages: list[BaseMessage] = list(self.messages)
        if memo:
            orch_messages.insert(len(orch_messages) - 1, SystemMessage(content=memo))

        if self.debug:
            yield trace.frame(
                "orchestrator_input",
                "orchestrator",
                "user turn → orchestrator",
                {
                    "system_prompt": ORCHESTRATOR_CONFIG.get("system_prompt", ""),
                    "delegates": [s.name for s in SUBAGENTS],
                    "routing_memo": memo,
                    "messages_total": len(self.messages),
                    "conversation_seen": trace.render_messages(orch_messages[-_TRACE_HISTORY:]),
                    "context": ctx.debug_view(),
                },
            )

        # 2. orchestrator phase — live routing / tool / step events
        emitted = 0
        try:
            async for mode, payload in self.orchestrator.astream(
                {"messages": orch_messages}, context=ctx, stream_mode=["updates", "custom"]
            ):
                if mode == "custom":
                    for ev in classify_custom(payload):
                        yield ev
                    if is_subagent_tool_end(payload):
                        while emitted < len(ctx.step_results):
                            yield step_event(ctx.step_results[emitted])
                            emitted += 1
            while emitted < len(ctx.step_results):  # drain any tail
                yield step_event(ctx.step_results[emitted])
                emitted += 1
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "content": str(e)}
            yield {"type": "end"}
            return

        # persist this turn's recall snippets so the NEXT turn's orchestrator can
        # resolve references without the sub-agents ever seeing the chat
        absorb_recalls(ctx, self.routing_notes)
        self.skills_loaded = ctx.skills_loaded  # carry loaded skills into the snapshot
        yield {"type": "state", "snapshot": self.snapshot()}

        # 3. writer phase — the terminal model call, streamed token-by-token
        yield {"type": "router", "target": "writer", "iteration": 0}
        payload_json, mode = build_writer_payload(
            self.messages, ctx.step_results, self.cart_service.cart
        )

        if self.debug:
            try:
                parsed_payload = json.loads(payload_json)
            except json.JSONDecodeError:
                parsed_payload = {"raw": payload_json}
            yield trace.frame(
                "writer_payload",
                "writer",
                f"writer input (mode={mode})",
                {"system_prompt": WRITER_CONFIG["system_prompt"], "mode": mode, "payload": parsed_payload},
            )

        text = ""
        try:
            if self.writer_buffered:
                text = await self._invoke_writer(payload_json)
            else:
                async for ev in self._stream_writer(payload_json):
                    if ev["type"] == "token":
                        yield ev
                    elif ev["type"] == "_final":
                        text = ev["content"]
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "content": str(e)}
            yield {"type": "end"}
            return

        if not text:
            text = _EMPTY_WRITER_FALLBACK

        # 4. deterministic blocks + the authoritative bot reply
        blocks = build_blocks(ctx.step_results, self.cart_service.cart, BLOCK_BY_SOP)
        self.messages.append(
            AIMessage(content=text, additional_kwargs={"blocks": blocks} if blocks else {})
        )

        yield {"type": "writer", "draft": text, "blocks": blocks}
        yield {"type": "bot", "content": text, "blocks": blocks}
        yield {"type": "state", "snapshot": self.snapshot()}
        yield {"type": "end"}

    async def _stream_writer(self, payload_json: str) -> AsyncGenerator[dict, None]:
        """Stream writer tokens, retrying once if the first attempt is empty.

        Empty-retry is stream-safe: an empty stream sent zero tokens, so the retry
        is invisible to the client.
        """
        for _attempt in (1, 2):
            parts: list[str] = []
            async for chunk, meta in self.writer.astream(
                {"messages": [HumanMessage(content=payload_json)]}, stream_mode="messages"
            ):
                if not isinstance(chunk, AIMessageChunk) or meta.get("langgraph_node") != "model":
                    continue
                t = _chunk_text(chunk)
                if t:
                    parts.append(t)
                    yield {"type": "token", "content": t}
            text = "".join(parts).strip()
            if text:
                yield {"type": "_final", "content": text}
                return
        yield {"type": "_final", "content": ""}

    async def _invoke_writer(self, payload_json: str) -> str:
        """Buffered writer (used when output-side guardrails are configured)."""
        result = await self.writer.ainvoke({"messages": [HumanMessage(content=payload_json)]})
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
                return m.content.strip()
        return ""

    # ----- sync convenience (tests / non-streaming callers) -----
    def run_turn(self, user_text: str) -> dict[str, Any]:
        """Run a turn to completion, returning a summary of the collected events."""

        async def _collect() -> list[dict]:
            return [ev async for ev in self.run_turn_stream(user_text)]

        events = asyncio.run(_collect())
        tokens = [e["content"] for e in events if e["type"] == "token"]
        bot = next((e for e in reversed(events) if e["type"] == "bot"), None)
        return {
            "events": events,
            "message": bot["content"] if bot else "",
            "blocks": bot["blocks"] if bot else [],
            "tokens": tokens,
        }


__all__ = ["ShoppingSession"]
