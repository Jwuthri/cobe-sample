"""``ShoppingSession`` — the per-conversation, streaming-first turn engine.

A turn is ONE ``StateGraph`` (see :mod:`lg_agent_v3.graph`): ``orchestrator → payload →
writer → blocks``. The session:

  1. **redacts** the user text (session-level ``pii`` rules) BEFORE it enters the
     transcript — the only guardrail concern that must live at the source (so raw PII
     never hits storage or the workers). Blocking guardrails live ON the agents
     (``before_agent``/``after_agent`` middleware), not here;
  2. drives the turn-graph with ``astream(stream_mode=["custom","messages"],
     subgraphs=True)`` — UI events (router / tool / step / bot / trace) raised anywhere
     via ``deps.emit`` (``get_stream_writer``) propagate up the **custom** channel from
     any depth; the **writer** node's model tokens surface on the **messages** channel
     and are forwarded (filtered by node namespace). No bus, no re-pump;
  3. appends the final reply to the transcript and emits the closing state + end.

Guardrails are owned per-agent: a block on the orchestrator routes the turn to a
verbatim refusal (the writer node delivers it); a block on a sub-agent comes back as a
flagged guardrail step and is delivered the same way.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from lg_agent_v3.agents import ROUTER_PROMPT, WORKERS, build_memo, orchestrator, writer
from lg_agent_v3.domain import CartService, MemoryStore
from lg_agent_v3.graph import build_turn_graph
from lg_agent_v3.runtime import ShoppingDeps
from lg_agent_v3.runtime import events, trace
from lg_agent_v3.runtime.guardrails import GuardrailSpec, redact_input
from lg_agent_v3.snapshot import build_snapshot

_TRACE_HISTORY = 24  # how many transcript turns to show in the orchestrator_input trace
_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"


@dataclass
class ShoppingSession:
    """One conversation: a clean transcript + a live cart + the turn-graph."""

    user_id: str = "demo"
    session_id: str = "sess-lgv3"
    cart_service: CartService = field(default_factory=CartService)
    store: MemoryStore = field(default_factory=MemoryStore)
    # the clean transcript: [{role: "user"|"assistant", content: str, blocks: [...]}]
    transcript: list[dict] = field(default_factory=list)
    # the orchestrator's cross-turn memory (per-step recalls), keyed by worker name.
    routing_notes: dict[str, str] = field(default_factory=dict)
    turn: int = 0
    debug: bool = True

    # guardrails: session-level pii REDACT rules (input is cleaned before the transcript);
    # blocking rules live on the agents, configured at build time. ``writer_buffered`` is
    # set when the writer carries an on_output guardrail (no token stream that turn).
    input_redact_rules: list[GuardrailSpec] = field(default_factory=list)
    writer_buffered: bool = False

    # injectable for tests (default to the module-level compiled graphs)
    orchestrator_agent: Any = field(default_factory=lambda: orchestrator)
    writer_agent: Any = field(default_factory=lambda: writer)
    _graph: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        # writer_buffered is handled in the session's token filtering, not the graph.
        self._graph = build_turn_graph(self.orchestrator_agent, self.writer_agent)

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

        # 1. session-level input redaction (pii) — clean BEFORE the text enters the
        #    transcript, so the raw value never hits storage or the workers.
        text, redact_hits = redact_input(self.input_redact_rules, user_text)
        for hit in redact_hits:
            yield events.guardrail("session:input", hit.type, hit.action)

        prior = list(self.transcript)
        self.transcript.append({"role": "user", "content": text})

        deps = ShoppingDeps(
            cart_service=self.cart_service,
            store=self.store,
            user_id=self.user_id,
            session_id=self.session_id,
            transcript=self.transcript,
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
                    "conversation_seen": _render_transcript(prior[-_TRACE_HISTORY:] + [{"role": "user", "content": text}]),
                    "context": deps.debug_view(),
                },
            )

        # 2. drive the turn-graph and stream its events live.
        #    - custom channel: every UI event (router/tool/step/token/bot/trace) raised
        #      via deps.emit(get_stream_writer), propagated from any depth (subgraphs=True);
        #    - messages channel: the WRITER node's model tokens (identified by namespace).
        state = {"messages": _to_message_history(prior) + [HumanMessage(content=text)]}
        final_text, final_blocks = "", []
        try:
            async for ns, mode, payload in self._graph.astream(
                state, context=deps, stream_mode=["custom", "messages"], subgraphs=True
            ):
                if mode == "custom":
                    ev = payload
                    if ev.get("type") == "bot":  # capture the authoritative reply for the transcript
                        final_text, final_blocks = ev.get("content", ""), ev.get("blocks", [])
                    if ev.get("type") == "token" and self.writer_buffered:
                        continue  # buffered (writer has an output guardrail) → suppress tokens
                    yield ev
                elif mode == "messages" and not self.writer_buffered:
                    chunk, meta = payload
                    if _is_writer_node(ns) and isinstance(chunk, AIMessageChunk):
                        t = _chunk_text(chunk)
                        if t:
                            yield events.token(t)
        except Exception as e:  # noqa: BLE001
            yield events.error(str(e))
            yield events.end()
            return

        # 3. finalize: append the reply, emit the closing state
        text_out = final_text or _EMPTY_WRITER_FALLBACK
        self.transcript.append({"role": "assistant", "content": text_out, "blocks": final_blocks})
        yield events.state(self.snapshot())
        yield events.end()

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


# --------------------------------------------------------------------------- #
# stream filtering: only the writer node's tokens reach the user
# --------------------------------------------------------------------------- #
def _is_writer_node(ns: tuple) -> bool:
    """The ``messages`` stream carries tokens from EVERY model node; keep only the
    ``writer`` turn-graph node's (its subgraph namespace starts ``writer:``)."""
    return bool(ns) and ns[0].split(":")[0] == "writer"


def _chunk_text(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


__all__ = ["ShoppingSession"]
