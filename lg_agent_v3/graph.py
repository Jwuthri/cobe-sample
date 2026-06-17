"""The turn-graph — orchestrator, writer, and blocks as nodes of one ``StateGraph``.

A turn is one graph: ``orchestrator → payload → writer → blocks → END``. The
orchestrator and writer are real nodes (not two separately-invoked agents); the cart +
step-results live on the shared ``ShoppingDeps`` context, passed by reference into the
graph and forwarded into every nested agent run. UI events are raised via ``deps.emit``
(``get_stream_writer``) and reach the session's ``astream(subgraphs=True)`` from any
depth — no bus.

Guardrail routing is native to the graph:
  * an **orchestrator** block sets ``refusal`` in state → the writer node delivers it
    verbatim (a refusal must keep its exact wording);
  * a **sub-agent** block comes back from ``run_subagent`` as a flagged guardrail step
    → the payload node promotes it to the same verbatim-refusal path.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from lg_agent_v3.agents import (
    BLOCK_BY_SOP,
    WRITER_SYSTEM,
    absorb_recalls,
    build_blocks,
    build_writer_payload,
)
from lg_agent_v3.runtime import ShoppingDeps
from lg_agent_v3.runtime import events, trace
from lg_agent_v3.snapshot import build_snapshot

_EMPTY_WRITER_FALLBACK = "Sorry, I couldn't produce a response. Could you rephrase that?"


class TurnState(TypedDict, total=False):
    messages: list  # orchestrator input (history + the new user turn)
    refusal: str | None  # set by a guardrail block → verbatim writer delivery
    payload: str  # writer payload json
    mode: str
    text: str  # final reply text
    blocks: list  # deterministic typed cards


def _last_ai_text(result: dict) -> str:
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return ""


def _snapshot(deps: ShoppingDeps) -> dict:
    return build_snapshot(
        user_id=deps.user_id,
        session_id=deps.session_id,
        cart=deps.cart_service.cart,
        transcript=deps.transcript,
    )


def build_turn_graph(orchestrator_agent: Any, writer_agent: Any) -> Any:
    """Compile the turn-graph closing over the (possibly guardrail-equipped) agents.

    The session drives this with ``astream(stream_mode=["custom","messages"],
    subgraphs=True)``: UI events raised via ``deps.emit`` (``get_stream_writer``)
    propagate up the custom channel from any depth; the **writer** node's model tokens
    surface on the messages channel and the session filters them by node namespace. So
    the writer node just ``ainvoke``s — its tokens stream natively, no re-pump.
    """

    async def orchestrator_node(state: TurnState, runtime: Runtime[ShoppingDeps]) -> dict:
        deps = runtime.context
        hits_before = len(deps.guardrail_hits)
        try:
            await orchestrator_agent.ainvoke({"messages": state["messages"]}, context=deps)
        except Exception as e:  # noqa: BLE001
            deps.emit(events.error(str(e)))
        absorb_recalls(deps)  # remember this turn's recalls for the next one
        deps.emit(events.state(_snapshot(deps)))
        # an orchestrator-level guardrail block → route the turn to a verbatim refusal
        block = next(
            (h for h in deps.guardrail_hits[hits_before:] if h.agent == "orchestrator" and h.action == "block"),
            None,
        )
        if block is not None:
            return {"refusal": block.message or "I'm not able to help with that request."}
        return {}

    def payload_node(state: TurnState, runtime: Runtime[ShoppingDeps]) -> dict:
        deps = runtime.context
        deps.emit(events.router("writer", 0))
        # a sub-agent guardrail block surfaced as a flagged step → same verbatim path
        refusal = state.get("refusal") or next(
            (s.details["guardrail"] for s in deps.steps if s.details and s.details.get("guardrail")),
            None,
        )
        if refusal:
            return {"refusal": refusal, "mode": "refusal"}
        payload, mode = build_writer_payload(deps.transcript, deps.steps, deps.cart_service.cart)
        if deps.debug:
            deps.emit(
                trace.frame(
                    "writer_payload",
                    "writer",
                    f"writer input (mode={mode})",
                    {"system_prompt": WRITER_SYSTEM, "mode": mode, "payload": json.loads(payload)},
                )
            )
        return {"payload": payload, "mode": mode}

    async def writer_node(state: TurnState, runtime: Runtime[ShoppingDeps]) -> dict:
        deps = runtime.context
        if state.get("refusal"):
            # a refusal has no model call (verbatim wording) → emit the token directly so
            # the UI still renders it like a streamed reply.
            deps.emit(events.token(state["refusal"]))
            return {"text": state["refusal"]}
        # ainvoke (NOT a manual stream): the session's astream(messages, subgraphs=True)
        # surfaces this node's model tokens natively. Retry once if empty — stream-safe,
        # an empty attempt sent zero tokens.
        text = ""
        for _attempt in (1, 2):
            result = await writer_agent.ainvoke({"messages": [HumanMessage(content=state["payload"])]}, context=deps)
            text = _last_ai_text(result)
            if text:
                break
        return {"text": text}

    def blocks_node(state: TurnState, runtime: Runtime[ShoppingDeps]) -> dict:
        deps = runtime.context
        text = state.get("text") or _EMPTY_WRITER_FALLBACK
        # a refusal carries no cards
        blocks = [] if state.get("refusal") else build_blocks(deps.steps, deps.cart_service.cart, BLOCK_BY_SOP)
        deps.emit(events.writer(text, blocks))
        deps.emit(events.bot(text, blocks))
        return {"text": text, "blocks": blocks}

    g = StateGraph(TurnState, context_schema=ShoppingDeps)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("payload", payload_node)
    g.add_node("writer", writer_node)
    g.add_node("blocks", blocks_node)
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "payload")
    g.add_edge("payload", "writer")
    g.add_edge("writer", "blocks")
    g.add_edge("blocks", END)
    return g.compile()


__all__ = ["build_turn_graph", "TurnState"]
