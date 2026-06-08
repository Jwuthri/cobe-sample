"""Diagnostic: measure OpenAI prompt-cache hit rate per graph node.

OpenAI caches prompt PREFIXES > 1024 tokens; a cache hit requires the prefix to
be byte-identical to a recent (<~5-10 min) request. Putting volatile content
(cart notes, step results) near the FRONT of the message list — or sliding a
history window — shifts the prefix every turn and defeats caching. This meter
attaches to a graph run and reports, per ``langgraph_node``, how many input
tokens were served from cache vs reprocessed.

Run:
    python -m agent_v5.cache_meter v4     # measure agent_v4 over the eval script
    python -m agent_v5.cache_meter v5     # measure agent_v5 (router) over the same
"""

from __future__ import annotations

import sys
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler


class CacheMeter(BaseCallbackHandler):
    """Tally input vs cached-input tokens per langgraph node."""

    def __init__(self) -> None:
        self._meta: dict = {}
        self.by_node: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [input, cached, calls]

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw) -> None:
        md = metadata or {}
        # checkpoint_ns looks like "product_rec_wrapper:<uuid>"; the prefix names
        # the leaf. Fall back to the bare node name.
        ns = md.get("checkpoint_ns") or ""
        leaf = ns.split(":", 1)[0] if ns else md.get("langgraph_node", "?")
        self._meta[run_id] = leaf or md.get("langgraph_node", "?")

    def on_llm_end(self, response, *, run_id=None, **kw) -> None:
        node = self._meta.pop(run_id, "?")
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                if not um:
                    continue
                inp = int(um.get("input_tokens", 0) or 0)
                cached = int((um.get("input_token_details") or {}).get("cache_read", 0) or 0)
                row = self.by_node[node]
                row[0] += inp
                row[1] += cached
                row[2] += 1

    def report(self, title: str) -> str:
        lines = [f"\n=== {title} ===", f"{'node':<26} {'calls':>6} {'input':>9} {'cached':>9} {'hit%':>6}"]
        tin = tc = 0
        for node, (inp, cached, calls) in sorted(self.by_node.items()):
            tin += inp
            tc += cached
            pct = (cached / inp * 100) if inp else 0.0
            lines.append(f"{node:<26} {calls:>6} {inp:>9,} {cached:>9,} {pct:>5.0f}%")
        total_pct = (tc / tin * 100) if tin else 0.0
        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<26} {'':>6} {tin:>9,} {tc:>9,} {total_pct:>5.0f}%")
        return "\n".join(lines)


SCRIPT = [
    "hey, what's up?",
    "what hoodies and caps do you carry?",
    "add the black hoodie to my cart",
    "my name is Julien Martin",
    "ship it to 500 Market St, San Francisco, CA 94110",
    "wait — first, where's my order ORD-7?",
    "ok back to it. use the 2 hour delivery",
    "is my order placed yet?",
    "pay with card, token tok_visa_42",
    "yes, place the order",
]


def run_v4(meter: CacheMeter) -> None:
    import uuid

    from agent_v4.graph import build_graph
    from agent_v4.state import AgentState
    from langchain_core.messages import HumanMessage

    graph = build_graph()
    state = AgentState(user_id="demo", session_id=f"cache-v4-{uuid.uuid4().hex[:6]}")
    cfg = {"callbacks": [meter]}
    for msg in SCRIPT:
        state = state.model_copy(
            update={
                "messages": state.messages + [HumanMessage(content=msg)],
                "draft_response": None,
                "draft_blocks": [],
                "validation_errors": [],
                "response_attempts": 0,
                "step_results": [],
                "iteration": 0,
                "done": False,
            }
        )
        result = graph.invoke(state, config=cfg)
        state = AgentState.model_validate(result)


def run_v5(variant: str = "router") -> str:
    """v5 subagents are invoked inside tools (callbacks don't reach them), so we
    read cache stats from the per-component usage tally instead."""
    from agent_v5.agent import ShoppingAgentV5

    agent = ShoppingAgentV5(variant=variant, session_id=f"cache-v5-{variant}")
    agg: dict[str, list[int]] = {
        "supervisor": [0, 0, 0],
        "subagents": [0, 0, 0],
        "writer": [0, 0, 0],
    }
    for msg in SCRIPT:
        r = agent.run_turn(msg)
        for comp, row in agg.items():
            u = r.usage_breakdown[comp]
            row[0] += u.get("input_tokens", 0)
            row[1] += u.get("cached_tokens", 0)
            row[2] += u.get("llm_calls", 0)
    lines = [
        f"\n=== prompt-cache hit rate — v5 ({variant}) ===",
        f"{'component':<26} {'calls':>6} {'input':>9} {'cached':>9} {'hit%':>6}",
    ]
    tin = tc = 0
    for comp, (inp, cached, calls) in agg.items():
        tin += inp
        tc += cached
        pct = (cached / inp * 100) if inp else 0.0
        lines.append(f"{comp:<26} {calls:>6} {inp:>9,} {cached:>9,} {pct:>5.0f}%")
    lines.append("-" * 60)
    lines.append(f"{'TOTAL':<26} {'':>6} {tin:>9,} {tc:>9,} {(tc/tin*100 if tin else 0):>5.0f}%")
    return "\n".join(lines)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "v4"
    if which == "v4":
        meter = CacheMeter()
        run_v4(meter)
        print(meter.report("prompt-cache hit rate — v4"))
    else:
        print(run_v5("router"))


if __name__ == "__main__":
    main()
