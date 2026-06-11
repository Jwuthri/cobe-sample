# `agent_agno_v4_1/` — the v4.1 shopping assistant, rebuilt on Agno

A faithful port of [`agent_v4_1`](../agent_v4_1/README.md) onto **[Agno](https://docs.agno.com) 2.6**,
swapping the LangGraph orchestration for Agno's `Team` + `Agent` primitives while
keeping every behavioral property that matters: a dedicated streaming writer,
typed "rich reply" blocks of different kinds, and the full cart-anchored checkout
state machine (including the stale-pricing recompute path).

The framework-agnostic core is **reused, not recopied** — the domain
(`cart` / `catalog` / `orders` / `pricing` / `serviceability`), the typed
`blocks`, the `prompts`, the `StepResult` model, and the pure checkout helpers
(`checkout_anchor_text`, `asks_for_step`) all import from `agent_v4_1`. Only the
orchestration layer (which was coupled to LangChain message types) is rewritten.

## Topology

```
                    ┌─────────────────────────────────────┐
   user turn  ─────▶│  Team (coordinate mode) = the router │
                    │   ├─ product_rec   browse + cart edit │   members share a live
                    │   ├─ checkout      drive the purchase │◀── CartService via
                    │   └─ order_status  past-order lookup  │   `dependencies` (by ref)
                    └──────────────┬──────────────────────-─┘
                                   │ member_responses[*].tools  → StepResult extraction
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  writer (separate Agent, no tools)   │  streams tokens live
                    │  composes from grounded StepResults  │  ({type:"token"})
                    │  + cart; blocks built deterministically
                    └─────────────────────────────────────┘
```

## How the Agno primitives map to v4_1

| Concern | v4_1 (LangGraph) | this package (Agno 2.6) |
|---|---|---|
| Router | `create_agent` orchestrator that calls sub-agent tools, emits `DONE` | `Team(mode=coordinate)` whose leader delegates to member agents (`agents.py`) |
| Sub-agents | `SubagentSpec` + `make_subagent_tool` | three member `Agent`s with their own tools (`agents.py`, `tools.py`) |
| Shared cart | `runtime.context.cart_service` | `run_context.dependencies["ctx"]` — Agno injects `run_context` by name; the live `CartService` mutates in place and propagates to every member (`context.py`, `tools.py`) |
| Checkout anchor | `cart_anchor` middleware | **dynamic instructions** — a callable on the checkout member that injects `checkout_anchor_text(cart)` every turn (`agents.py`) |
| Tool-result → facts | parse LangChain `ToolMessage`s | parse Agno `member_responses[*].tools` (`ToolExecution`) (`extractors.py`) |
| Writer | terminal `create_agent`, `astream(stream_mode="messages")` | terminal `Agent`, `arun(stream=True)` → `RunContentEvent` deltas (`session.py`) |
| Blocks | `agent_v4_1/shopping/blocks.py` | **reused verbatim** — the hallucination firewall (ids/prices never written by a model) |

## Why a dedicated writer instead of the Team's own answer

In coordinate mode the team leader would synthesize a final answer itself. We
**discard** it and run a separate writer instead, because the writer composes only
from *grounded* `StepResult`s + the cart — never from raw tool output it could
misread — and the typed blocks are assembled deterministically alongside it. That
grounding is what lets the writer stream freely: the facts are already locked.

## Checkout, with all its complexity

The checkout member is stateless and re-anchored every turn off the cart's
`step`. The cart is the source of truth, so a cart edit mid-checkout is handled
correctly: lowering a quantity at `ready_to_confirm` invalidates the shipping
quote + tax, the `step` becomes `awaiting_pricing`, and the anchor instructs the
member to recompute (`quote_shipping` + `compute_tax`) before the (recomputed)
total can be confirmed. `confirm_checkout` stays gated by the cart invariant
`blockers()` — the real safety net.

## Run / test

```bash
# live server — same web UI as v2/v4/v4_1/v5; the reply types out token-by-token
uvicorn server.main_agno_v4_1:app --reload --port 8001

# unit tests — 21 tests, NO real LLM (Team/writer are faked)
uv run pytest tests_agno_v4_1 -q
```

Model resolution mirrors v4_1: `AGENT_AGNO_V4_1_MODEL` → `AGENT_V4_1_MODEL` →
`AGENT_V2_OPENAI_MODEL` → `gpt-4.1-mini`. A live run needs `OPENAI_API_KEY`. The
reasoning-model temperature quirk is handled (gpt-5.x get the default temperature
rather than a 400).

## Known nuance

For a pure "what's in my cart" view, a fast model sometimes answers from the
injected cart note instead of calling `get_cart_summary`, so the cart *card* is
occasionally skipped (the prose answer is still correct). Cart **edits**
(add/remove/quantity), product lists, order status, and checkout summaries always
produce their block.
