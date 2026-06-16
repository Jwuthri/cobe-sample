# google_adk_agent_v1

A multi-agent shopping assistant — browse a catalog, manage a cart, and run a
careful checkout to a confirmed order — rebuilt **from scratch on
[Google ADK](https://google.github.io/adk-docs/)** (the Agent Development Kit). It is
a clean-room rewrite of `agent_v4_1` and a sibling of `pydantic_agent_v1`: same
behavior and the same frontend wire contract, reorganized so the whole thing is easy
to read end to end.

> One sentence: a **router** reads the conversation and delegates to three **worker**
> agents (browse / checkout / order-status); a dedicated **writer** streams the one
> reply; the **cart** is the single source of truth and the only thing that can say
> an order was placed.

---

## The big picture

```
                    ┌──────────────────────────────────────────────┐
   user turn ─────► │  ShoppingSession.run_turn_stream  (session.py)│
                    └──────────────────────────────────────────────┘
                        │  1. input guardrails (pre-flight)
                        │  2. orchestrator  ──► delegates to ONE worker per request
                        │  3. writer        ──► streams the reply, token by token
                        │  4. blocks + bot  ──► deterministic cards + final message
                        ▼
   ┌───────────────┐  delegate (agent-as-tool)   ┌───────────────────────────┐
   │ orchestrator  │ ──────────────────────────► │ product_rec │ browse + cart│
   │  (the router) │                             │ checkout    │ the purchase │
   │  reads chat,  │                             │ order_status│ past orders  │
   │  resolves refs│ ◄────── terse summary ───── └───────────────────────────┘
   └───────────────┘                                    │ all mutate ▼
                                                  ┌───────────────────────────┐
   writer ◄── grounded payload + cart ──────────  │  CartService → Cart        │
   (streams)                                       │  (domain/ — the spec)      │
                                                  └───────────────────────────┘
```

Three layers, each with one job:

| Layer | Folder | Job | Knows about |
| --- | --- | --- | --- |
| **domain** | `domain/` | The store, as pure logic — the checkout state machine, pricing, catalog. The behavioral **spec**. | nothing (no LLM/agent imports) |
| **runtime** | `runtime/` | Thin, generic glue: shared deps, the event vocabulary, model resolution, the agent-as-tool wrapper, the ADK runner helpers. | domain + Google ADK |
| **agents** | `agents/` | Five minds, **one self-contained file each**. | domain + runtime |

`session.py` + `snapshot.py` tie them into a streaming turn the web UI consumes.

---

## Read the code in this order

1. **`domain/cart.py`** — the `Cart`, the `CheckoutStep` machine, and `blockers()`.
   Everything else exists to move this object forward. Start here.
2. **`domain/cart_service.py`** — the only thing that mutates a cart, plus the
   freshness/invalidation rules (change the zip → shipping/tax/serviceability reset).
3. **`agents/checkout.py`** — ★ the star. The checkout worker, its tools, and the
   deterministic "Checkout progress" anchor that is re-injected on every run.
4. **`agents/orchestrator.py`** — the router: how it resolves references and routes.
5. **`runtime/delegation.py`** + **`session.py`** — the agent-as-tool wrapper and the
   four-phase streaming turn.

---

## How checkout stays honest

Checkout is the hard part, so three things make "confirmed" mean confirmed:

- **The cart is the memory, not the chat.** `Cart.step` is *derived* from which
  fields are filled in. Every checkout run gets a fresh, deterministic
  `checkout_progress(cart)` block (an ADK instruction provider), so the model never
  rediscovers state from a growing thread.
- **The blocker gate is the real safety net.** `confirm_checkout` refuses while
  `cart.blockers()` is non-empty — the model cannot place an incomplete order no
  matter what it says, and the writer is told never to claim an order is placed
  unless `cart.confirmed` is true.
- **Edits can't double-add or quote stale totals.** Checkout has no `add_item` tool
  (a double-add is structurally impossible), and a quantity change invalidates the
  shipping/tax quotes (→ the `awaiting_pricing` step) so a stale total can never be
  confirmed.

---

## How ADK maps to the design

ADK gives the moving parts as first-class features. The interesting bit is *how* each
concept lands — and the one place it diverges from a `ctx.deps`-style framework:

| Need | here (Google ADK) |
| --- | --- |
| sub-agent as a tool | a plain async tool fn → `run_subagent`, which runs the worker on a fresh, isolated session (`runtime/delegation.py` + `runtime/runner_util.py`) |
| checkout progress anchor | the checkout agent's `instruction=` **provider** (re-rendered each run) |
| hide checkout on empty cart | the orchestrator's `before_model_callback` strips the `checkout` tool's declaration while the cart is empty (structural, not prompt-enforced) |
| token streaming | `RunConfig(streaming_mode=SSE)` → partial-text delta events |
| shared cart across agents | **a registry key in session state** — see below |
| "one worker per step" | `deps.lock` serializes delegated runs (ADK's analogue of `parallel_tool_calls=False`) |

**The shared-cart trick (the one real ADK-ism).** Pydantic AI handed every tool a live
`ctx.deps`. ADK instead gives tools a *session* whose `state` dict is **deep-copied**
as it flows between the runner and the (sub-)agents — so a live `CartService` placed
in state would be *cloned*, and a mutation in one agent would never reach the next. So
we keep the live object in a tiny process-level **registry** (`runtime/registry.py`)
and put only a short **string key** in session state (a string survives a deep-copy).
Every tool resolves the key back to the one shared cart:

```python
deps = registry.get(tool_context.state[RUNTIME_KEY])  # the one live CartService
```

One cart is mutated in place across the orchestrator and every worker — the
"shared deps, by reference" property, reconstructed structurally for ADK.

---

## Two ideas worth knowing

**Context isolation.** Workers never see the conversation. The orchestrator is the
sole reader of the transcript: it resolves "the green one" / "add it" into a concrete
`query` (e.g. `add P-4 to the cart`) and passes only that. A worker runs in a fresh
ADK session whose only message is that `query` — a clean function of `(query + the
shared cart)`. This keeps interpretation in one place, cuts tokens, and shrinks the
prompt-injection surface.

**The hallucination firewall.** The writer streams prose, but the structured cards
beside it (product lists, order summaries, totals) are built **deterministically**
from verified step results + the live cart (`agents/writer/blocks.py`). The model
never writes an id or a total, so it cannot hallucinate one — which is exactly why
the writer can be the terminal call and stream freely.

---

## File map

```
google_adk_agent_v1/
├── domain/                 # the store — pure logic, no LLM concepts
│   ├── cart.py             #   Cart + CheckoutStep machine + blockers()  ← the spec
│   ├── cart_service.py     #   the only mutator + invalidation policy
│   ├── catalog.py  pricing.py  serviceability.py  orders.py  memory.py
├── runtime/                # thin generic glue
│   ├── deps.py             #   ShoppingDeps: the shared state every tool sees
│   ├── registry.py         #   the live-cart-by-reference trick (string key in state)
│   ├── delegation.py       #   Worker + run_subagent  (agent-as-tool, once)
│   ├── runner_util.py      #   run/stream an ADK agent on a fresh session
│   ├── model.py  events.py  step.py  trace.py  guardrails.py
├── agents/                 # five minds, one file each
│   ├── orchestrator.py     #   the router (reads chat, resolves refs, delegates)
│   ├── product_rec.py      #   worker: browse + cart edits
│   ├── checkout.py         #   worker: drive the purchase            ★
│   ├── order_status.py     #   worker: past orders
│   ├── writer/             #   the voice: prompt + grounded payload + typed blocks
│   └── tools.py            #   every action, as thin domain wrappers
├── session.py              # the four-phase streaming turn engine
└── snapshot.py             # cart + transcript → the frontend's AgentSnapshot
```

---

## Run it

```bash
# API server (same SSE contract as the other engines → the web UI just works)
uvicorn server.main_google_adk_agent_v1:app --reload --port 8005
# point the frontend at it:  AGENT_V2_API_URL=http://localhost:8005

# tests (26, fully offline — agents driven by a scripted ADK BaseLlm)
uv run pytest tests_google_adk_agent_v1 -q
```

**Model.** Resolved from the env: `GOOGLE_ADK_AGENT_V1_MODEL` → `AGENT_V2_OPENAI_MODEL`
→ `gpt-5.4-mini`. A `gemini-*` id runs natively on Gemini (needs a `GOOGLE_API_KEY`);
anything else is wrapped in ADK's `LiteLlm` — a bare id is assumed OpenAI and runs on
the repo's existing `OPENAI_API_KEY`, so the ADK port is a true behavioral twin of
`pydantic_agent_v1` (same model, same key). The package loads the repo `.env` on import.

**Verified end to end (real gpt-5.4-mini):** full checkout — add → identity →
address + serviceability → 2h delivery (auto shipping + tax quote, total **$74.35**) →
cash → ready → `yes` → receipt `RCPT-9000`; plus reference resolution ("add the green
one" → P-4), a mid-flow quantity edit ("make it 2"), and serviceability-by-city ("do
you ship to paris?" → asks for a ZIP). The writer streams tokens throughout.
