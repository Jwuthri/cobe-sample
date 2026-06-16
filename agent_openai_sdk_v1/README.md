# agent_openai_sdk_v1

A multi-agent shopping assistant — browse a catalog, manage a cart, and run a
careful checkout to a confirmed order — rebuilt **from scratch on the
[OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/)**. It
is a clean-room rewrite of `pydantic_agent_v1`: same behavior and the same
frontend wire contract, but on the SDK's native primitives.

> One sentence: a **router** reads the conversation and delegates to three **worker**
> agents (browse / checkout / order-status) via `agent.as_tool()`; a dedicated
> **writer** streams the one reply; the **cart** is the single source of truth and
> the only thing that can say an order was placed.

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
| **runtime** | `runtime/` | Thin, generic glue: shared context, the event vocabulary, the agent-as-tool wrapper. | domain + the SDK |
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
5. **`session.py`** — the four-phase streaming turn.

---

## Why this is smaller than `pydantic_agent_v1`

The OpenAI Agents SDK has a direct equivalent for every primitive Pydantic AI
gave the previous build, plus it surfaces inner sub-agent events through
`Runner.run_streamed().stream_events()` — so the bespoke event bus + background
task collapse:

| Need | `pydantic_agent_v1` | here |
| --- | --- | --- |
| sub-agent as a tool | bespoke `Worker` + `run_subagent` | `agent.as_tool(tool_name, …, custom_output_extractor=…, is_enabled=…)` |
| checkout progress anchor | `@agent.instructions` | `Agent(instructions=callable)` (same `(ctx, agent) -> str`) |
| hide checkout on empty cart | tool `prepare=` returning `None` | `is_enabled=callable` on the worker's `as_tool()` |
| streaming inner events | `asyncio.Queue` bus + background task | `Runner.run_streamed().stream_events()` |
| token streaming | `run_stream().stream_text(delta=True)` | raw `ResponseTextDeltaEvent` from `stream_events()` |
| shared cart across agents | `RunContext` deps | `RunContextWrapper[ShoppingContext]` |

---

## How checkout stays honest

Checkout is the hard part, so three things make "confirmed" mean confirmed:

- **The cart is the memory, not the chat.** `Cart.step` is *derived* from which
  fields are filled in. Every checkout run gets a fresh, deterministic
  `checkout_progress(cart)` block (injected by `Agent(instructions=callable)`),
  so the model never rediscovers state from a growing thread.
- **The blocker gate is the real safety net.** `confirm_checkout` refuses while
  `cart.blockers()` is non-empty — the model cannot place an incomplete order no
  matter what it says, and the writer is told never to claim an order is placed
  unless `cart.confirmed` is true.
- **Edits can't double-add or quote stale totals.** Checkout has no `add_item` tool
  (a double-add is structurally impossible), and a quantity change invalidates the
  shipping/tax quotes (→ the `awaiting_pricing` step) so a stale total can never be
  confirmed.

---

## Two ideas worth knowing

**Context isolation.** Workers never see the conversation. The orchestrator is the
sole reader of the transcript: it resolves "the green one" / "add it" into a concrete
`query` (e.g. `add P-4 to the cart`) and passes only that. A worker is a clean
function of `(query + the shared cart)`. This keeps interpretation in one place, cuts
tokens, and shrinks the prompt-injection surface.

**The hallucination firewall.** The writer streams prose, but the structured cards
beside it (product lists, order summaries, totals) are built **deterministically**
from verified step results + the live cart (`agents/writer/blocks.py`). The model
never writes an id or a total, so it cannot hallucinate one — which is exactly why
the writer can be the terminal call and stream freely.

---

## File map

```
agent_openai_sdk_v1/
├── domain/                 # the store — pure logic, no LLM concepts
│   ├── cart.py             #   Cart + CheckoutStep machine + blockers()  ← the spec
│   ├── cart_service.py     #   the only mutator + invalidation policy
│   ├── catalog.py  pricing.py  serviceability.py  orders.py  memory.py
├── runtime/                # thin generic glue
│   ├── context.py          #   ShoppingContext: the shared state every tool sees
│   ├── delegation.py       #   Worker + build_worker_tool (agent-as-tool, once)
│   ├── step.py  model.py  events.py  trace.py  guardrails.py
├── agents/                 # five minds, one file each
│   ├── orchestrator.py     #   the router (reads chat, resolves refs, delegates)
│   ├── product_rec.py      #   worker: browse + cart edits
│   ├── checkout.py         #   worker: drive the purchase            ★
│   ├── order_status.py     #   worker: past orders
│   ├── writer/             #   the voice: prompt + grounded payload + typed blocks
│   └── tools.py            #   every action, as thin @function_tool wrappers
├── session.py              # the four-phase streaming turn engine
└── snapshot.py             # cart + transcript → the frontend's AgentSnapshot
```

---

## Run it

```bash
# API server (same SSE contract as the other engines → the web UI just works)
uvicorn server.main_openai_sdk_v1:app --reload --port 8004
# point the frontend at it:  AGENT_V2_API_URL=http://localhost:8004

# tests (offline — pure domain / blocks / progress / guardrails / snapshot)
uv run pytest tests_agent_openai_sdk_v1 -q
```

**Model.** Resolved from the env: `AGENT_OPENAI_SDK_V1_MODEL` →
`AGENT_V2_OPENAI_MODEL` → `gpt-5.4-mini`. Needs `OPENAI_API_KEY` (the package
loads the repo `.env` on import).

**Wire contract.** Every SSE event yielded by `ShoppingSession.run_turn_stream`
matches the shape `web/lib/types.ts` already understands — `user`, `router`,
`tool_start`, `tool_end`, `step`, `guardrail`, `token`, `writer`, `bot`, `state`,
`trace`, `error`, `end` — so the existing web UI swaps engines via the env var
above without any frontend changes.
