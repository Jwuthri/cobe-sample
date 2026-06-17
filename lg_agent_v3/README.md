# lg_agent_v3

A multi-agent shopping assistant — browse a catalog, manage a cart, and run a careful
checkout to a confirmed order — built on **LangChain + LangGraph**. It evolves
[`lg_agent_v2`](../lg_agent_v2) with two architecture changes:

1. **One turn-graph.** The orchestrator and writer are **nodes of a single `StateGraph`**
   (`orchestrator → payload → writer → blocks`), not two separately-invoked agents. See
   [`graph.py`](graph.py). The session drives it with
   `astream(stream_mode=["custom","messages"], subgraphs=True)`: UI events raised via
   `deps.emit` (`get_stream_writer`) propagate from any depth, and the **writer** node's
   tokens surface natively (filtered by namespace) — no bus, no manual re-pump.
2. **Real per-agent guardrails.** Every agent (orchestrator, each sub-agent, the writer)
   carries its own `before_agent`/`after_agent` guardrail middleware
   (`blocklist` / `pii` / `llm_judge`). A block on the orchestrator routes the turn to a
   **verbatim refusal**; a block on a sub-agent comes back as a flagged guardrail step
   and is delivered the same way. The session keeps only an input **redactor** (clean PII
   before it enters the transcript). See [`runtime/guardrails.py`](runtime/guardrails.py).

Everything else — the three-layer architecture (domain / runtime / agents), the
agent-as-tool delegation, the deterministic blocks "hallucination firewall", and the
frontend wire contract — is unchanged from `lg_agent_v2`.

> One sentence: a **router** reads the conversation and delegates to **worker** agents
> (browse / checkout / order-status); a dedicated **writer** streams the one reply; every
> agent owns its guardrails; the **cart** is the single source of truth.

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
| **runtime** | `runtime/` | Thin, generic glue: shared deps, the event vocabulary, model resolution, the agent-as-tool wrapper, the middleware primitives. | domain + LangChain |
| **agents** | `agents/` | Five minds, **one self-contained file each**. | domain + runtime |

`session.py` + `snapshot.py` tie them into a streaming turn the web UI consumes. The
`domain/`, `snapshot.py`, and the writer's `payload`/`blocks` are **byte-for-byte the
pydantic build** (modulo the package name): the spec didn't change, only the harness.

---

## What "just changing the harness" means

Pydantic AI gave a few things as first-class features. LangChain expresses the same
ideas; this table is the entire diff between the two builds:

| Need | `pydantic_agent_v1` | here (`lg_agent_v2`) |
| --- | --- | --- |
| an agent | `Agent(model, instructions=…, tools=[…])` | `create_agent(model, system_prompt=…, tools=[…])` (compiled LangGraph graph) |
| sub-agent as a tool | `await child.run(deps=ctx.deps)` inside a tool | `await child.ainvoke({"messages":[…]}, context=deps)` inside a `@tool` (`runtime/delegation.py`) |
| shared cart across agents | `RunContext` deps, passed by reference | `context_schema=ShoppingDeps` + `ToolRuntime.context`, passed by reference |
| checkout progress anchor | `@checkout.instructions` (re-rendered each run) | `dynamic_instructions(checkout_progress)` middleware (`runtime/middleware.py`) |
| hide checkout on empty cart | the checkout tool's `prepare=` | `hide_tool("checkout", cart_empty)` middleware |
| one worker per compound msg | `parallel_tool_calls=False` | `no_parallel_tools()` middleware (→ `bind_tools`) |
| token streaming | `run_stream().stream_text(delta=True)` | `astream(stream_mode="messages")`, filter the `model` node's `AIMessageChunk`s |
| live router/tool/step events | a per-turn `asyncio.Queue` bus + background task | **identical** — the same bus + background task, verbatim |

That last row is the point: `session.py`, `runtime/deps.py`, and `runtime/delegation.py`
keep Pydantic AI's exact streaming shape (an `asyncio.Queue` the workers `emit` onto
while the orchestrator runs in a background `create_task`). Only the *model calls*
inside that skeleton changed.

---

## Read the code in this order

1. **`domain/cart.py`** — the `Cart`, the `CheckoutStep` machine, and `blockers()`.
   Everything else exists to move this object forward. Start here.
2. **`domain/cart_service.py`** — the only thing that mutates a cart, plus the
   freshness/invalidation rules (change the zip → shipping/tax/serviceability reset).
3. **`runtime/middleware.py`** — the three primitives (`dynamic_instructions`,
   `hide_tool`, `no_parallel_tools`) that stand in for Pydantic AI's natives.
4. **`agents/checkout.py`** — ★ the star. The checkout worker, its tools, and the
   deterministic "Checkout progress" anchor that is re-injected on every run.
5. **`agents/orchestrator.py`** — the router: how it resolves references and routes.
6. **`session.py`** — the four-phase streaming turn.

---

## How checkout stays honest

Checkout is the hard part, so three things make "confirmed" mean confirmed:

- **The cart is the memory, not the chat.** `Cart.step` is *derived* from which fields
  are filled in. Every checkout run gets a fresh, deterministic `checkout_progress(cart)`
  block (a `dynamic_instructions` middleware), so the model never rediscovers state from
  a growing thread.
- **The blocker gate is the real safety net.** `confirm_checkout` refuses while
  `cart.blockers()` is non-empty — the model cannot place an incomplete order no matter
  what it says, and the writer is told never to claim an order is placed unless
  `cart.confirmed` is true.
- **Edits can't double-add or quote stale totals.** Checkout has no `add_item` tool (a
  double-add is structurally impossible), and a quantity change invalidates the
  shipping/tax quotes (→ the `awaiting_pricing` step) so a stale total can never be
  confirmed.

---

## Two ideas worth knowing

**Context isolation.** Workers never see the conversation. The orchestrator is the sole
reader of the transcript: it resolves "the green one" / "add it" into a concrete `query`
(e.g. `add P-4 to the cart`) and passes only that. A worker is a clean function of
`(query + the shared cart)`. This keeps interpretation in one place, cuts tokens, and
shrinks the prompt-injection surface.

**The hallucination firewall.** The writer streams prose, but the structured cards
beside it (product lists, order summaries, totals) are built **deterministically** from
verified step results + the live cart (`agents/writer/blocks.py`). The model never
writes an id or a total, so it cannot hallucinate one — which is exactly why the writer
can be the terminal call and stream freely.

---

## File map

```
lg_agent_v2/
├── domain/                 # the store — pure logic, no LLM concepts  (== pydantic build)
│   ├── cart.py             #   Cart + CheckoutStep machine + blockers()  ← the spec
│   ├── cart_service.py     #   the only mutator + invalidation policy
│   ├── catalog.py  pricing.py  serviceability.py  orders.py  memory.py
├── runtime/                # thin generic glue (the harness)
│   ├── deps.py             #   ShoppingDeps: the shared context every tool sees + the bus
│   ├── delegation.py       #   Worker + run_subagent  (agent-as-tool, once)
│   ├── middleware.py       #   dynamic_instructions / hide_tool / no_parallel_tools
│   ├── step.py  model.py  events.py  trace.py  guardrails.py
├── agents/                 # five minds, one file each
│   ├── orchestrator.py     #   the router (reads chat, resolves refs, delegates)
│   ├── product_rec.py      #   worker: browse + cart edits
│   ├── checkout.py         #   worker: drive the purchase            ★
│   ├── order_status.py     #   worker: past orders
│   ├── writer/             #   the voice: prompt + grounded payload + typed blocks
│   └── tools.py            #   every action, as thin domain wrappers (LangChain @tool)
├── session.py              # the four-phase streaming turn engine
└── snapshot.py             # cart + transcript → the frontend's AgentSnapshot
```

---

## Run it

```bash
# API server (same SSE contract as the other engines → the web UI just works)
uvicorn server.main_lg_agent_v3:app --reload --port 8007
# point the frontend at it:  AGENT_V2_API_URL=http://localhost:8007

# tests (33, fully offline — agents driven by scripted fake chat models)
uv run pytest tests_lg_agent_v3 -q
```

**Model.** Resolved from the env: `LG_AGENT_V3_MODEL` → `AGENT_V2_OPENAI_MODEL` →
`openai:gpt-5.4-mini`. Needs `OPENAI_API_KEY` (the package loads the repo `.env` on
import).

**Verified end to end (real gpt-5.4-mini):** the full checkout runs through the
turn-graph — add → identity → address + serviceability → 2h delivery (auto shipping + tax
quote, total **$74.35**) → cash → ready → `yes` → receipt `RCPT-9000`; the writer streams
tokens throughout (same flow + total as `lg_agent_v2`). A configured orchestrator
guardrail (`blocklist`/`llm_judge`) blocks off-policy input and the writer delivers the
refusal verbatim with no model spend.
