# lg_agent

A clean, config-driven multi-agent shopping assistant built on LangChain
`create_agent` / LangGraph, with **true token streaming**.

It is a ground-up rewrite of `agent_v4_1` with the same behavior and the same
frontend contract, reorganized so that **every agent's definition lives in one
obvious place** and the moving parts are small and single-purpose.

```
lg_agent/
├── core/                  the reusable platform (knows nothing about shopping)
└── shopping/              the demo tenant (e-commerce assistant)
```

---

## The two layers

### `core/` — the platform

Tenant-agnostic machinery for turning a declarative config into a running agent.
It never imports a tenant.

| module | responsibility |
|---|---|
| `config.py` | the `AgentConfig` **JSON contract** — what a tenant writes to define an agent |
| `registry.py` | the four capability registries: `TOOLS`, `SKILLS`, `GUARDRAILS`, `MIDDLEWARE` |
| `builder.py` | `build_agent(config)` → a compiled `create_agent` graph |
| `subagent.py` | `SubAgent` + `as_tool()` — wrap a config-built agent as an orchestrator tool |
| `model.py` · `tools.py` · `skills.py` · `guardrails.py` · `middleware.py` | the resolver behind each config field |
| `context.py` · `step.py` · `trace.py` | per-turn shared state, the `StepResult`, and debug tracing |

### `shopping/` — the demo tenant

| area | responsibility |
|---|---|
| `domain/` | the mock e-commerce model: cart, catalog, pricing, serviceability, orders, memory. Pure logic — no LLM concepts. |
| `tools/` | the leaf tools, grouped by area (`catalog` / `cart` / `checkout` / `orders`) |
| `agents/` | the agents (see below) |
| `session.py` | `ShoppingSession.run_turn_stream` — the streaming turn engine |
| `snapshot.py` · `events.py` | the frontend projections (cart snapshot + SSE event mapping) |
| `middleware.py` · `setup.py` | tenant middleware + one-time registry registration |

---

## Two kinds of agent, handled differently

This is the core idea of the rewrite.

### Pre-defined agents — `agents/orchestrator/`, `agents/writer/`

The orchestrator and the writer have **fixed roles**, so each is a hand-authored
package whose abstractions are separate files:

```
agents/orchestrator/          agents/writer/
├── __init__.py   build+config ├── __init__.py   build+config
├── prompt.py     how it routes├── prompt.py     the voice
└── routing.py    ref-resolution├── payload.py    its grounded INPUT
                               └── blocks.py     its typed OUTPUT schema
```

Open the folder, see everything about that agent.

### On-the-fly sub-agents — `agents/subagents/`

`product_rec`, `checkout`, and `order_status` are **built from a JSON config**
that references registry tools *by name*. Each is one self-contained file:

```python
# agents/subagents/product_rec.py
CONFIG = {                                   # ← pure JSON (validates against AgentConfig)
    "name": "product_rec",
    "system_prompt": PROMPT,
    "tools": registry_specs(PRODUCT_REC_TOOLS),   # tools referenced by name
    "middleware": [{"name": "log_tool_calls", ...}],
}

def extract(ctx, messages, before) -> StepResult: ...   # a few Python hooks
def build_input(ctx, query) -> dict: ...

SUBAGENT = SubAgent(name="product_rec", config=CONFIG, extract=extract, ...)
```

Because `CONFIG` is plain JSON, the same agent could just as well be loaded from a
file, a database, or an API — that is the "on-the-fly" story. The only Python is
the handful of hooks that bridge the run to the shared cart.

---

## The topology (agent-as-tool)

```
user turn
   │
   ▼
input guardrails ──(refusal)──▶ reply           (pre-flight, before any model call)
   │ (clean)
   ▼
ORCHESTRATOR  ── routes to ──▶  product_rec │ checkout │ order_status   (sub-agent tools)
   │  (resolves "the green one" → P-4; sub-agents never see the chat)        │
   │                                                                          ▼
   │                                                       each appends a StepResult
   ▼
WRITER  ◀── grounded payload (step results + cart) ──  composes the reply
   │  (LAST model call → tokens stream straight to the client)
   ▼
deterministic blocks (ids/prices verbatim) + final bot message
```

**Why it streams.** The writer is the terminal model call, so nothing gates its
output — its tokens flow to the client as it generates them. The structured cards
(product lists, order summaries) are built deterministically from verified step
results, never written by the model, so it cannot hallucinate an id or a total.
That grounding-at-construction is what lets the writer stream freely.

**Context isolation.** The orchestrator is the *sole* reader of the conversation.
It resolves every reference into a self-contained `query` and passes only that to a
sub-agent. A sub-agent operates on `(query + shared cart)` — never the transcript —
which keeps interpretation in one place and cuts tokens + prompt-injection surface.

**Checkout state.** The checkout agent doesn't carry a long prompt of step rules —
each turn the `cart_anchor` middleware injects a deterministic "Checkout progress"
block rendered from `cart.step` (what's ✓, what's next). The cart is the source of
truth, so the agent never re-derives state from a growing thread. (Skills remain a
platform feature — see the `AgentConfig.skills` contract — but the demo's agents
don't use them.)

---

## Adding things

**A new leaf tool** → write it in `shopping/tools/<area>.py`, add it to a tool
group in `tools/__init__.py`. It is registered automatically.

**A new on-the-fly sub-agent** → add `shopping/agents/subagents/<name>.py` (config
+ hooks + a `SUBAGENT`), then list it in `subagents/__init__.py::SUBAGENTS`.

**A new guardrail / middleware kind** → register a factory in `core/guardrails.py`
or `core/middleware.py`; configs can then reference it by name.

---

## Run

```bash
# API (the web UI defaults to http://localhost:8001)
uvicorn server.main_lg:app --reload --port 8001

# tests — 60 tests, no real LLM (fakes via GenericFakeChatModel / ToolCallingFake)
uv run pytest tests_lg -q
```

The model is `openai:gpt-5.4-mini` (per-agent in each config); override per agent
by editing that agent's `model` block, or globally via the `LG_AGENT_MODEL` /
`AGENT_V2_OPENAI_MODEL` env var.

---

## Frontend contract

`server/main_lg.py` speaks the same SSE/JSON contract as the existing web client:

* `POST /api/session` → `{session_id}`
* `GET  /api/state/{id}` → `AgentSnapshot` (cart + transcript)
* `POST /api/turn/{id}` → Server-Sent Events:
  `user · state · guardrail · router · agent · tool_start · tool_end · step ·
  trace · token · writer · bot · end · error`

The `AgentSnapshot` shape and the event vocabulary are identical to `agent_v4_1`,
so the unchanged frontend renders it the same — including live token streaming
(`{type:"token"}`) and the deep-trace debug panel (`{type:"trace"}`).

---

## Persistence & session replay

Every turn's events **and** state snapshots are teed to SQLite
([core/event_store.py](core/event_store.py)) — main agent *and* sub-agents, because
the orchestrator's stream already carries the sub-agents' traffic (tool calls, step
results, deep-trace frames), so teeing that one stream captures everything. The sink
is decoupled: the session takes an optional `events_store` and flushes one batched
transaction per turn off the event loop; no store wired → nothing is written.

Extra endpoints for the "load previous session" UI:

* `GET /api/sessions` → all stored sessions (id, turns, `live` = still in memory)
* `GET /api/events/{id}` → every persisted event in order (`data` = the original event)
* `GET /api/snapshots/{id}` → the state snapshot per `state` event

**Replay is faithful by construction:** the web client re-feeds the stored events
through the *same* `logEntriesFor` projection the live stream uses, so a loaded
session renders identically — same event log, same cart, same transcript, every
sub-agent row included. A session still in memory is resumable; an archived one
(after a server restart) loads read-only. DB path: `lg_agent_events.db` in the CWD,
override with `LG_AGENT_EVENTS_DB`.
