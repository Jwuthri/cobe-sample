# `lg_agent_v2` — Technical PRD

> **One-liner.** A multi-tenant, streaming multi-agent assistant: a **router** reads the
> conversation and delegates to **worker** sub-agents (agent-as-tool); a dedicated
> **writer** streams the one reply; domain **state** (the cart, for the shopping tenant)
> is the single source of truth. Built on LangChain `create_agent` + LangGraph, ported
> from `pydantic_agent_v1` with the harness swapped.

**Status legend**

| Badge | Meaning |
| --- | --- |
| ✅ **Built** | Implemented in `lg_agent_v2/` today, tested. |
| 🟢 **Built (v3)** | Implemented & live-verified in the sibling package **`lg_agent_v3`** (this doc's design, realized). |
| 🔵 **Lift** | Implemented & working in `lg_agent/core/` — to be lifted for the multi-tenant story. |
| 🟡 **Design** | Proposed in this doc, not yet implemented anywhere. |

> **Where the design lives now.** `lg_agent_v2` is the **bus + two-graph + session-level
> guardrails** baseline. The **unified turn-graph** (§3) and **real per-agent guardrails**
> (§10.6) described here are **implemented and live-verified in `lg_agent_v3`** — including
> the streaming approach §3 flagged as the open risk, now solved (`subgraphs=True` +
> namespace-filtered writer tokens). Sections below are marked 🟢 where `lg_agent_v3`
> realizes them.

---

## 1. Goals & non-goals

**Goals**
- One streaming turn engine that routes a user message to one-or-more sub-agents, streams a single grounded reply, and emits deterministic structured cards ("blocks").
- **Built-in agents** (`checkout`, `order_status`, `product_rec`) shipped as code.
- **Tenant agents defined on the fly** as a pure-JSON `AgentConfig` (no Python) — knowledge-base agents, store-lookup agents, support agents, etc.
- A **capability registry** (tools / middleware / skills / guardrails) so JSON agents reference platform capabilities by name.
- **Full event + state persistence** for debugging, audit, and session reload.
- The cart / domain state survives a server restart (**resume**, not just replay).
- True token streaming; deterministic, hallucination-proof structured cards.

**Non-goals (for v2)**
- Multi-process write concurrency on the event DB beyond SQLite WAL (single-writer).
- Storing raw payment tokens (PII — deliberately reduced to a boolean).
- (Deferred, not precluded) mid-tool-loop crash recovery — the unified turn-graph + a LangGraph checkpointer is the path when it's wanted (§11.4); v2 ships turn-boundary durability first.

---

## 2. Architecture at a glance ✅

Three layers, each with one job. The split is the whole design: the **domain** is the
behavioral spec, the **runtime** is the harness, the **agents** are the minds.

```
                    ┌──────────────────────────────────────────────────────────┐
   user turn ─────► │  turn-graph (StateGraph):  orchestrator → payload →        │
                    │                            writer → blocks → END           │
                    │  every node owns before_agent/after_agent guardrails        │
                    └──────────────────────────────────────────────────────────┘
                        │  orchestrator  ──► delegates to N workers (one tool/step)
                        │  writer        ──► streams the reply, token by token
                        │  blocks + bot  ──► deterministic cards + final message
                        ▼
   ┌───────────────┐  delegate (agent-as-tool)   ┌───────────────────────────┐
   │ orchestrator  │ ──────────────────────────► │ product_rec │ browse + cart│
   │  (router node)│                             │ checkout    │ the purchase │
   │  reads chat,  │                             │ order_status│ past orders  │
   │  resolves refs│ ◄────── terse summary ───── │ <tenant JSON agents…>      │
   └───────────────┘   (block → Command(goto=writer))  nested ainvokes, isolated │
                                                       │ all mutate ▼
   writer node ◄── grounded payload + state ─────  ┌───────────────────────────┐
   (streams; the only node                          │  app_state (Cart / …)     │
    whose tokens reach the user)                     │  domain/ — the spec       │
                                                    └───────────────────────────┘
```

| Layer | Folder | Job | Imports |
| --- | --- | --- | --- |
| **domain** | `lg_agent_v2/domain/` | The store as pure logic: `Cart` + `CheckoutStep` machine + `blockers()`, pricing, catalog, serviceability, orders, memory. No LLM concepts. | nothing |
| **runtime** | `lg_agent_v2/runtime/` | Generic harness: `ShoppingDeps` (shared context), `run_subagent` (agent-as-tool), `StepResult`, model resolution, event vocab, deep-trace, middleware primitives, guardrails. | domain + LangChain |
| **agents** | `lg_agent_v2/agents/` | Five minds, one self-contained file each + `tools.py` + `names.py`. | domain + runtime |

`session.py` + `snapshot.py` tie them into one streaming turn the web UI consumes.

---

## 3. The turn engine — one LangGraph turn-graph 🟡

> **Architecture decision (🟢 built in `lg_agent_v3/graph.py`).** A turn is **one
> `StateGraph`** — the orchestrator and writer are **nodes in the same graph**, not two
> separately-invoked agents. (`lg_agent_v2` is the bus + two-graph baseline; §3.3.)

```
  [orchestrator] ──► [payload] ──► [writer] ──► [blocks] ──► END
   routes to N        builds the    ainvokes;     deterministic cards
   workers (tools);   grounded       tokens        + final {type:"bot"}
   a guardrail block  writer input   surface
   sets refusal       (or refusal)   natively
   sub-agents are nested ainvokes
```

| node | role |
| --- | --- |
| `orchestrator` | `create_agent` subgraph; routes to worker delegate-tools; its guardrails are `before_agent`/`after_agent` middleware (§10.6); a block sets `{"refusal": msg}` in state. |
| `payload` | `build_writer_payload(transcript, steps, cart)` → seeds the writer node's input; promotes any sub-agent guardrail step to the refusal path. |
| `writer` | `create_agent` subgraph; **`ainvoke`** (its tokens surface natively, §3.2); a refusal is emitted verbatim; an `on_output` guardrail → buffered (session suppresses tokens). |
| `blocks` | deterministic `build_blocks(steps, cart)` → emits the final `{type:"bot"}`. |

### 3.1 Why one graph (not two) — what it buys

1. **Guardrail routing is in the graph.** An orchestrator block sets `refusal` in state and
   the linear `payload → writer` path delivers it verbatim — no session-level branching.
   (`Command(goto="writer")` is the native alternative; `lg_agent_v3` uses the simpler
   refusal-flag + linear edges.)
2. **One `astream` carries everything** → the `asyncio.Queue` bus + background task are
   gone. All UI events emit on the custom channel via `get_stream_writer()`.
3. **Checkpointer-ready.** A LangGraph checkpointer over the turn-graph would give native
   mid-turn resume of the *orchestration* (§11.4; not yet wired). Sub-agents stay **nested
   ainvokes inside delegate tools** (not nodes), preserving agent-as-tool + context isolation.

### 3.2 Streaming over one graph (🟢 — the risk is solved)

The session drives the turn-graph with `astream(stream_mode=["custom","messages"],
**subgraphs=True**)`:

```python
async for ns, mode, payload in turn_graph.astream(
        state, context=deps, stream_mode=["custom", "messages"], subgraphs=True):
    if mode == "custom":
        yield payload                                   # router/tool/step/bot/trace/guardrail
    elif mode == "messages":
        chunk, meta = payload
        if ns and ns[0].split(":")[0] == "writer" and isinstance(chunk, AIMessageChunk):
            yield {"type": "token", "content": _chunk_text(chunk)}   # ONLY the writer node
```

- **custom events** (`router`/`tool_start`/`tool_end`/`step`/`bot`/`trace`/`guardrail`) —
  raised anywhere via `deps.emit` → `get_stream_writer()`. **`subgraphs=True` is required**:
  it propagates custom events from *any depth* — including a tool nested inside a
  sub-agent's `ainvoke` (with `subgraphs=False` those are swallowed). This is what replaced
  the bus.
- **message tokens** — every model node streams; emit **only the `writer` node's** by its
  subgraph **namespace** (`ns[0].split(":")[0] == "writer"`), dropping the orchestrator's
  and sub-agents' model chatter. So the writer node just `ainvoke`s — its tokens surface
  natively, no manual re-pump. Empty-retry is a small `ainvoke`-twice loop in the node
  (stream-safe: an empty attempt emits nothing).

> ✅ **Resolved.** The "identify the writer node in nested-subgraph metadata" risk this doc
> originally flagged is solved by `subgraphs=True` + the `ns[0]` namespace prefix. Live
> check (`lg_agent_v3`): full checkout streams, **`"".join(tokens) == bot` every turn**
> (the filter captures exactly the writer's tokens), total `$74.35`. One caveat surfaced:
> under the parent's forced message-streaming, *test fakes* need a `_stream` that yields the
> scripted message as one chunk (word-chunking emits nothing for an empty tool-call message).

### 3.3 Baseline — `lg_agent_v2` (the bus + two graphs)

`lg_agent_v2/session.py` runs the orchestrator in `asyncio.create_task` draining an
`asyncio.Queue` bus, then invokes the writer as a **separate** graph via
`writer.astream(stream_mode="messages")`. It is a verbatim port of `pydantic_agent_v1` and
is functionally equivalent for the happy path; `lg_agent_v3` is the consolidation into one
graph with graph-native streaming + per-agent guardrails.

> 📌 **Event ordering under parallel delegation (both designs).** `run_subagent` does its
> single `await worker.agent.ainvoke(...)`, then flushes *all* of that worker's events
> synchronously (no `await`/yield between emits). So each worker's event burst is atomic
> and lands in completion order; parallel workers never interleave mid-burst.

### 3.4 Transcript representation

The session keeps `transcript: list[dict]` of `{role, content, blocks}` — **not** native
LangChain messages. The orchestrator's history is built fresh each turn via
`_to_message_history(transcript)` → `[HumanMessage|AIMessage(text-only)]`. This keeps the
router prompt clean (no tool-call noise in history) and makes the transcript trivially
serializable (→ persistence + resume, §11).

---

## 4. The orchestrator (router) — `agents/orchestrator.py` ✅

The **sole reader of the conversation**. It resolves references → concrete ids, routes
each distinct request to exactly one worker, then emits `DONE`. It never writes the
user-facing reply.

```python
orchestrator = create_agent(
    model=build_model(0.0),
    tools=[_product_rec, _checkout, _order_status],   # delegate tools (agent-as-tool)
    system_prompt=ROUTER_PROMPT,
    context_schema=ShoppingDeps,
    middleware=[
        dynamic_instructions(build_memo),     # reference memo (live state + recalls)
        hide_tool(CHECKOUT, _cart_empty),     # empty-cart guard (structural)
        no_parallel_tools(),                  # one worker per model step
    ],
    name="orchestrator",
)
```

### 4.1 Reference resolution (`build_memo`)

The orchestrator is the only place that interprets "the green one" / "add it" / "make it
2". `build_memo(deps)` renders a memo from two domain-agnostic sources, **never raw chat**:
- `deps.routing_context()` — live structured state (the current cart; plus, mid-checkout, the exact step so a terse "2h"/"cash" routes to checkout not smalltalk).
- `deps.routing_notes` — per-sop `recall` snippets a worker surfaced in a *previous* turn (e.g. product_rec's "recently shown products").

`absorb_recalls(deps)` persists this turn's `recall`s back into `routing_notes` after the
orchestrator phase, so next turn's memo can resolve references — **without the workers
ever seeing the chat**.

### 4.2 The guards (middleware)

| Guard | Primitive | Effect |
| --- | --- | --- |
| Reference memo | `dynamic_instructions(build_memo)` | Appends a transient `SystemMessage(memo)` to the model request each call. |
| Empty-cart guard | `hide_tool(CHECKOUT, _cart_empty)` | Drops the `checkout` delegate from the tool list while the cart is empty → "add X" can't misroute to checkout. Structural, re-evaluated every model call. |
| Single tool/step | `no_parallel_tools()` | Sets `parallel_tool_calls=False` (forwarded to `bind_tools`) so a compound message routes one worker per step → ordered events, race-free shared state. |
| **Guardrails** | `before_agent`/`after_agent` middleware | The orchestrator owns its guardrails like every agent (§10.6). On a block it returns `Command(goto="writer", update={refusal})` — the turn jumps to the writer node, which delivers the refusal. |

> ⚠️ **Gotcha — caching.** The reference memo embeds the live cart, so it changes most
> turns. It is **appended at the very end** of the message list, never the front — so the
> stable prefix (`system + tools + history`) stays prompt-cacheable. See §8.2 for why
> "append, never splice."

---

## 5. The workers & agent-as-tool — `runtime/delegation.py` ✅

The orchestrator's "tools" **are** the worker agents. `run_subagent` is the single
wrapper written once:

```python
async def run_subagent(deps, worker, query) -> str:
    deps.emit(events.router(worker.name))                # 1. announce routing
    before = worker.snapshot(deps) if worker.snapshot else None   # 2. pre-run snapshot
    result = await worker.agent.ainvoke(                  # 3. run on the isolated query,
        {"messages": [HumanMessage(content=query)]}, context=deps)  #    SHARED context=deps
    for ev in _tool_events(result["messages"]):          # 4. inner tool calls → UI rows
        deps.emit(ev)
    sr = worker.extract(deps, result["messages"], before) # 5. distill a StepResult
    deps.steps.append(sr); deps.emit(events.step(sr))
    summary = worker.summarize(sr, deps) if worker.summarize else sr.summary
    deps.emit(events.agent(f"{worker.name}_wrapper"))
    return summary                                        # 6. TERSE summary — all the LLM reads
```

### 5.1 Context isolation (the core invariant)

A worker **never sees the conversation**. It receives one self-contained `query` (the
orchestrator already resolved references → concrete ids) + the **shared `deps`** (one live
cart). A worker is a clean function of `(query + state)`. This:
- keeps interpretation in one place (the router),
- cuts tokens and shrinks the prompt-injection surface,
- means **sub-agents carry no independent state** — the cart *is* their state. (Critical
  for resume, §11: there is no per-sub-agent state to restore — restoring the shared
  `app_state` restores all of them.)

### 5.2 `Worker` + hooks

```python
@dataclass(frozen=True)
class Worker:
    name: str           # delegate-tool name AND StepResult.sop
    agent: Any          # compiled create_agent graph
    extract: Extractor  # run messages → StepResult (REQUIRED)
    prompt: str = ""    # static instructions (debug trace)
    block: str | None = None     # writer block kind this worker produces
    snapshot: Snapshotter | None = None  # pre-run state for diffing
    summarize: Summarizer | None = None  # default: sr.summary
```

`extract` parses the run's `ToolMessage`s (via `tool_returns`) + diffs `before`/`after`
state into a `StepResult`. `tool_returns`/`_tool_events` parse LangChain message types
(`AIMessage.tool_calls`, `ToolMessage`) — version-stable, no reliance on internal stream
events.

### 5.3 `StepResult` — the two-audience contract

```python
class StepResult(BaseModel):
    sop: str                  # which worker ran
    summary: str = ""         # terse — the ONLY thing the orchestrator reads back
    asks: list[str] = []      # what the user still needs to provide
    next_sop: str | None = None   # soft routing hint (e.g. product_rec → checkout)
    details: dict | None = None   # grounded facts → deterministic blocks + writer
    recall: str | None = None     # private to the orchestrator (reference resolution)
```

- **orchestrator** reads only `summary` (so it can't hallucinate ids/totals it never saw).
- **writer + block builder** read `details` (the grounded facts → cards).
- `recall` is the orchestrator's cross-turn memory, never shown to the writer.

---

## 6. The writer — `agents/writer/` ✅

The single customer-facing voice and the **last node** of the turn-graph. **No tools**,
temperature 0.3. It composes prose **only** and is the only node whose model tokens reach
the user (§3.2).

### 6.1 Grounded payload (`writer/payload.py`)

`build_writer_payload(transcript, steps, cart) → (payload_json, mode)`. The writer
composes from verified facts, never raw tool output. `pick_mode` chooses presentation:

| mode | when | payload carries |
| --- | --- | --- |
| `smalltalk` | no worker ran | recent conversation only |
| `info` | product_rec / order_status ran | `step_results[*].details` + a minimal cart summary if non-empty |
| `checkout` | checkout ran | a full checkout cart summary (items, subtotal, shipping/tax if fresh, grand_total, **actionable** blockers, `ready_to_confirm`, `confirmed`, `receipt_id`) |
| `refusal` | a guardrail blocked upstream | the refusal `message` (+ reason) so the writer delivers ONE consistent refusal — see §10.6 |

`recall` is excluded from `step_results` in the payload (orchestrator-private). The writer
prompt forbids claiming an order is placed unless `cart.confirmed` is true.

### 6.2 Streaming, retry & its own guardrails

Tokens stream straight to `{type:"token"}` (the writer node, filtered out of the unified
stream). Empty output retried once (stream-safe). The writer is grounded at *construction*,
so there is no post-generation validator gating the stream.

> ⚠️ **Writer `on_output` guardrails → buffered.** You can't stream tokens then retract
> them. In `lg_agent_v3` the writer node always `ainvoke`s and its tokens surface via the
> parent stream (§3.2); when `writer_buffered` is set (the writer has an `on_output`
> guardrail), the **session suppresses token forwarding** that turn, so the client gets only
> the final, `after_agent`-scrubbed reply via the `{type:"bot"}` event. (A manual flag
> today — auto-detecting it from the writer's guardrail config is a small follow-up.)

---

## 7. Blocks — the hallucination firewall ✅ / 🟡

Prose and structured cards are **two independent outputs**, joined only at the final
`bot` event. The model writes the sentence; **deterministic code writes the id/total.**

### 7.1 How it works today (built-in workers)

```python
# session.py phase 4 — AFTER the writer streams:
blocks = build_blocks(deps.steps, self.cart_service.cart, BLOCK_BY_SOP)
yield events.bot(text, blocks)     # text from LLM, blocks from code
```

`build_blocks` is a lookup, not a decision:
- `BLOCK_BY_SOP = {w.name: w.block for w in WORKERS}` → `{product_rec:"product_reco", checkout:"checkout", order_status:"order_status"}`. The block kind is **fixed per worker**.
- For each `StepResult`, it maps `sr.sop` → kind, then fills a typed pydantic card (`ProductRecoBlock` / `OrderStatusBlock` / `CheckoutBlock`) from `sr.details` + the **live cart** — ids/prices/totals copied verbatim.

So the writer can't hallucinate a card field — it doesn't write the cards. The writer
prompt only introduces them in prose ("Here are the hoodies:") without re-dumping fields.

### 7.2 Blocks for on-the-fly JSON agents 🟡

A tenant JSON agent ships no Python `extract` and no bespoke card. **`output_format` is
the block contract.** A declarative agent that wants a card declares `output_format`
(a JSON-schema object); `create_agent(response_format=…)` forces the model's structured
output into `result["structured_response"]`. The generic path:

```python
# generic extract for ALL declarative agents — no per-agent Python
def generic_extract(deps, result, cfg) -> StepResult:
    structured = result.get("structured_response")
    return StepResult(
        sop=cfg.name,
        summary=(structured or {}).get("summary") or _last_ai_text(result),
        details={"structured": structured} if structured else None,
        recall=(structured or {}).get(cfg.recall_field) if cfg.recall_field else None,
    )

# generic block — one extra kind the frontend renders generically
def custom_block(sr):
    data = (sr.details or {}).get("structured")
    return {"kind": "structured", "agent": sr.sop, "data": data} if data else None
```

- JSON agent **with** `output_format` → `{"kind":"structured","agent":<name>,"data":<schema object>}` — frontend renders generic key/value (or a `display_hint` field the tenant adds to its schema).
- JSON agent **without** `output_format` → prose only, no block.

`BLOCK_BY_SOP` extends to `{built-in names → typed kind, custom names → "structured"}`.
The firewall property holds: the structured card comes from the model's *constrained*
output, not free text.

---

## 8. The shared context & middleware primitives ✅

### 8.1 `ShoppingDeps` → generic `TurnContext` (🟡 rename for multi-tenant)

`ShoppingDeps` is the LangChain **context** (`context_schema=ShoppingDeps`): one instance
per turn, forwarded by reference into the orchestrator and every nested
`worker.agent.ainvoke(..., context=deps)`, so one live `CartService` is mutated in place
across the whole turn — no copy-back.

For the multi-tenant story, generalize it (🟡):

```python
@dataclass
class TurnContext:
    # platform plumbing — always present
    user_id: str; session_id: str
    bus: asyncio.Queue | None = None      # UI event sink (shipped); under the unified
                                          # turn-graph this is dropped → get_stream_writer (§3.1)
    steps: list[StepResult] = field(default_factory=list)
    routing_notes: dict[str, str] = field(default_factory=dict)
    debug: bool = False
    # facts about the human — generic, cross-tenant, all optional
    user: UserProfile = field(default_factory=UserProfile)   # id, location?, timezone?, locale?
    # the tenant's domain object — ONE opaque slot
    app_state: Any = None                 # CartService for shopping; a KB handle / None otherwise
    def routing_context(self) -> dict[str, str]: return {}   # tenant overrides
```

> 📌 **Three kinds of state, three homes.** User facts (location/timezone/locale) → `user`,
> read by any tenant. The tenant's domain object (cart, KB index) → `app_state`, `Any`,
> read only by that tenant's tools. Platform plumbing → fixed fields. **Don't grow the
> cart with location/tz** — those aren't cart concerns.

### 8.2 The three middleware primitives — `runtime/middleware.py`

These replace Pydantic AI natives. Each implements both `wrap_model_call` and
`awrap_model_call`.

| Primitive | Replaces | Mechanic |
| --- | --- | --- |
| `dynamic_instructions(fn)` | `@agent.instructions` | **Appends** `SystemMessage(fn(ctx))` at the end of `request.messages` each model call. |
| `hide_tool(name, pred)` | tool `prepare=` | Removes a tool from `request.tools` while `pred(ctx)` holds. |
| `no_parallel_tools()` | `parallel_tool_calls=False` | Merges `{"parallel_tool_calls": False}` into `request.model_settings` (→ `bind_tools`). |

> ⚠️ **Gotcha #1 — append, never splice.** A dynamic note must be appended at the very
> **end** of the message list. The model node always runs with the last message either a
> `HumanMessage` (turn start) or a `ToolMessage` (mid tool-loop). Inserting a
> `SystemMessage` *before* a trailing `ToolMessage` splits it from its
> `AIMessage(tool_calls)` → provider 400 *"tool_call_ids did not have response messages"*.
> Appending is valid in both cases **and** maximally cache-friendly. (Regression:
> `tests_lg_agent_v2/test_session.py::test_dynamic_note_never_splits_a_tool_call_pair`.)

> ⚠️ **Gotcha #2 — serial mutators, parallel readers.** Cart-mutating workers
> (`product_rec`, `checkout`) carry `no_parallel_tools()`: a compound "add X and Y" must
> run one tool per step — parallel `add_item` both races the shared cart list *and* leaves
> an unanswered `tool_call_id`. Read-only workers (`order_status`) stay parallel so
> "status of order 1 and order 2" fans out. When the orchestrator itself fans out to
> independent **read-only** sub-agents (e.g. refund-lookup + status), keep them parallel;
> only guard the mutating ones (a `threading.Lock` on `app_state` for mutators is the
> escape hatch if two mutating sub-agents can ever co-occur).

---

## 9. Built-in agent registry ✅

`checkout`, `order_status`, `product_rec` ship as **Python-hook workers** (`Worker`
dataclass: a compiled `create_agent` + `extract`/`snapshot`/`summarize` hooks + a fixed
`block` kind). They are the "defaults" every tenant gets. The registry is just:

```python
WORKERS = [product_rec.WORKER, checkout.WORKER, order_status.WORKER]   # routing-priority order
BLOCK_BY_SOP = {w.name: w.block for w in WORKERS}
```

`build_orchestrator(model=None, worker_agents=None)` wraps each into a delegate tool;
`worker_agents` lets tests (and tenants) swap a worker's compiled agent. Its name +
`description` are the **routing surface** the orchestrator sees.

### The star: `checkout` (`agents/checkout.py`)

Checkout never has to *remember* — the cart is the truth:
- **The progress anchor.** `dynamic_instructions(checkout_progress)` injects a
  deterministic "what's done / what's next" block (rendered from `Cart.step`) on every
  model call.
- **The blocker gate.** `confirm_checkout` refuses while `cart.blockers()` is non-empty —
  the model cannot place an incomplete order. `cart.confirmed` (not prose) is the source
  of truth that an order was placed.
- **No `add_item` tool** here (double-add structurally impossible); a quantity edit trips
  the shipping/tax freshness fingerprint → `awaiting_pricing` → recompute, so a stale total
  can't be confirmed.

---

## 10. On-the-fly sub-agents via JSON 🔵 (exists in `lg_agent/core`, to lift)

A tenant defines an agent as a pure-JSON `AgentConfig` — **no Python**. This builder
already exists and works in `lg_agent/core/`; the v2 task is to lift it next to the
flat runtime and route declarative agents through `run_subagent`'s generic hooks (§7.2).

### 10.1 `AgentConfig` schema (`lg_agent/core/config.py`, Pydantic v2, `extra="forbid"`)

```jsonc
{
  "name": "Acme Support",                       // required, min_length 1
  "description": "Customer support agent.",      // routing surface (the ONLY routing signal)
  "system_prompt": "You are a friendly…",        // required, min_length 1
  "instructions": ["Be concise.", "English."],   // appended as "## Additional instructions" bullets
  "model": {"provider_model": "openai:gpt-5-mini", "temperature": 0.0, "max_tokens": null},
  "skills":  [ {"kind":"custom", "name":"checkout_flow", "description":"…", "skill":"long text"},
               {"kind":"registry", "name":"some_registered_skill"} ],
  "tools":   [ {"kind":"registry", "name":"check_order_status"},
               {"kind":"http", "name":"create_zendesk_ticket", "method":"POST",
                "url":"https://acme.zendesk.com/api/v2/tickets",
                "headers":{"Authorization":"Bearer {api_token}"},
                "parameters":{"type":"object","properties":{
                    "api_token":{"type":"string"}, "subject":{"type":"string"}},
                  "required":["subject"]}} ],
  "guardrails": [ {"type":"pii","action":"redact","on_input":true,"params":{"entity":"email"}},
                  {"type":"blocklist","action":"block","on_input":true,"message":"…",
                   "params":{"phrases":["sue","lawsuit"]}},
                  {"type":"llm_judge","action":"block","on_input":true,"message":"…",
                   "params":{"policy":"Do not answer anything about X.","model":"openai:gpt-5-nano"}} ],
  "middleware": [ {"name":"model_call_counter","params":{}},
                  {"name":"max_turns","params":{"max_turns":30}} ],
  "output_format": { "type":"object",            // → response_format → structured card (§7.2)
    "properties": {"summary":{"type":"string"}, "status":{"enum":["open","resolved"]}},
    "required": ["summary","status"] }
}
```

**Validation (fails at parse, not runtime):** every model is `extra="forbid"`;
`provider_model` must contain `":"`; HTTP-tool `{placeholders}` in url/headers must be
declared in `parameters.properties`; `output_format` must be `{"type":"object","properties":…}`;
tool/skill names unique.

### 10.2 `build_agent` — config → `create_agent` (`lg_agent/core/builder.py`)

```python
def build_agent(config, *, context_schema=None, checkpointer=None, store=None,
                name=None, delegates=None):
    cfg = to_config(config)                                   # dict → validated AgentConfig
    tools = resolve_tools(cfg.tools) + list(delegates or [])  # registry/http tools + sub-agents
    return create_agent(
        model=resolve_model(cfg.model),                       # init_chat_model(provider:model, temp, max_tokens)
        tools=tools,
        system_prompt=_compose_prompt(cfg),                   # system_prompt + "## Additional instructions"
        response_format=cfg.output_format,                    # structured output → blocks
        middleware=_compile_middleware(cfg),                  # [skills?, *guardrails, *middleware]
        context_schema=context_schema, checkpointer=checkpointer, store=store,
        name=name or _slug(cfg.name),
    )
```

Middleware order is `[SkillsMiddleware?, *guardrails, *named-middleware]` (skills first so
their block is visible). `delegates` are sub-agents-already-wrapped-as-tools (carry Python
hooks → not pure config) — this is how an **orchestrator** is itself a `build_agent` with
the workers passed as `delegates`.

### 10.3 The capability registry (`lg_agent/core/registry.py`)

One generic `Registry[T]` backs four singletons. A tenant **populates them once at import**
(e.g. `lg_agent.shopping.setup.register_shopping()`); JSON configs reference capabilities
by **name**, resolved at build time.

| Registry | Key | Stored value | Resolved from |
| --- | --- | --- | --- |
| `TOOLS` | tool name | concrete LangChain tool (must expose `.name`) | `{"kind":"registry","name":…}` → `TOOLS.get(name)` |
| `MIDDLEWARE` | name | factory `(**params) → AgentMiddleware` | `MiddlewareSpec.name` |
| `SKILLS` | name | a `Skill` | `RegistrySkillSpec.name` |
| `GUARDRAILS` | `type` | factory `(GuardrailSpec) → AgentMiddleware` | `GuardrailSpec.type` |

API: `register(name, item, *, replace=False, **meta)` (duplicate raises unless
`replace=True`), `get(name)` (missing raises, lists available), `has`, `names()`,
`catalog()` (introspection for a builder UI). `register_tool(tool, …)` derives the key
from `tool.name`.

> **So "use the agent registry such as checkout / order_status":** the *built-in workers*
> are Python-hook `Worker`s (§9), wired as delegates. *Tenant tools* (and the leaf tools
> the built-ins use) are registered in `TOOLS` and referenced by JSON agents via
> `{"kind":"registry","name":…}`. Both kinds of agent become delegate tools on the same
> orchestrator; routing is by name + description.

### 10.4 Declarative HTTP tools (`lg_agent/core/tools.py`)

`compile_http_tool(spec) → StructuredTool`. Mechanics:
- `{placeholder}` tokens in `url` + header values are filled from the model's args via
  `template.format(**kwargs)` at call time.
- **Secret-stripping (load-bearing):** every kwarg that fed a url/header placeholder is
  removed from the request body — a token passed as `Authorization: Bearer {api_token}`
  lands only in the header, never the payload.
- `GET` → remaining args become query params; other verbs → JSON body. `args_schema` is
  the raw `parameters` JSON-schema dict. `raise_for_status()` then return `response.text`.
- Auth is just templating: the model supplies the token as a declared arg; there is no
  server-side secret injection here (🟡 a tenant-secret-injection layer is future work).

### 10.5 Skills (`lg_agent/core/skills.py`) — two-channel, on-demand

- **Channel A (always-on):** `SkillsMiddleware` prepends an "Available skills" block (one
  line per skill: `name + description`, with a `(loaded)` marker) — transient, every call.
- **Channel B (on-demand):** a `load_skill(skill_name)` tool injects the skill's full
  `content` as a `ToolMessage` (persists in history once loaded) and writes
  `skills_loaded += [name]` (an append-without-dup state reducer). The model loads a skill
  only when relevant → cheap by default, deep when needed.

### 10.6 Guardrails — owned by the agent, not the session 🟢

> **🟢 Built in `lg_agent_v3/runtime/guardrails.py`** (`blocklist` / `pii` / `llm_judge` as
> `before_agent`/`after_agent` middleware; `compile_guardrails(specs, agent_name)`;
> `redact_input` for the session-level sanitizer). Block delivery in v3: an orchestrator
> block sets `refusal` in graph state → writer delivers verbatim; a sub-agent block is
> caught in `run_subagent` (it snapshots `len(deps.guardrail_hits)` around the ainvoke) and
> returned as a flagged `StepResult(details={"guardrail": msg})` → the payload node promotes
> it to the same verbatim-refusal path. Middleware records a `GuardrailHit` + emits a
> `{type:"guardrail"}` event.

> **Architecture decision.** Guardrails are a **per-agent capability**: every agent
> (orchestrator, every sub-agent, the writer) declares its own `guardrails: [...]` in its
> config, and `build_agent` compiles them into `before_agent`/`after_agent` middleware on
> **that** agent's graph. There is no session-level guardrail *phase* — the session keeps
> only an input **redactor** (see below). This is the only model where a guardrail like
> *"the support agent may not give legal advice"* actually works: it's an **output**
> guardrail on the support sub-agent, detectable only after that agent generates —
> something a pre-flight pass on the user's input fundamentally cannot do.

**Why `before_agent`/`after_agent` (not `before_model`/`after_model`).** These hooks fire
**once per agent run** — one input check on entry, one output check on exit — instead of
re-firing on every model call inside the agent's tool loop. (Across *repeated* delegations
in a turn they still re-run, which is correct: different input + different output each call.)

**The three rule types** (`compile_guardrail_middleware(cfg.guardrails)` → middleware):

| type | input side | output side | notes |
| --- | --- | --- | --- |
| `blocklist` | phrase/regex match → refusal | replace offending `AIMessage` by id | `action` stored; behavior hardwired to block/replace |
| `llm_judge` | structured-output judge vs `params.policy` (own `model`, `tags=["nostream"]`) → refusal | replace by id | **fails open** on judge error (availability) |
| `pii` | langchain `PIIMiddleware`; `action` = strategy (block/redact/mask/hash) | same | `block` raises `PIIDetectionError` |

**How a block is delivered (the two flows):**

```
orchestrator guardrail fires
   → before_agent returns Command(goto="writer", update={refusal})   (native, one graph — §3)
   → the writer node delivers the refusal (mode="refusal")

sub-agent guardrail fires (input OR output)
   → the sub-agent run ends in a refusal; run_subagent flags it:
        StepResult(sop=worker, summary="[GUARDRAIL] <msg>", details={"guardrail": msg})
        return "GUARDRAIL_BLOCK: <msg>"      ← the orchestrator reads this as the tool result
   → it rides the normal agent-as-tool → step_results → writer pipeline; the writer relays it
```

The guardrail middleware *signals* (records a hit on `deps`/context + emits a
`{type:"guardrail", stage:"<agent>:<side>", rule, action}` event), it does **not** deliver
the message itself — so delivery stays in one place (the writer), with one consistent
voice. (Use verbatim delivery instead of writer-wrapping for policy-exact refusals.)

**The only session-level residue: input redaction.** Redaction must rewrite the user text
*before* it enters the canonical `transcript` (which is persisted to SQLite and handed to
sub-agents as their `query` + to the writer). An in-graph redactor would only clean a
model's transient copy, leaving raw PII on disk + downstream. So a tiny session step
sanitizes the input before the transcript append — input preprocessing, not a guardrail
phase:

```python
self.transcript.append({"role":"user", "content": redact(user_text)})   # clean at the source
```

> The v2 runtime today still ships the simple session-level `Blocklist`/`PiiRedact`
> pre-flight (`runtime/guardrails.py`). Lifting `lg_agent/core` brings `llm_judge`, the
> `PIIMiddleware` strategies, and **per-agent before/after_agent** wiring, after which the
> session pre-flight collapses to the redactor above. 🔵→🟡

---

## 11. Persistence, debugging & resume

### 11.1 What's stored — `SQLiteEventStore` ✅ (wired in the v2 server)

The server tees every turn's UI events to SQLite (`lg_agent.core.event_store`, already
imported by `server/main_lg_agent_v2.py`). **Three tables:**

```sql
sessions (session_id PK, user_id, created_at, last_seen)
events   (id PK AUTOINCREMENT, session_id, user_id, turn, seq, type, ts, data)   -- data = json(event)
snapshots(id PK AUTOINCREMENT, session_id, turn, ts, cart, snapshot)             -- one row per "state" event
```

- **`events`** = the complete verbatim UI stream, append-only, ordered by `id` (= turn
  order = within-turn `seq`). Every event of every type — `user`, `state`, `router`,
  `tool_start/end`, `step`, `token`, `writer`, `bot`, `guardrail`, `error`, `end`, and
  (when `debug=True`) deep-trace frames — is one row, `data` = the full event JSON. **This
  is the debugging goldmine:** you can replay exactly what the UI saw, including the
  internal `trace` frames (orchestrator input, sub-agent in/out, writer payload).
- **`snapshots`** = a derived projection: every `state` event's full snapshot exploded into
  `cart` (cart sub-dict, for cheap "cart at turn N") + `snapshot` (full dict). A normal
  turn emits ~3 `state` events (start / post-orchestrator / end); the **last** snapshot of a
  turn (highest `id`) is the authoritative post-turn state.

**Write path:** `record_turn(session_id, user_id, turn, rows)` — one transaction per turn,
flushed in a worker thread in `run_turn_stream`'s `finally`, wrapped in `try/except: pass`
(persistence can never break a turn). **Granularity = one batch per turn at turn end** → a
hard crash mid-turn loses that turn's events (nothing partial is written).

**Snapshot shape** (`snapshot.py`, what each `state` event carries and what's denormalized):
```jsonc
{ "user_id", "session_id", "active_sop": null, "skills_loaded": [...],
  "cart": { "step", "cart_id", "items":[{id,name,qty,unit_price,line_total,tags}],
            "customer", "address", "serviceable", "serviceable_options", "delivery_option",
            "shipping": {cost,eta_hours}|null, "tax": {amount,rate}|null, "promo": …|null,
            "payment_method", "card_token_set": <bool — raw token NEVER stored>,
            "subtotal", "grand_total", "blockers":[{code,message}],
            "ready_to_confirm", "confirmed", "receipt_id" },
  "messages": [ {"role","content","blocks":[…]} ], "iteration": 0, "done": true }
```

### 11.2 Debugging endpoints ✅

`GET /api/sessions` (list + `events`/`turns` counts + `live`), `GET /api/events/{id}`
(replay the verbatim stream), `GET /api/snapshots/{id}` ("state at turn N"),
`GET /api/state/{id}` (in-memory only).

### 11.3 Reload — the honest current state ⚠️

> 📌 **Today, "reload" is UI-level replay, NOT server-side session resume.** There is no
> `from_events` / `from_snapshot` / `rehydrate` constructor anywhere. The frontend's "load
> previous session" re-runs every stored event through the *same projection the live stream
> used* and takes the last `state.snapshot` for the cart panel — read-only. `live=false`
> (session not in the in-memory `SESSIONS` dict) → the loaded view **cannot be chatted to**.
> Posting a turn to an archived id silently starts a **fresh empty** `ShoppingSession`.
>
> (This is also why the recurring "ghost 400" was a *replayed* old event, not a live
> failure — the DB faithfully re-shows a past broken turn.)

| After a server restart | Status |
| --- | --- |
| View any past session's transcript + cart panel, in order, original timestamps | ✅ reconstructable (read-only) |
| "Cart at turn N" / "state at turn N" | ✅ (snapshots table) |
| Continue chatting in a past session (warm `ShoppingSession`) | ❌ not implemented |
| `routing_notes` (reference-resolution memory) | ❌ never persisted (no column) |
| Raw `card_token` | ❌ by design (only `card_token_set` bool) |
| In-flight mid-tool-loop graph state | ❌ not checkpointed |

### 11.4 Resume design — "reload the agent & each sub-agent to their state" 🟡

**Key simplifier from §5.1: sub-agents carry no independent state.** They are pure
functions of `(query + app_state)`. So "reload each sub-agent to its state" reduces to
**reload the shared `app_state` + the transcript** — there is nothing per-sub-agent to
restore. Concretely, add a rehydrate path:

```python
@classmethod
def from_snapshot(cls, snap: dict, *, routing_notes: dict | None = None, **kw) -> "ShoppingSession":
    s = cls(user_id=snap["user_id"], session_id=snap["session_id"], **kw)
    s.transcript = snap["messages"]                    # already {role,content,blocks} — 1:1
    s.cart_service = CartService(Cart.from_view(snap["cart"]))   # rebuild Cart from _cart_view
    s.turn = <max turn from events>
    s.routing_notes = routing_notes or {}              # re-derive from persisted step events (optional)
    return s
```

Why this is clean for v2:
- The **transcript** is stored 1:1 (`snapshot.messages` == v2's `transcript` shape); the
  orchestrator rebuilds its message history from text each turn anyway, so nothing native
  is lost.
- The **cart** is fully in `snapshot.cart` (`step` is derived, so it self-heals). Add a
  `Cart.from_view(cart_dict)` reconstructor (inverse of `_cart_view`).
- `_get_or_create` (server) becomes: in-memory hit → use it; miss → `from_snapshot(last
  state snapshot for that id)` so the session is **resumable**, not blank.

**Two complementary persistence layers (with the unified turn-graph, §3):**

| Layer | Persists | Mechanism | Resume granularity |
| --- | --- | --- | --- |
| **Checkpointer** | the turn-graph state (messages, routing progress, mid-tool-loop position) | `StateGraph(checkpointer=SqliteSaver)`, `thread_id = session_id` | **mid-turn** — a crashed turn resumes where it stopped |
| **app_state snapshot** | the **cart** (the domain object) | the `state`-event snapshot (§11.1) → `Cart.from_view` | **turn boundary** |

> ⚠️ **The checkpointer does NOT persist the cart.** The cart lives in the LangGraph
> **context** (`deps.app_state`), which is per-invocation runtime — *not* a checkpointed
> state channel. So the unified graph + checkpointer gives native resume of the
> *conversation/turn graph*, but the cart still needs its own persistence (the snapshot, or
> promote the cart into a reduced state channel). Net resume = **checkpointer (graph) +
> app_state snapshot (cart)**.

Gaps to decide (🟡):
- **`routing_notes`** — give it a column in the `state` snapshot (cheapest), or re-derive
  by scanning persisted `step` events for `recall`. Losing it only degrades reference
  resolution for one turn.
- **`card_token`** — never stored (PII). On resume, a card cart re-shows the
  `missing_card_token` blocker → re-collect, or store an encrypted token-ref out of band.
- **Where the cart should live** — keep it in context (`app_state`, shared-by-reference
  across nested sub-agent ainvokes, persisted via the snapshot) OR promote it into a
  checkpointed state channel (auto-persisted, but you lose the mutate-by-reference model
  that agent-as-tool relies on). The snapshot path is simpler and is the v2 default;
  revisit only if turn-boundary cart durability proves insufficient.

---

## 12. Model & config resolution ✅

- Per-agent model via `build_model(temperature)` → `init_chat_model(MODEL_NAME, temperature=…)`.
- `MODEL_NAME` env chain: `LG_AGENT_V2_MODEL` → `AGENT_V2_OPENAI_MODEL` → `openai:gpt-5.4-mini` (bare id assumed OpenAI).
- Temps: orchestrator/workers `0.0` (deterministic routing), writer `0.3` (warmth).
- **Prompt-cache strategy:** keep the prefix (`system + tools + history`) stable; put all
  volatile per-turn content (the reference memo, the checkout anchor) in the **appended
  tail** (§8.2). Keep per-session tool sets stable where possible (the empty-cart guard is
  a one-transition-per-session toggle, acceptable; per-turn tool churn for many tenant
  tools would not be — vary by prompt, not by hiding).

---

## 13. Testing strategy ✅

26 → **29 offline tests** (`tests_lg_agent_v2/`), zero real LLM calls:
- Agents driven by `ToolCallingFake` (a `GenericFakeChatModel` with a `bind_tools` no-op) scripting `AIMessage`s; the writer streams via the same fake.
- `make_session(orchestrator=…, product_rec=…, writer=…)` rebuilds agents with fakes and injects them — the LangChain analogue of pydantic's `.override`.
- Pure-logic tests (domain / blocks / checkout-progress / guardrails) are harness-agnostic.
- Regression guards from real bugs: `test_dynamic_note_never_splits_a_tool_call_pair` (the append-not-splice 400), `test_cart_mutating_workers_disable_parallel_tool_calls`, `test_routing_memo_is_appended_in_the_cacheable_tail`.

---

## 14. Wire contract / API ✅

Same SSE/JSON contract as every sibling engine (the web UI is unchanged):
- `POST /api/session` → `{session_id}`
- `GET  /api/state/{id}` → the `AgentSnapshot` (cart + messages)
- `POST /api/turn/{id}` → SSE stream of events (incl. live writer tokens)
- (+ debugging: `/api/sessions`, `/api/events/{id}`, `/api/snapshots/{id}`)

Server: `uvicorn server.main_lg_agent_v2:app --reload --port <PORT>`; point the frontend
with `AGENT_V2_API_URL`.

---

## 15. Build sequence (what to do, in order)

| # | Work | Status |
| --- | --- | --- |
| 1 | Flat 3-layer runtime + 5 built-in agents + bus streaming + blocks (`lg_agent_v2`) | ✅ done |
| 2 | SQLite event/snapshot tee + debug endpoints | ✅ wired |
| 5 | **Per-agent guardrails** — `before_agent`/`after_agent` middleware on orchestrator + every sub-agent + writer; session keeps only the input redactor (§10.6) | 🟢 v3 |
| 6 | **Unify orchestrator + writer into one `StateGraph`** turn-graph; graph-native streaming via `astream(stream_mode=["custom","messages"], subgraphs=True)` + namespace-filtered writer tokens; bus dropped (§3) | 🟢 v3 |
| 3 | Generalize `ShoppingDeps` → `TurnContext` (`app_state` + `user`) | 🟡 |
| 4 | Lift `lg_agent/core` declarative layer (config/builder/registry/tools/guardrails/skills) next to the runtime | 🔵→lift |
| 7 | LangGraph checkpointer on the turn-graph (`thread_id = session_id`) → native mid-turn resume; pairs with the app_state snapshot (§11.4) | 🟡 |
| 8 | Generic `extract` + `custom_block` for declarative agents (`output_format` = the block contract) | 🟡 |
| 9 | Register built-in leaf tools + tenant tools in `TOOLS`; route JSON agents as delegates | 🟡 |
| 10 | `Cart.from_view` + `ShoppingSession.from_snapshot` + server resume in `_get_or_create` | 🟡 |
| 11 | (optional) `routing_notes` persistence; tenant-secret injection for HTTP tools | 🟡 |

> Steps 5–6 (the graph + real guardrails) shipped in **`lg_agent_v3`** (port 8007, 33
> tests, live-verified `$74.35`). Steps 3–4, 7–11 remain open.

---

## Appendix A — file map

```
lg_agent_v2/
├── domain/        cart.py ★ · cart_service.py · catalog/pricing/serviceability/orders/memory.py
├── runtime/       deps.py (ShoppingDeps→TurnContext) · delegation.py (run_subagent) ·
│                  middleware.py (dynamic_instructions/hide_tool/no_parallel_tools) ·
│                  step.py · model.py · events.py · trace.py · guardrails.py
├── agents/        orchestrator.py · product_rec.py · checkout.py ★ · order_status.py ·
│                  writer/ (prompt/payload/blocks) · tools.py · names.py
├── session.py     the turn-graph driver (target: one StateGraph orchestrator→payload→
│                  writer→blocks; shipped: bus + background task — §3.3)
└── snapshot.py    cart + transcript → AgentSnapshot

(to lift)  lg_agent/core/  config.py · builder.py · registry.py · tools.py · guardrails.py ·
                            skills.py · event_store.py
```

## Appendix B — glossary

- **sop** — "standard operating procedure" = the worker name that produced a `StepResult`; the shared vocabulary for routing, block selection, and the `step` event.
- **block** — a deterministic typed card (product list / order status / checkout summary / generic structured), built by code from `StepResult.details` + live state.
- **recall** — domain-rendered text a worker emits for the orchestrator to remember next turn (reference resolution); never shown to the writer.
- **the firewall** — prose (LLM) and cards (code) are separate outputs; the model never writes an id or a total.
