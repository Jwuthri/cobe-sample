# agent_v4_1 — config-driven sub-agents on `create_agent`, streaming-first

A cleaner rebuild of `agent_v4`'s declarative idea on the `agent_v5` agent-as-tool
topology — with **true token streaming** as the headline feature. One
`AgentConfig` dict defines an agent (model, prompt, tools, skills, guardrails,
middleware, structured output); the orchestrator and writer are both
`langchain.agents.create_agent`.

## Two layers

```
agent_v4_1/
  core/      reusable, tenant-agnostic platform — the thing a config targets
    config.py      the AgentConfig pydantic contract (extra="forbid")
    registry.py    TOOLS / SKILLS / MIDDLEWARE / GUARDRAILS
    models.py      resolve_model → init_chat_model("provider:model", …)
    tools.py       registry lookup + declarative HTTP-tool compiler
    skills.py      Skill + load_skill + SkillsMiddleware (transient injection)
    guardrails.py  pii / blocklist / llm_judge + the input pre-flight engine
    middleware.py  model_call_counter / max_turns / tool_call_limit / log_tool_calls
    factory.py     build_agent(config) → create_agent(...)
    subagent.py    SubagentSpec + make_subagent_tool (ONE generic wrapper)
  shopping/  the demo tenant — registers into core
    domain/        self-contained mock e-commerce model (cart, catalog, …)
    tools.py       the ~18 @tool functions (no skill gating)
    agents.py      the 5 agent dicts + the SUBAGENTS specs
    middleware.py  cart_anchor + empty_cart_guard
    blocks.py      deterministic typed blocks (the hallucination firewall)
    session.py     ShoppingSession.run_turn_stream — the streaming pipeline
server/main_v4_1.py  FastAPI SSE bridge (port 8001, same web UI)
```

`core` never imports `shopping`; registration flows shopping → core.

## The streaming story (why v4 couldn't, and how v4.1 does)

In `agent_v4` the user-facing text is born in the `writer` node, inspected by
`validator`, and only materialized in `emit` — by the time it's emittable, every
token already exists, so the server can only forward node-granular updates.
`agent_v5` is worse: the whole turn blocks, then synthetic events are sent.

**The fix is to make the writer's model call the last thing in the turn** and move
every validation duty off the token path:

| validation duty | v4.1 home (stream-safe) |
|---|---|
| input safety (blocklist / llm_judge / PII-in) | **pre-flight gate** before any model call — a refusal is instant; a redaction rewrites the user text before it enters the transcript |
| content grounding ("never claim order placed") | **construction-time** — the writer only sees verified `step_results` + cart; blocks are built deterministically (ids/prices verbatim, the model never writes them) |
| retry if the writer emits nothing | **inherently stream-safe** — an empty stream sent zero tokens, so the retry is invisible |
| output checks on sub-agents | free — tool results are never user-visible |
| output checks on the writer (PII-out / llm_judge) | the one trade-off — forces **buffered mode** (no tokens that turn); opt-in via writer config, default off |

So the writer streams its tokens straight to the client (`{type:"token"}`) with
nothing after it. There is no post-generation validator gating the stream because
the grounding already happened. A turn looks like:

```
user → state
     → router/tool_start/tool_end/step …   (orchestrator phase, live)
     → state
     → router(writer) → token token token … (writer phase, live)
     → writer → bot → state → end
```

(or `user → state → guardrail → bot(refusal) → state → end` if input is blocked.)

The orchestrator is streamed with `stream_mode=["updates","custom"]` (so sub-agent
and judge tokens can never leak); the writer with `stream_mode="messages"`.

## Deep-trace debugging (`{type:"trace"}`)

The events above tell you *that* a sub-agent ran and *what* it summarized. With
`ShoppingSession(debug=True)` (the default) the engine also emits a backward-safe
`{type:"trace", phase, agent, title, data}` frame for the layer below — the exact
payloads moving between the actors:

| phase | emitted by | `data` answers |
|---|---|---|
| `orchestrator_input` | session | what the orchestrator sees: the conversation, its system prompt, the delegate sub-agents, the `routing_memo` (cart + recently shown products), and the initial runtime context |
| `subagent_input` | sub-agent tool | what the orchestrator sends *into* a sub-agent: its prompt, tools, `isolated: true`, and `input_seen` — the self-contained `query` (+ optional cart note), NOT the chat |
| `subagent_output` | sub-agent tool | what the sub-agent hands *back*: its raw messages, the distilled `StepResult`, and the terse string the orchestrator LLM actually reads |
| `context` | sub-agent tool | the `TurnContext` after that sub-agent mutated it (step_results, usage, cart) — watch it evolve call by call |
| `writer_payload` | session | the exact grounded JSON the writer composes its reply from + the writer system prompt + mode |

Session-level frames are `yield`ed straight; in-graph frames (the sub-agent ones,
which run *inside* the orchestrator graph) ride the custom stream via
`core/trace.py::emit_trace` and are lifted back out by `events.classify_custom`.
Everything is pre-trimmed (`_MAX_FIELD_CHARS`) and JSON-safe. The frontend renders
each frame as a collapsible JSON tree under a **TRACE** row; a `trace` toggle in
the Events header hides them. Set `debug=False` to drop the whole layer (zero
frames, no overhead) — the core turn behavior is identical either way.

Message arrays (`input_seen` / `conversation_seen` / `raw_messages`) render with
**role chips** (human / ai / system / tool). Because of context isolation (below) a
sub-agent's `input_seen` is normally just ONE `human` message — the orchestrator's
instruction — tagged `note: "orchestrator's instruction (the tool query)"`. Only
the orchestrator's `conversation_seen` carries the full transcript.

**Turn delimiter:** each turn increments `ShoppingSession.turn`, carried on the
`{type:"user", …, turn}` event; the Events panel draws a "Turn N" divider before
each user row so successive turns don't blur together.

## The config contract

```python
{
  "name": "...", "description": "...", "system_prompt": "...",
  "instructions": ["..."],                          # appended as bullet rules
  "model": {"provider_model": "openai:gpt-5-mini", "temperature": 0.0},
  "skills":  [{"kind": "custom", "name": "...", "description": "...", "skill": "..."}],
  "tools":   [{"kind": "registry", "name": "..."},
              {"kind": "http", "name": "...", "method": "POST", "url": "...",
               "headers": {...}, "parameters": {<json-schema>}}],
  "guardrails": [{"type": "pii|blocklist|llm_judge", "action": "...",
                  "on_input": true, "on_output": false, "message": "...", "params": {...}}],
  "middleware": [{"name": "max_turns", "params": {"max_turns": 30}}],
  "output_format": {<json-schema object>}           # → response_format, or null
}
```

`agent_v4_1/examples.py::EXAMPLE_AGENT_CONFIG` is the reference (it validates
verbatim — see `tests_v4_1/test_config.py`).

### Sub-agents as one generic wrapper

`SubagentSpec` (in `core/subagent.py`) replaces v5's three copy-pasted `@tool`
functions: the skeleton (snapshot → build input → run → extract `StepResult` →
return a terse summary) is written once; per-agent differences are small plug-in
callables (`shopping/extractors.py`). The orchestrator LLM only ever reads the
terse summary; rich data rides `StepResult.details` → deterministic blocks.

### Context isolation — the orchestrator owns interpretation

Sub-agents do **not** see the conversation. The orchestrator is the *sole* reader
of the transcript: it resolves the user's references ("the green one", "add it",
"make it 2") into a concrete, self-contained `query` and passes only that. A
sub-agent operates on `(query + shared structured state)` — the **cart is its
memory, not the chat**. `build_input(ctx, query)` may add deterministic state
notes (product_rec's cart note; checkout's `cart_anchor` progress block) but never
history.

To resolve references without the chat, the session hands the orchestrator a
**routing memo** (`ShoppingSession._routing_memo`) — assembled from two
**domain-agnostic** sources, so the engine never hardcodes "products":

  * **live state** — `ctx.routing_context() -> {label: text}` (a `TurnContext`
    hook; `ShoppingContext` returns the current cart). Recomputed each turn.
  * **persisted recalls** — `StepResult.recall`, a free-text snippet a step
    surfaces for next turn ("Recently shown products: P-4 …"). The session keeps
    the latest per sop in `routing_notes` and never inspects it; the *domain*
    extractor renders the text (`shopping/extractors.py`). A future `order_status`
    or `doc_search` sub-agent populates the same field — no engine change.

So "the green one" binds to an id deterministically, in one place. Why this shape:
one LLM owns interpretation (no split-brain where router *and* sub-agent both parse
intent), fewer tokens, smaller prompt-injection surface, each sub-agent is a clean
testable function — and the carry-context mechanism is generic to any tenant.
Trade-off: the router prompt must produce good queries — the `subagent_input`
trace is how you verify it does.

## Run

```bash
# server (same web UI as v2/v4/v5)
uvicorn server.main_v4_1:app --reload --port 8001
# watch tokens stream in:
curl -N -X POST localhost:8001/api/turn/s1 \
  -H 'content-type: application/json' -d '{"message":"show me hoodies"}'

# web client (types the reply out live)
cd web && npm run dev

# tests (no real LLM calls)
uv run pytest tests_v4_1 -q
```
