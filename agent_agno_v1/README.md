# agent_agno_v1 — the v4.1 shopping assistant, rebuilt on Agno

A faithful port of [`agent_v4_1`](../agent_v4_1/README.md) onto the **Agno**
framework (2.6.x). Same behaviour, same SSE/JSON contract, same web UI — different
engine. Where v4.1 wired a LangGraph `create_agent` orchestrator + a dedicated
streaming writer, this package uses an **Agno coordinate-mode `Team`**: the team
leader (the "speaking supervisor") routes to three member sub-agents over a single
live cart, then **authors the user-facing reply itself — streamed token-by-token**.

## Two layers

```
agent_agno_v1/
  core/      reusable, tenant-agnostic Agno platform — the thing a config targets
    config.py      the AgentConfig pydantic contract (extra="forbid")
    models.py      resolve_model → OpenAIChat(id, temperature)
    registry.py    TOOLS / GUARDRAILS name→item registries
    tools.py       registry lookup + declarative HTTP-tool compiler (Agno Function)
    guardrails.py  self-contained input pre-flight (blocklist / pii / llm_judge)
    events.py      Agno stream-event  →  frontend SSE-event bridge
    context.py     TurnContext — the per-turn observation (tool events → StepResults)
    step_result.py the StepResult model
    factory.py     build_agent(cfg)→Agent ; build_team(cfg, members)→Team(coordinate)
  shopping/  the demo tenant — registers into core
    domain/        self-contained mock e-commerce (cart, catalog, pricing, …) — no framework imports
    tools.py       the ~18 @tool functions (read the cart via run_context.dependencies)
    prompts.py     supervisor (router + writer voice, merged) + 3 member prompts
    agents.py      the member config dicts + the supervisor TEAM config
    extractors.py  checkout anchor + tool-event → StepResult distillation
    blocks.py      deterministic typed blocks (the hallucination firewall)
    platform.py    assemble per-session: register tools, build members + team
    session.py     ShoppingSession.run_turn_stream — the streaming pipeline
server/main_agno_v1.py  FastAPI SSE bridge (port 8001, the SAME web UI as v2/v4/v5)
```

`core` never imports `shopping`; registration flows shopping → core.

## Topology: an Agno coordinate-mode Team

```
user → [input guardrails pre-flight]
     → Team(mode=coordinate) leader
         ├─ delegate_task_to_member → product_rec  ┐
         ├─ delegate_task_to_member → checkout      ├─ members, one live cart (shared by deps)
         └─ delegate_task_to_member → order_status ┘
     → leader composes the single reply  → streamed tokens → bot
```

| v4.1 concept | Agno primitive used here |
|---|---|
| router orchestrator + dedicated writer | one `Team(mode=coordinate)` — the leader routes **and** writes |
| sub-agents as tools | `Team` members (`product_rec` / `checkout` / `order_status`) |
| `ShoppingContext.cart_service` (by ref) | `dependencies={"cart": CartService}` — forwarded to members by reference |
| tool reads `runtime.context.cart_service` | tool declares `run_context: RunContext`, reads `run_context.dependencies["cart"]` |
| cart-anchor middleware | a fresh cart snapshot + progress anchor in `session_state` each turn (`add_session_state_to_context`) |
| empty-cart guard middleware | supervisor prompt ("never delegate to checkout while empty") + checkout has no `add_item` |
| tool-call-limit middleware | native `Team(tool_call_limit=…)` |
| true token streaming | `team.arun(stream=True, stream_events=True)` → leader `TeamRunContent` deltas |
| deterministic blocks (firewall) | unchanged — built from `StepResult`s + cart, ids/prices verbatim |
| checkout confirm gate | unchanged — the cart `blockers()` invariant inside `confirm_checkout` (NOT Agno HITL) |
| input guardrails | a self-contained pre-flight engine (no LangChain) |

### Why not Agno's `requires_confirmation` HITL for checkout?

Because the real gate is the deterministic cart invariant, not the model deciding
to call a tool. `confirm_checkout` refuses while `cart.blockers()` is non-empty, so
`cart.confirmed` (never model prose) is the source of truth — model-proof and
unit-testable without an LLM. Agno HITL would also force a pause→resume round-trip
inside one HTTP turn for zero gating benefit.

## The streaming story (how the leader's tokens reach the client)

The team stream interleaves leader and member events; the session discriminates
purely on the Agno `.event` string:

| `.event` | meaning | becomes |
|---|---|---|
| `TeamToolCallStarted` (delegate tool) | the leader routes to a member | `{type:"router", target}` |
| `ToolCallStarted` / `ToolCallCompleted` | a **member** domain tool ran | `{type:"tool_start"}` / `{type:"tool_end"}` + recorded |
| `TeamToolCallCompleted` (delegate) | a delegation finished | distil a `StepResult` → `{type:"step"}` |
| **`TeamRunContent`** | the **leader's** reply delta | **`{type:"token"}`** (user-facing) |
| `RunContent` | a member's internal chatter | dropped (never user-facing) |
| `TeamRunCompleted` | end of leader reply | authoritative final text |

The leader's reply is the last thing in the turn (nothing validates after it), so
its tokens stream straight to the client; structured cards are built
deterministically, so the streamed prose can't invent an id or a total.

> **2.6.x gotchas baked in:** `arun(stream=True)` returns an async generator
> (no `await`); tools inject via `run_context: RunContext` (a bare `session_state`
> param is **not** injected); Agno presents member ids url-safe (`product-rec`),
> canonicalised back to the sop vocabulary in the session; `AGNO_TELEMETRY=false`
> is set on import to keep CI offline.

## Run

```bash
# server (same web UI as v2/v4/v5; needs OPENAI_API_KEY in .env)
uvicorn server.main_agno_v1:app --reload --port 8001
# watch tokens stream in:
curl -N -X POST localhost:8001/api/turn/s1 \
  -H 'content-type: application/json' -d '{"message":"show me hoodies"}'

# web client (types the reply out live — unchanged from v4.1)
cd web && npm run dev

# hermetic tests (no real LLM — a scripted fake Agno stream drives the pipeline)
uv run pytest tests_agno_v1 -q

# live end-to-end smoke (spends OpenAI tokens — full browse→checkout flow)
python -m tests_agno_v1._smoke_live
```

Model defaults via the env chain `AGENT_AGNO_V1_MODEL → AGENT_V2_OPENAI_MODEL →
gpt-4.1-mini` (OpenAI only). The config schema's `provider_model` accepts
`"openai:gpt-4.1-mini"` or a bare id.
```
