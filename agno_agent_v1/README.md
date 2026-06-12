# agno_agent_v1 — the `agent_v4_1` shopping assistant, rebuilt on Agno

A faithful, from-scratch re-implementation of [`agent_v4_1`](../agent_v4_1) on top
of the **Agno** SDK (`agno.Agent`). Same behavior, same streaming-first turn shape,
same hallucination firewall — but every line is written against Agno, with **no
imports from `agent_v4_1`** (or any other package).

The topology is unchanged from v4_1: a **router orchestrator** delegates to
**context-isolated sub-agents**, their grounded results become **deterministic
blocks**, and a dedicated **writer streams the reply token-by-token**.

## Two layers, clean separation

```
agno_agent_v1/
  domain/            pure e-commerce business logic — NO framework
    cart.py            cart model: step machine, freshness fingerprints, blockers gate
    cart_service.py    all mutations + the invalidation policy (repricing-on-edit)
    catalog.py  pricing.py  serviceability.py  orders.py  memory.py
  agent/             the Agno agent layer — one home per abstraction
    prompts.py         the *voice*  — one system prompt per agent
    tools.py           the *actions* — constrained Agno function-tools over the domain
    skills.py          the *skill*  — state-derived instruction (the checkout anchor)
    builder.py         AgentSpec  → agno.Agent  (declarative definition → compiled agent)
    agents.py          the five agents assembled (orchestrator / 3 subs / writer)
    subagent.py        ONE generic "agent-as-tool" wrapper (not N copies)
    extractors.py      sub-agent tool execs → grounded StepResult
    blocks.py          deterministic typed blocks (the hallucination firewall)
    writer_payload.py  the grounded JSON the writer composes from
    context.py         ShoppingContext (rides Agno `dependencies`) + StepResult
    models.py  events.py  guardrails.py  session.py
server/main_agno_agent_v1.py   FastAPI SSE bridge (port 8001, same web UI)
```

`domain/` is the **specification of correct behavior** (cart math, the checkout
step machine, the confirmation gate). The agent layer drives it but never
re-implements its rules. `domain/` imports nothing from `agent/`.

## Each agent abstraction, well-defined

An agent is **data** (`AgentSpec`) plus three orthogonal pieces:

| abstraction | where | what it is |
|---|---|---|
| **prompt** | `prompts.py` | the agent's fixed voice/role/rules (a system prompt) |
| **tool**   | `tools.py`   | an action the agent can take (a typed function over the domain) |
| **skill**  | `skills.py`  | a *state-derived* instruction injected each run (the checkout progress anchor) |

`build_agent(spec)` compiles these into an `agno.Agent`. The **skill** is the
interesting wiring: a declared skill becomes Agno **callable `instructions`** —
re-evaluated against the live cart on every run, so the checkout agent always sees
a fresh "Checkout progress" block (v4_1's `cart_anchor` middleware, ported).

## How it maps onto Agno (the load-bearing mechanics)

* **Shared cart via `dependencies`.** One `ShoppingContext` rides
  `dependencies={"ctx": ctx}`. Agno passes the dict *by reference* from the
  orchestrator into every sub-agent run; tools read it back off
  `run_context.dependencies["ctx"]`. A single live `CartService` mutates in place,
  end-to-end — the analogue of v4_1's `runtime.context`.
* **Agent-as-tool.** Each sub-agent is an `agno.Agent` wrapped as an orchestrator
  tool via `Function.from_callable` (custom name + description, `run_context`
  auto-excluded from the schema). The wrapper runs the sub-agent, distills a
  `StepResult` from its `RunOutput.tools`, and returns a terse summary — the only
  thing the orchestrator LLM ever reads.
* **Context isolation.** Sub-agents do **not** see the conversation. The
  orchestrator is the sole transcript reader: it resolves references ("the green
  one" → P-4) using a deterministic **routing memo** (live cart + persisted
  per-step `recall` snippets) and passes a self-contained `query`.
* **Token streaming.** The writer is the *last* model call. It streams via
  `agent.arun(payload, stream=True)`, and each `RunContentEvent.content` (a delta)
  becomes a `{type:"token"}` SSE frame. Nothing runs after it, so tokens reach the
  client live.
* **Deterministic blocks.** Product cards / order cards / the checkout summary are
  built in Python from `StepResult.details` + the cart, ids/prices verbatim. The
  model never writes them, so it cannot hallucinate a total — which is *why* the
  writer can stream freely.
* **Empty-cart guard.** The orchestrator is built per turn; while the cart is
  empty the `checkout` delegate is withheld, so a first "add X" cannot misroute.
* **gpt-5.x temperature.** `resolve_model` omits a custom temperature for
  `gpt-5*` / `o*` (they reject it); deterministic vs. writer temps apply elsewhere.

## A turn

```
user → state
     → router/tool_start/tool_end/step …   (orchestrator phase: routing + grounded results)
     → state
     → router(writer) → token token token … (writer phase, live)
     → writer → bot → state → end
```

(or `user → state → guardrail → bot(refusal) → state → end` if input is blocked).
With `ShoppingSession(debug=True)` (the default) each phase also emits a
backward-safe `{type:"trace"}` frame exposing the exact payloads between actors.

## Run

```bash
# server (same web UI as v2/v4/v4_1/v5)
uvicorn server.main_agno_agent_v1:app --reload --port 8001
# watch tokens stream:
curl -N -X POST localhost:8001/api/turn/s1 \
  -H 'content-type: application/json' -d '{"message":"show me hoodies"}'

# tests (deterministic layer — no real LLM, no API key)
uv run pytest tests_agno_agent_v1 -q
```

Set `OPENAI_API_KEY` and (optionally) `AGENT_V2_OPENAI_MODEL` / `AGNO_AGENT_V1_MODEL`
in `.env` (defaults to `gpt-5.4-mini`).

## Verified workflows

Exercised end-to-end against the live model (`tests` cover the deterministic
layer; these are the integration scenarios):

* browse → `add P-1` → checkout field-by-field (identity → address →
  serviceability → delivery → payment) → **blocked** premature confirms → confirm;
* **backtrack**: edit a quantity on a ready cart → shipping + tax go stale, total
  becomes `None` (`awaiting_pricing`) → re-quote → ready;
* change the delivery option → shipping re-quoted, tax recomputed, total updated;
* reference resolution ("add the green one" → P-4), compound routing ("add a
  hoodie **and** check order ORD-7" → two sub-agents), non-serviceable delivery
  (2h refused for Paris), promo rules (SHOES20 refused with no shoes), smalltalk.
