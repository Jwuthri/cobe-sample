# openai_agent_v1 — agent_v4_1, rebuilt on the OpenAI Agents SDK

A clean-room rebuild of [`agent_v4_1`](../agent_v4_1) on the **OpenAI Agents SDK**
(`openai-agents`) instead of LangChain `create_agent`. Same config-driven,
agent-as-tool topology; same streaming-first session; same deterministic blocks —
but every framework primitive is the SDK's. **Imports nothing from `agent_v4_1`**:
the domain, prompts, blocks, extractors, and writer payload are ported in full.

```
openai_agent_v1/
  core/      reusable, tenant-agnostic platform — the thing a config targets
    config.py      the AgentConfig pydantic contract (extra="forbid") — identical
    messages.py    a tiny Msg vocabulary (replaces LangChain message objects)
    registry.py    TOOLS / SKILLS / MIDDLEWARE / GUARDRAILS
    models.py      provider:model → SDK model id + ModelSettings
    tools.py       registry lookup + declarative HTTP-tool compiler (FunctionTool)
    skills.py      Skill + load_skill (a dynamic-instructions PortMiddleware)
    guardrails.py  pii / blocklist / llm_judge + the input pre-flight engine
    middleware.py  PortMiddleware: instruction transforms / tool gates / turn budgets
    subagent.py    SubagentSpec + make_subagent_tool (ONE generic wrapper)
    factory.py     build_agent(config) → agents.Agent
    context.py     TurnContext + the live event bus
    trace.py       deep-trace rendering
  shopping/  the demo tenant — registers into core
    domain/        self-contained mock e-commerce model (cart, catalog, …)
    tools.py       the ~18 @function_tool functions
    agents.py      the 5 agent dicts + the SUBAGENTS specs
    middleware.py  cart_anchor + empty_cart_guard (PortMiddleware)
    blocks.py      deterministic typed blocks (the hallucination firewall)
    session.py     ShoppingSession.run_turn_stream — the streaming pipeline
  cli.py           scripted/interactive eval harness
server/main_openai_v1.py  FastAPI SSE bridge (port 8002, same web UI)
```

`core` never imports `shopping`; registration flows shopping → core.

## How each v4_1 mechanism maps onto the SDK

| agent_v4_1 (LangChain) | openai_agent_v1 (OpenAI Agents SDK) |
|---|---|
| `create_agent(model, tools, system_prompt, middleware)` | `agents.Agent(name, instructions, tools, model, model_settings)` |
| `ToolRuntime[ShoppingContext]` (shared cart) | `RunContextWrapper[ShoppingContext]`, the SAME object threaded into each sub-agent run |
| `@tool` | `@function_tool` |
| orchestrator `astream(stream_mode=["updates","custom"])` | `Runner.run` driven in a background task; sub-agent tools push live UI events onto a per-turn **event bus** (`asyncio.Queue` on the context) the session drains |
| writer `astream(stream_mode="messages")` | `Runner.run_streamed` → `ResponseTextDeltaEvent.delta` tokens (the writer is the LAST model call, nothing after it) |
| sub-agent re-pump (`stream_subagent`) | `Runner.run_streamed` inside the tool, forwarding inner `tool_call`/`tool_call_output` items to the bus |
| `cart_anchor` middleware (`wrap_model_call`) | a **dynamic-instructions** transform (the checkout progress block is appended to the agent's instructions, re-rendered each run) |
| `empty_cart_guard` middleware | a **tool gate** composed into the checkout delegate's `is_enabled` (hidden while the cart is empty) |
| `tool_call_limit` / `max_turns` middleware | a per-run **turn budget** → `Runner.run(max_turns=…)` |
| `log_tool_calls` middleware | no-op — UI events are emitted structurally by the session + sub-agent engine |
| input guardrails (langchain middleware pre-flight) | the same pre-flight engine, reimplemented framework-free (`core/guardrails.py`) |
| LangChain `HumanMessage`/`AIMessage`/`ToolMessage` | one small `Msg` record (`core/messages.py`); `items_to_msgs` converts a `RunResult.new_items` list back into it so the (verbatim) extractors read `role=="tool"` + `name` + `content` |

### The streaming story (preserved)

The writer's model call is the last thing in the turn, so its tokens stream
straight to the client (`{type:"token"}`) with nothing after it. Validation duties
live off the token path: input safety is a **pre-flight gate**; content grounding
is **construction-time** (the writer only sees verified `step_results` + cart;
blocks are built deterministically, so the model never writes an id or a price);
the empty-writer retry is **stream-safe** (an empty stream sent zero tokens).

The orchestrator runs in a background task. Its sub-agent tools (`make_subagent_tool`)
push live events — `router`, `tool_start`/`tool_end` (forwarded from the sub-agent's
own streamed run), the deep-trace frames, `agent`, and the distilled `step` — onto a
per-turn `asyncio.Queue` the session drains in order. Parallel tool calls are
disabled on the orchestrator so a compound message routes one sub-agent per turn,
keeping the bus ordered.

### Context isolation — the orchestrator owns interpretation

Sub-agents do **not** see the conversation. The orchestrator is the sole reader of
the transcript: it resolves the user's references ("the green one", "make it 2")
into a concrete, self-contained `query` and passes only that. To resolve references
without the chat it gets a **routing memo** — assembled from two domain-agnostic
sources: live state (`ctx.routing_context()`, the current cart) and persisted
per-step recalls (`StepResult.recall`). A sub-agent operates on
`(query + shared structured state)` — the cart is its memory, not the chat.

## Run

```bash
# scripted complex scenario (browse → backtrack → checkout → confirm)
uv run python -m openai_agent_v1.cli --demo
uv run python -m openai_agent_v1.cli --demo --trace   # + deep-trace markers

# interactive
uv run python -m openai_agent_v1.cli

# server (same web UI as v2/v4/v4_1)
uvicorn server.main_openai_v1:app --reload --port 8002
curl -N -X POST localhost:8002/api/turn/s1 \
  -H 'content-type: application/json' -d '{"message":"show me hoodies"}'

# tests (no real LLM calls)
uv run pytest tests_openai_v1 -q
```

The model is read from `OPENAI_AGENT_V1_MODEL` → `AGENT_V4_1_MODEL` →
`AGENT_V2_OPENAI_MODEL` → `openai:gpt-5.4-mini`.
