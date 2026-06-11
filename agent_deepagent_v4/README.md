# agent_deepagent_v4 — Shopping Assistant on LangChain **deepagents**

> **In one breath:** the same customer-facing shopping assistant as `agent_v4`,
> rebuilt on the [**deepagents**](https://docs.langchain.com/oss/python/deepagents)
> harness. One **orchestrator** (the main deep agent) routes each message to
> **worker subagents** through the built-in `task` tool, then hands the final
> wording to a single **writer** subagent. Checkout is **safe**: a blocker gate
> + a human-approval `interrupt()` + writer-honesty mean an order is only ever
> placed after an explicit, validated approval.

This package is **self-contained** — it imports nothing from `agent_v4` or any
other package in the repo. The domain (cart, catalog, pricing, serviceability,
orders, memory) is rewritten under `domain/`.

---

## 1. Topology

```
                         ┌─────────────────────────────────────────────┐
   user message ───────▶ │  ORCHESTRATOR  (main deep agent)            │
                         │  routes via the built-in `task` tool;       │
                         │  never speaks to the customer directly      │
                         └───┬───────────┬───────────┬───────────┬─────┘
                  task(...)  │           │           │           │  task(...)
                             ▼           ▼           ▼           ▼
                    ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐
                    │  product   │ │ checkout │ │   order    │ │  writer  │
                    │   agent    │ │  agent   │ │  status    │ │  agent   │
                    │ browse +   │ │ safe     │ │  agent     │ │ the one  │
                    │ cart edits │ │ fulfill  │ │ past order │ │ voice    │
                    └─────┬──────┘ └────┬─────┘ └─────┬──────┘ └────┬─────┘
                          │ scoped tools + shared cart (via context)│
                          ▼            ▼             ▼               ▼
                    ┌────────────────────────────────────────────────────┐
                    │ domain/  Cart · CartService · catalog · pricing ·   │
                    │          serviceability · orders · long-term memory │
                    └────────────────────────────────────────────────────┘
```

Each turn: **orchestrator → (worker task)\* → writer task → return the writer's
message verbatim.** A pure greeting goes straight to the writer; a compound
message ("show hoodies AND where's my order ORD-7") fans out one `task` call per
intent before the writer composes a single reply.

| Agent | Role | Files |
| --- | --- | --- |
| **orchestrator** | The only router. Delegates via `task`, never writes prose itself, always finishes through the writer. | `agents/orchestrator/{prompt,agent}.py` |
| **writer-agent** | The single customer-facing voice. Reads the live cart for totals/confirmation so numbers are verbatim. | `agents/writer/{prompt,tools,agent}.py` |
| **product-agent** | Browse + **cart contents**: search, lookups, serviceability, add/remove/quantity. | `agents/product_rec/{prompt,tools,agent}.py` (+ `shopping` skill) |
| **checkout-agent** | **Safe fulfillment**: identity → address → serviceability → delivery → payment → place order. Cannot add/remove products. | `agents/checkout/{prompt,tools,skills,agent}.py` (+ `checkout` skills) |
| **order-status-agent** | Status / tracking of a past order. | `agents/order_status/{prompt,tools,agent}.py` |

Every worker is a **declarative `SubAgent` dict** — exactly the JSON-object shape
the deepagents docs use:

```python
{
    "name": "product-agent",
    "description": "Browse + cart contents …",   # how the orchestrator decides to delegate
    "system_prompt": PRODUCT_REC_PROMPT,
    "tools": PRODUCT_REC_TOOLS,                   # scoped — only this agent's tools
    "model": "openai:gpt-5.4-mini",               # optional per-agent override
    "skills": ["shopping"],                        # progressive-disclosure knowledge
}
```

---

## 2. The one trick that makes it work: a shared cart over runtime **context**

deepagents subagents run in **isolated contexts** — the orchestrator only sees a
worker's final result, not its tool calls. So the cart can't ride on messages.
Instead it rides on **runtime context**, which deepagents *propagates unchanged
to every subagent*:

```python
@dataclass
class ShopContext:
    user_id: str
    session_id: str
    cart_service: CartService     # one instance, shared by reference
    require_approval: bool = True # feature flag for the checkout interrupt
```

Every tool reads it through `runtime: ToolRuntime[ShopContext]` →
`runtime.context.cart_service`. Because all subagents share the *same*
`CartService` reference, an item the **product-agent** adds is visible to the
**checkout-agent** later in the same turn, and to the **writer** when it composes
the reply. The cart persists across turns because the harness keeps one
`CartService` per session (`runtime.SESSIONS`) and re-passes it as context.

> This mirrors `agent_v4`'s `RuntimeContext`/`CartService` exactly — the cart is
> the single source of truth; tools are dumb wrappers over `CartService`, which
> owns all mutation + quote-invalidation policy.

---

## 3. Safe checkout — three independent layers

1. **The blocker gate (domain invariant).** `Cart.blockers()` lists everything
   missing/inconsistent (empty cart, missing field, unserviceable address, stale
   shipping/tax quote, missing card token…). `confirm_checkout` **refuses** while
   any blocker remains and `CartService.confirm()` raises rather than place an
   unsafe order. This is enforced in code, independent of the model.

2. **Human-in-the-loop approval (`interrupt()`).** Once the cart is clean,
   `confirm_checkout` calls LangGraph's `interrupt()` with a rich payload (items,
   grand total, payment method, ship-to). The run **pauses**; the order is placed
   only when resumed with `Command(resume={"approved": True})`. A `{"approved":
   False}` resume leaves the cart unconfirmed. (Requires a checkpointer +
   `thread_id`, which the runtime always provides. Toggle with the
   `require_approval` context flag.)

3. **Writer honesty (prompt + authoritative read).** The writer reads
   `read_cart()` and is instructed to **never** claim the order is placed unless
   `confirmed == true` — even if the customer just said "confirm". The receipt id
   only exists after layer 2 succeeds.

The agent literally cannot place a premature or unapproved order — verified by
`tests_deepagent_v4/` (`test_premature_confirm_is_blocked`,
`test_safe_checkout_reject_does_not_place`).

```
ready_to_confirm ──"yes"──▶ confirm_checkout()
                              │ blockers? ──yes──▶ refuse, writer asks for missing info
                              │ no
                              ▼
                          interrupt(summary)  ──▶ run pauses (needs_approval)
                              │ resume {approved:false} ─▶ NOT placed
                              │ resume {approved:true}
                              ▼
                          CartService.confirm() ─▶ confirmed=true, RCPT-#### minted,
                                                    address/payment/order → long-term memory
```

---

## 4. The validator

`validator.ResponseValidatorMiddleware` is attached to the orchestrator and runs
`after_agent`: it guarantees the turn ends with a **non-empty** customer message
(graceful fallback otherwise). It deliberately does **no** content regexing —
confirmation-safety is owned by the three layers above — mirroring how `agent_v4`
settled its validator down to a pure structural net after its regex "gate"
produced false positives.

---

## 5. Skills (deepagents-native, progressive disclosure)

Checkout and product knowledge live as real `SKILL.md` files under `skills/`,
loaded through the framework's `SkillsMiddleware`:

```
skills/
  checkout/                # → checkout-agent  "skills": ["checkout"]
    checkout-flow/SKILL.md # step order, internal vs user steps, confirmation safety
    payments/SKILL.md      # payment methods + promo codes
  shopping/                # → product-agent   "skills": ["shopping"]
    recommendations/SKILL.md  # resolving "the cheaper one", browse→buy handoff
```

At startup each skill's `name` + `description` is injected into that agent's
prompt; the full body is read on demand. The deepagents `FilesystemBackend`
(rooted at `skills/`) is what lets the subagents read them. Tool-gating and step
safety are **not** delegated to skills — those stay in the `Cart` domain
invariants; skills carry the *playbook*, the cart enforces the *rules*.

---

## 6. Mapping from `agent_v4`

| `agent_v4` (hand-wired LangGraph) | `agent_deepagent_v4` (deepagents harness) |
| --- | --- |
| `supervisor` node (LLM classifier + loop) | **orchestrator** prompt + the `task` tool loop |
| `*_wrapper` leaf nodes | **subagents** (declarative `SubAgent` dicts) |
| `writer` node | **writer-agent** subagent |
| `validator` node | `ResponseValidatorMiddleware` (`after_agent`) |
| `RuntimeContext` / `CartService` via `context=` | `ShopContext` / `CartService` via `context=` (propagated to subagents) |
| custom `load_skill` tool + `skills_loaded` gating | deepagents `SKILL.md` + `SkillsMiddleware` |
| checkout confirmation safety (prompt + blockers) | blockers gate **+ `interrupt()` HITL** + writer honesty |
| `StepResult` records routed by the supervisor | the `task` tool's returned summaries |

---

## 7. Run it

Uses the repo's existing `.env` (`OPENAI_API_KEY`, `AGENT_V2_OPENAI_MODEL`), so
no new secrets. Model resolution: `AGENT_DEEPAGENT_V4_MODEL` →
`AGENT_V4_OPENAI_MODEL` → `AGENT_V2_OPENAI_MODEL` → `gpt-5.4-mini`.

```bash
# Interactive CLI (drives the approval prompt for you):
uv run python -m agent_deepagent_v4.cli

# Web console (the existing web/ frontend, no FE changes needed):
uv run uvicorn server.main_deepagent_v4:app --reload --port 8001   # FE rewrites /api → :8001
cd web && npm run dev                                              # http://localhost:3000
#   …or keep the server on 8002 and start the web app with
#   AGENT_V2_API_URL=http://localhost:8002 npm run dev

# Tests — offline domain + structure (no key needed):
uv run pytest tests_deepagent_v4/test_domain.py tests_deepagent_v4/test_build.py -q
# Live end-to-end journeys (needs OPENAI_API_KEY):
uv run pytest tests_deepagent_v4/test_agent_journey.py -q
```

Programmatic:

```python
from agent_deepagent_v4 import run_turn, resume_turn

r = run_turn("sess-1", "add the black hoodie and check me out")
# … provide name, address, delivery, payment …
r = run_turn("sess-1", "yes, place the order")
if r.needs_approval:                       # safe-checkout pause
    r = resume_turn("sess-1", {"approved": True})
print(r.reply, r.cart["receipt_id"])
```

---

## 7a. Frontend integration

`server/main_deepagent_v4.py` speaks the **same SSE + snapshot contract** the
existing `web/` Next.js console already uses for `agent_v4`
(`web/lib/api.ts`, `web/lib/types.ts`), so the browser UI works with **no
frontend changes**:

- `POST /api/session`, `GET /api/state/{id}`, `POST /api/turn/{id}` (SSE).
- The deep agent isn't a fixed graph, so each turn is run and then its SSE
  events are **reconstructed from the orchestrator's trace** — every `task` tool
  call becomes a `router` + `tool_start`/`tool_end` + `agent` (+ `step`) event,
  so the event panel shows the delegation trace; the final reply becomes
  `writer` + `bot`; the cart panel renders from the `state` snapshot.
- **Safe checkout over a UI with no approval widget:** the HITL pause is
  surfaced **conversationally** — when the order is ready, the bot replies with
  the summary and asks the customer to reply "yes"; that next "yes"/"no" resumes
  the `interrupt()`. The order still only places on an explicit approval (the
  gate is preserved, *not* auto-approved).

## 8. File map

```
agent_deepagent_v4/
  config.py            model resolution + .env loader
  context.py           ShopContext (shared cart + require_approval flag)
  messages.py          text_of() — flatten content blocks to text
  validator.py         ResponseValidatorMiddleware (non-empty reply net)
  runtime.py           sessions, run_turn / resume_turn, cart snapshot
  cli.py               interactive REPL with the approval prompt
  domain/              Cart · CartService · catalog · pricing · serviceability · orders · memory
  agents/
    orchestrator/      prompt + build_orchestrator (assembles the whole agent)
    product_rec/       prompt + tools + SubAgent spec
    checkout/          prompt + tools + skills + SubAgent spec
    order_status/      prompt + tools + SubAgent spec
    writer/            prompt + tools + SubAgent spec
  skills/              SKILL.md files (checkout-flow, payments, recommendations)
server/main_deepagent_v4.py   FastAPI bridge
tests_deepagent_v4/           offline domain/build tests + live journey tests
```

---

## 9. Design notes & honest limitations

- **Writer relay.** The orchestrator finishes every turn by delegating to the
  writer and returning its message verbatim. That's the cost of "1 orchestrator
  + 1 separate writer": one extra model hop, and a soft reliance on the
  orchestrator copying the writer's text through. The validator backstops
  emptiness; the prompts keep the orchestrator out of the customer's voice.
- **Single chokepoint for HITL.** We use the tool-level `interrupt()` (rich
  payload) and deliberately do **not** also set `interrupt_on={"confirm_checkout":
  True}` — the framework's HITL middleware fires *before* the tool and would
  pre-empt our payload. `interrupt_on` is the declarative alternative if you
  don't need a custom approval payload.
- **Filesystem tools.** Because skills require a `FilesystemBackend`, every
  subagent also carries the harness's read/ls/grep file tools (over the trusted,
  read-only `skills/` dir). They're unused by the shopping flow; the prompts keep
  agents on their domain tools.
- **Prose-only replies in the UI.** The writer returns prose, so the frontend's
  typed *cards* (product/order/checkout blocks) don't render — the chat shows the
  text and the cart panel visualizes the cart. Adding a `response_format` to the
  writer to also emit typed blocks is the natural next step.
- **In-memory state.** Carts (`SESSIONS`), the checkpointer, and the store are
  in-process. Swap `InMemorySaver`/`InMemoryStore` for durable backends in prod.
```
