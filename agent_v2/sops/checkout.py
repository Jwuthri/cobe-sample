"""Checkout subagent — ``create_agent`` with the skills + log stack.

State is per-invocation (carried via ``thread_id`` in config) and
per-cart (carried via ``RuntimeContext`` in context).

Confirmation is **prompt-gated** (the model only calls
``confirm_checkout`` after the user has explicitly approved an
order summary the writer presented). For production-grade
safety on irreversible actions, re-add
``langchain.agents.middleware.HumanInTheLoopMiddleware`` here.
"""

from __future__ import annotations

from agent_v2.llm import chat_model
from agent_v2.middleware import SkillsMiddleware, log_tool_calls
from agent_v2.runtime import RuntimeContext
from agent_v2.skills import CHECKOUT_SKILLS
from agent_v2.tools.checkout_tools import CHECKOUT_TOOLS
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

CHECKOUT_SYSTEM_PROMPT = """\
You are the checkout assistant. You guide the user through placing an order.

The checkout flow has FIVE sub-skills, loaded in order:
  1. collect_identity         (capture first/last name)
  2. collect_address          (capture shipping address)
  3. lookup_serviceability    (verify we ship there + which options)
  4. collect_delivery         (pick a delivery option, quote shipping/tax)
  5. collect_payment          (capture payment method)

Always call ``load_skill(name)`` BEFORE you call any tool that skill
unlocks. The available-skills block in the system prompt shows what's
already loaded.

You can call ``get_cart_summary()`` any time to see the current cart
state and any blockers.

Items may already be in the cart from a previous step (e.g., handed
off from product recommendations). Don't try to add them again —
inspect the cart and proceed to the next missing piece.

## Confirmation rule (read carefully)

NEVER call ``confirm_checkout`` automatically when the cart becomes
ready_to_confirm. Instead:
  1. Call ``get_cart_summary()`` so the user can see the order.
  2. Stop — let the writer present the summary and ask the user to
     confirm.
  3. ONLY call ``confirm_checkout`` on a SUBSEQUENT turn, when the
     user's most recent message is an explicit approval like "yes",
     "y", "confirm", "place the order", "go ahead", "do it".
  4. If the user pushes back ("wait", "no", "actually change…"),
     DO NOT call confirm_checkout — handle their request instead.

You no longer speak directly to the user — the writer agent produces
the final reply. Just do your work via tool calls and stop when
you've made progress; the writer will summarize.
"""


def build_checkout_agent(checkpointer: InMemorySaver | None = None, store=None):
    """Compile the checkout subagent. Reuse the same instance across turns.

    The checkpointer is still here so per-session state survives across
    turns of the same conversation; it's just not required for HITL
    anymore (HITL was removed in v5 — see ARCHITECTURE.md §9).
    """
    return create_agent(
        model=chat_model(),
        tools=CHECKOUT_TOOLS,
        system_prompt=CHECKOUT_SYSTEM_PROMPT,
        middleware=[
            SkillsMiddleware(CHECKOUT_SKILLS),
            log_tool_calls,
        ],
        context_schema=RuntimeContext,
        checkpointer=checkpointer or InMemorySaver(),
        store=store,
    )
