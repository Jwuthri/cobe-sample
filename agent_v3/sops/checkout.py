"""Checkout subagent — an Agno ``Agent`` with native Skills + the gating hook.

v2 used a custom ``SkillsMiddleware`` (a ``load_skill`` tool + an
"available skills" prompt block) plus per-tool ``_require_skill`` checks.
v3 replaces that with:

  - ``skills=Skills(loaders=[LocalSkills(<checkout skills dir>)])`` — the
    native Agno progressive-disclosure mechanism. Agno injects the skill
    summaries into the system prompt and exposes ``get_skill_instructions``
    so the agent can load a skill's full body on demand.
  - ``tool_hooks=[skill_gate_hook]`` — re-implements the *gating*: a tool
    stays locked until its skill has been loaded (recorded in
    ``session_state["skills_loaded"]`` when ``get_skill_instructions`` runs).

State is threaded per-turn via ``session_state`` (skills_loaded) +
``dependencies`` (the live cart) — no checkpointer needed.
"""

from __future__ import annotations

from pathlib import Path

from agno.agent import Agent
from agno.skills import LocalSkills, Skills

from agent_v3.gating import skill_gate_hook
from agent_v3.models import chat_model
from agent_v3.tools.checkout_tools import CHECKOUT_TOOLS

CHECKOUT_SKILLS_PATH = str(Path(__file__).resolve().parent.parent / "skills" / "checkout")

CHECKOUT_SYSTEM_PROMPT = """\
You are the checkout assistant. You guide the user through placing an order.

The checkout flow has FIVE skills, loaded in order. Call
get_skill_instructions(name) to load a skill BEFORE using any tool it
covers — a tool stays locked until its skill is loaded:
  1. collect-identity        (capture first/last name -> set_customer)
  2. collect-address         (capture shipping address -> set_address)
  3. lookup-serviceability   (verify we ship there -> lookup_serviceability)
  4. collect-delivery        (pick option -> set_delivery_option, quote_shipping, compute_tax)
  5. collect-payment         (capture payment -> attach_payment, confirm_checkout)

The available skills (with one-line summaries) are listed in your system
context. Load the one for the current step, follow its instructions, then
move to the next.

You can call get_cart_summary() any time to see the current cart state and
any blockers.

Items may already be in the cart (e.g. handed off from product
recommendations). Don't add them again — inspect the cart and proceed to
the next missing piece.

## Confirmation rule (read carefully)

NEVER call confirm_checkout automatically when the cart becomes
ready_to_confirm. Instead:
  1. Call get_cart_summary() so the user can see the order.
  2. Stop — let the writer present the summary and ask the user to confirm.
  3. ONLY call confirm_checkout on a SUBSEQUENT turn, when the user's most
     recent message is an explicit approval like "yes", "y", "confirm",
     "place the order", "go ahead", "do it".
  4. If the user pushes back ("wait", "no", "actually change…"), DO NOT
     call confirm_checkout — handle their request instead.

You no longer speak directly to the user — the writer agent produces the
final reply. Just do your work via tool calls and stop when you've made
progress; the writer will summarize.
"""


def build_checkout_agent() -> Agent:
    return Agent(
        name="checkout",
        model=chat_model(),
        tools=CHECKOUT_TOOLS,
        skills=Skills(loaders=[LocalSkills(CHECKOUT_SKILLS_PATH)]),
        tool_hooks=[skill_gate_hook],
        instructions=CHECKOUT_SYSTEM_PROMPT,
        telemetry=False,
    )
