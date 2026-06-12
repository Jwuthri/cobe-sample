"""The ``checkout`` sub-agent — drive one purchase from identity to confirmation.

Self-contained like the other sub-agents. The distinctive piece here is
:func:`checkout_progress`: a deterministic "what's done / what's next" block
rendered from the cart's step. It is injected on every checkout model call by the
``cart_anchor`` middleware (see :mod:`lg_agent.shopping.middleware`) so the agent
never has to rediscover state from a growing thread — the cart is the truth.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from lg_agent.core.step import StepResult
from lg_agent.core.subagent import SubAgent
from lg_agent.shopping.agents.subagents.checkout_skills import CHECKOUT_SKILLS, STEP_SKILL
from lg_agent.shopping.agents.subagents.names import CHECKOUT
from lg_agent.shopping.tools import CHECKOUT_TOOLS, registry_specs

MODEL = "openai:gpt-5.4-mini"

# The base prompt is deliberately LEAN — the per-step detail lives in skills that
# are loaded on demand (see checkout_skills.py). Each step's tools are hidden until
# its skill is loaded, so the agent must load the skill to act.
PROMPT = """\
You are the checkout assistant. You move ONE order forward, one step at a time.

Every turn you get a "Checkout progress" block: the authoritative state of the
order (which fields are ✓) plus the CURRENT step and the skill it needs.

How you work:
  - Load the skill for what the user is doing, then follow it. Usually that is the
    skill named for the CURRENT step in the progress block. BUT if the user wants to
    CHANGE a field that is already ✓ — a different delivery option, payment method,
    or address — load THAT field's skill instead (see the Available skills list),
    e.g. "change delivery to 2h" → load_skill('collect_delivery'), "use card now" →
    load_skill('collect_payment'). A field's tools are HIDDEN until its skill is
    loaded, so loading the RIGHT skill is how you unlock the tool you need.
  - CLASSIFY the user's latest message before acting: is it a name, an address, a
    delivery option, a payment method, or a yes/no? Act ONLY on the field it
    actually contains, and NEVER invent a value the user didn't give. Your incoming
    message is a routing instruction, not the customer's data.
  - NEVER re-capture a field already marked ✓ unless the user is explicitly changing
    it — trust the progress block.
  - Internal steps (serviceability, pricing) need no user input: load their skill
    and perform them when you reach them.

Always-available actions (no skill needed — call directly when the user asks):
  - apply_promo(code) — apply a discount/promo code (e.g. "use WELCOME10").
  - get_cart_summary() — re-check the cart if you genuinely need to.

Items are already in the cart — you have no add-item tool. You don't speak to the
user; the writer composes the reply. Load the step's skill, do the work via tools,
and stop.
"""

DESCRIPTION = (
    "Move an order forward: capture identity, shipping address, delivery option, "
    "and payment, then place the order ONLY on the user's explicit 'yes'. Requires "
    "items already in the cart. Pass the user's latest checkout-relevant message as "
    "`query` (their name, an address, a delivery choice, a payment method, or 'yes')."
)

CONFIG = {
    "name": CHECKOUT,
    "description": "Drive an in-progress purchase from identity to payment to confirmation.",
    "system_prompt": PROMPT,
    "model": {"provider_model": MODEL, "temperature": 0.0},
    "tools": registry_specs(CHECKOUT_TOOLS),
    "skills": CHECKOUT_SKILLS,  # one per step, loaded on demand; each unlocks its tools
    "middleware": [
        {"name": "cart_anchor", "params": {}},
        {"name": "log_tool_calls", "params": {"log_prefix": CHECKOUT}},
    ],
}


# =============================================================================
# the deterministic "Checkout progress" block (also used by cart_anchor middleware)
# =============================================================================
def asks_for_step(step_value: str, cart) -> list[str]:
    """What the user still needs to provide at the current step."""
    if step_value == "collecting_identity":
        return ["first name", "last name"]
    if step_value == "collecting_address":
        if cart.serviceable is False:  # the saved address doesn't ship — need a different one
            return [f"a different, serviceable shipping address (we don't ship to zip {cart.address.zip_code})"]
        return ["street", "city", "state", "zip code"]
    if step_value == "awaiting_serviceability":
        return ["(internal: serviceability lookup)"]
    if step_value == "collecting_delivery":
        opts = ", ".join(cart.serviceable_options) or "available delivery options"
        return [f"delivery option ({opts})"]
    if step_value == "collecting_payment":
        return ["payment method (card / cash / wallet)", "card_token if paying by card"]
    return []


def _resume_line(step_value: str) -> str:
    """Point the agent at the skill for the current step (detail lives in the skill)."""
    skill = STEP_SKILL.get(step_value)
    if skill:
        return f"Current step: {step_value} → load_skill('{skill}') for instructions, then act."
    if step_value == "confirmed":
        return "Order already placed — do nothing."
    return "Continue with the next missing field."


def checkout_progress(cart) -> str:
    """The authoritative state block injected each turn.

    Shows what's ✓ and names the skill the current step needs — the *how-to* itself
    is loaded on demand from that skill, keeping this block (and the prompt) lean.
    """
    c = cart

    def mark(done: bool, value: str) -> str:
        return f"✓ {value}".rstrip() if done else "— not provided"

    name = f"{c.customer.first_name or ''} {c.customer.last_name or ''}".strip()
    identity = mark(bool(c.customer.first_name), name)
    address = mark(
        c.address.is_complete(),
        f"{c.address.street}, {c.address.city} {c.address.zip_code}",
    )
    if c.serviceable is True:
        serviceability = f"✓ ships here (options: {', '.join(c.serviceable_options)})"
    elif c.serviceable is False:
        serviceability = "✗ NOT serviceable — ask for a different address"
    else:
        serviceability = "— not checked"
    delivery = mark(bool(c.delivery_option), c.delivery_option or "")
    payment = mark(bool(c.payment_method), c.payment_method or "")
    if c.shipping_is_fresh() and c.tax_is_fresh():
        pricing = f"✓ shipping {c.shipping.cost} + tax {c.tax.amount} → total {c.grand_total}"
    elif c.delivery_option:
        pricing = "✗ STALE — recompute via the collect_delivery skill"
    else:
        pricing = "— not computed"

    return (
        "Checkout progress (authoritative — never redo a ✓ field):\n"
        f"  identity:       {identity}\n"
        f"  address:        {address}\n"
        f"  serviceability: {serviceability}\n"
        f"  delivery:       {delivery}\n"
        f"  payment:        {payment}\n"
        f"  pricing:        {pricing}\n"
        f"{_resume_line(c.step.value)}"
    )


# =============================================================================
# hooks
# =============================================================================
def build_input(ctx, query: str) -> dict:
    """Just the instruction — the progress anchor comes from the cart_anchor
    middleware, and the cart is the source of truth (no history needed)."""
    return {"messages": [HumanMessage(content=query)]}


def extract(ctx, messages, before) -> StepResult:
    cart = ctx.cart_service.cart
    asks: list[str] = []
    if cart.step.value.startswith("collecting_"):
        asks = asks_for_step(cart.step.value, cart)
    elif cart.ready_to_confirm() and not cart.confirmed:
        asks = ["explicit yes to place the order"]
    return StepResult(
        sop=CHECKOUT,
        summary=f"checkout subagent finished at step={cart.step.value}; items={len(cart.items)}",
        asks=asks,
        next_sop=None,
        cart_diff={"step": cart.step.value},
    )


def summarize(sr: StepResult, ctx) -> str:
    asks_note = f" Needs from user: {', '.join(sr.asks)}." if sr.asks else ""
    confirmed = " ORDER CONFIRMED." if ctx.cart_service.cart.confirmed else ""
    return f"{sr.summary}.{asks_note}{confirmed}"


SUBAGENT = SubAgent(
    name=CHECKOUT,
    description=DESCRIPTION,
    config=CONFIG,
    build_input=build_input,
    extract=extract,
    summarize=summarize,
    block="checkout",
)

__all__ = ["SUBAGENT", "CONFIG", "PROMPT", "checkout_progress", "asks_for_step"]
