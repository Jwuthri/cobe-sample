"""The ``checkout`` worker — drive ONE purchase from identity to confirmation.

This is the heart of the assistant, and the design goal is that the model never has
to *remember* anything: the cart is the truth. Every run, a dynamic instruction
(:func:`checkout_progress`) injects a deterministic "what's done / what's next" block
rendered from :pyattr:`Cart.step`. In the LangChain build this was a bespoke
``cart_anchor`` middleware; in ADK it is an instruction provider — re-evaluated on
every run, so a mid-turn mutation updates it for free.

Two more safety nets keep "confirmed" honest:
  * there is no ``add_item`` tool here (adding is product_rec's job), so a double-add
    is structurally impossible;
  * ``confirm_checkout`` is gated by the cart's invariant ``blockers()`` — the model
    cannot place an order the domain considers incomplete, no matter what it says.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

from google_adk_agent_v1.agents import tools
from google_adk_agent_v1.agents.names import CHECKOUT
from google_adk_agent_v1.runtime import ShoppingDeps, StepResult, Worker, gen_config, make_model
from google_adk_agent_v1.runtime import registry

PROMPT = """\
You are the checkout assistant. You move ONE order forward.

Every turn you are given a "Checkout progress" block — the authoritative state of
the order (the cart persists every captured field across turns). Advance the order
as far as you can THIS turn:

  - CLASSIFY the user's message FIRST: is it a name, an address (has a street
    number / zip), a delivery option, a payment method, or a yes/no? Set ONLY the
    field it actually contains — even if that field is not the next one in order.
    If the message is an address but the name is still missing, call set_address
    and STOP; leave the name empty for the writer to ask. An address is NOT a name.
    Field-label words ("shipping", "address", "delivery", "payment") are NOT a
    name — never pass them to set_customer.
  - Start from the first field that is not yet ✓ and go in order.
  - INTERNAL steps need no user input — always perform them when you reach them:
      * lookup_serviceability() right after an address is set,
      * quote_shipping() AND compute_tax() right after a delivery option is set.
  - STEPS THAT NEED THE USER — the customer's NAME, the shipping ADDRESS, the
    delivery option, the payment method, and the final confirmation — use the
    user's LATEST message ONLY if it actually provides that value. If it does NOT,
    STOP there and do nothing further (the writer will ask them). NEVER invent or
    guess a value: do NOT call set_customer with a name the user didn't state, do
    NOT call set_address with an address they didn't give, do NOT pick a delivery
    option or payment method for them, and NEVER invent a card token — pass
    card_token only if the user actually gave one.
  - If the saved address is NOT serviceable and the user gives a corrected value
    (often just a new ZIP), call set_address AGAIN — reuse the street/city already
    shown in the progress block and swap in the new value.
  - Your incoming message is a ROUTING INSTRUCTION, not the customer's data. Never
    treat the words in it as a name or address — e.g. a message like "user wants to
    checkout with the current cart" contains NO name, so you must NOT call
    set_customer("user", "wants to checkout..."). With no name yet, stop at identity
    and let the writer ask for it.
  - NEVER re-capture a field already marked ✓ unless the user is explicitly changing
    it. Re-doing completed steps is the #1 mistake here — trust the progress block.

Step → tool cheat-sheet:
  - identity:        set_customer(first_name, last_name, email?)
  - address:         set_address(street, city, zip_code, state?, country?)
  - serviceability:  lookup_serviceability()
  - delivery:        set_delivery_option(option) THEN quote_shipping() THEN compute_tax()
  - payment:         attach_payment(method, card_token?)   (card needs a token)
  - promo:           apply_promo(code)  (any time the user gives a discount code)
Call get_cart_summary() only if you genuinely need to double-check something.

Items are already in the cart from product selection — you have no add-item tool;
never try to add products.

## Confirmation rule (read carefully)

NEVER call confirm_checkout just because the cart is ready. Only call it when the
user's LATEST message is an explicit approval — "yes", "y", "confirm", "place the
order", "go ahead", "do it". If the cart is ready but the user hasn't said yes, do
NOTHING and stop — the writer will present the summary and ask. If the user pushes
back ("wait", "no", "actually change…"), handle that instead.

You don't speak to the user directly — a separate writer composes the reply. Do your
work via tool calls, then end your turn by replying with the single word DONE (an
internal marker the user never sees). Always produce that DONE line, even if you took
no action this turn.
"""

DESCRIPTION = (
    "Move an order forward: capture identity, shipping address, delivery option, and "
    "payment, then place the order ONLY on the user's explicit 'yes'. Requires items "
    "already in the cart. Pass the user's latest checkout-relevant message as `query` "
    "(their name, an address, a delivery choice, a payment method, or 'yes')."
)


def _instruction(ctx: ReadonlyContext) -> str:
    """Inject the authoritative checkout state on every run (the cart is the truth)."""
    key = ctx.state.get(registry.RUNTIME_KEY)
    if key is None:
        return PROMPT
    return PROMPT + "\n\n" + checkout_progress(registry.get(key).cart_service.cart)


agent = LlmAgent(
    name=CHECKOUT,
    model=make_model(),
    generate_content_config=gen_config(0.0),
    instruction=_instruction,
    tools=[
        tools.remove_item,
        tools.set_quantity,
        tools.set_customer,
        tools.set_address,
        tools.lookup_serviceability,
        tools.set_delivery_option,
        tools.quote_shipping,
        tools.compute_tax,
        tools.apply_promo,
        tools.attach_payment,
        tools.confirm_checkout,
        tools.get_cart_summary,
    ],
)


# =========================================================================== #
# the deterministic "Checkout progress" block
# (also consumed by the writer payload + the checkout block builder)
# =========================================================================== #
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


_NEXT_STEP_HINT = {
    "collecting_products": "items missing — this shouldn't happen mid-checkout.",
    "collecting_identity": "identity — capture the customer's name with set_customer.",
    "collecting_address": "address — capture the shipping address with set_address.",
    "awaiting_serviceability": "serviceability — call lookup_serviceability().",
    "collecting_delivery": "delivery — set_delivery_option the user chose, then quote_shipping() + compute_tax().",
    "collecting_payment": "payment — attach_payment with the user's method (card needs a token).",
    "awaiting_pricing": (
        "pricing — the cart changed, so the shipping quote and tax are stale. Recompute "
        "NOW yourself: call quote_shipping() then compute_tax(). Do NOT confirm yet — the "
        "refreshed total must be shown so the user can approve it."
    ),
    "ready_to_confirm": "ready — if the user's latest message is an explicit yes/confirm, call confirm_checkout(); otherwise do nothing.",
    "confirmed": "order already placed — do nothing.",
}


def checkout_progress(cart) -> str:
    """The authoritative 'what's done / what's next' block injected each run.

    The cart is the source of truth, so we render its state explicitly instead of
    making the model rediscover it from a growing thread. ``cart.step`` drives the
    single NEXT STEP.
    """
    c = cart

    def mark(done: bool, value: str) -> str:
        return f"✓ {value}".rstrip() if done else "— not provided"

    name = f"{c.customer.first_name or ''} {c.customer.last_name or ''}".strip()
    identity = mark(bool(c.customer.first_name), name)
    address = mark(c.address.is_complete(), f"{c.address.street}, {c.address.city} {c.address.zip_code}")
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
        pricing = "✗ STALE — cart changed; recompute with quote_shipping() then compute_tax()"
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
        f"Resume from: {_NEXT_STEP_HINT.get(c.step.value, 'the next missing field.')}\n"
        "Advance using the user's latest message + automatic internal steps; stop at "
        "the first field that needs info the user hasn't given."
    )


# =========================================================================== #
# hooks
# =========================================================================== #
def extract(deps: ShoppingDeps, run_events, before) -> StepResult:
    cart = deps.cart_service.cart
    asks: list[str] = []
    if cart.step.value.startswith("collecting_"):
        asks = asks_for_step(cart.step.value, cart)
    elif cart.ready_to_confirm() and not cart.confirmed:
        asks = ["explicit yes to place the order"]
    return StepResult(
        sop=CHECKOUT,
        summary=f"checkout finished at step={cart.step.value}; items={len(cart.items)}",
        asks=asks,
        next_sop=None,
    )


def _summarize(sr: StepResult, deps: ShoppingDeps) -> str:
    asks_note = f" Needs from user: {', '.join(sr.asks)}." if sr.asks else ""
    confirmed = " ORDER CONFIRMED." if deps.cart_service.cart.confirmed else ""
    return f"{sr.summary}.{asks_note}{confirmed}"


WORKER = Worker(
    name=CHECKOUT,
    agent=agent,
    extract=extract,
    prompt=PROMPT,
    block="checkout",
    summarize=_summarize,
)
