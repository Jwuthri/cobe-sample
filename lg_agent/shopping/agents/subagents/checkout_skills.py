"""The checkout sub-agent's per-step skills — loaded on demand, one per step.

Each skill is the detailed how-to for ONE checkout step. The base checkout prompt
stays lean; the agent loads the skill named for the current step (see
:data:`STEP_SKILL`) and only THEN can reach that step's tools — because each skill
``unlocks`` its tools, they are hidden until the skill is loaded. So progress is:

    load_skill('collect_identity')  →  set_customer(...)  →  (step advances)

This keeps the prompt small (step detail is loaded only when needed) and makes each
step visible in the UI as a ``load_skill`` call. The cart's ``step`` + ``blockers``
remain the real safety net; the gating just keeps the agent on rails one step at a
time.
"""

from __future__ import annotations

COLLECT_IDENTITY = """\
You are capturing the customer's name.

- If the user's latest message states a real first AND last name, call
  set_customer(first_name, last_name, email?).
- An address, a zip, or field-label words ("shipping", "address", "delivery",
  "payment") are NOT a name — never pass them to set_customer. If the message
  contains no actual name, do nothing and stop (the writer will ask for it).
"""

COLLECT_ADDRESS = """\
You are capturing the shipping address.

- If the user's message gives a full address (street + city + zip), call
  set_address(street, city, zip_code, state?, country?).
- CORRECTION CASE: if the progress block shows the saved address is NOT serviceable
  and the user gives a corrected value (often just a new ZIP), call set_address
  AGAIN — reuse the street/city already shown in the progress block and swap in the
  new value. E.g. saved "181823 ave oak, san francisco 93123" + user says "my zip is
  94123" → set_address("181823 ave oak", "san francisco", "94123").
- If the message is not an address and nothing needs correcting, do nothing and stop
  (the writer will ask).
- After set_address succeeds, load_skill('lookup_serviceability') to verify we
  ship there.
"""

LOOKUP_SERVICEABILITY = """\
You are verifying the address is serviceable (an internal step — no user input).

- Call lookup_serviceability().
- If it is NOT serviceable, stop — the writer will ask the user for a different
  address.
- If it IS serviceable, load_skill('collect_delivery').
"""

COLLECT_DELIVERY = """\
You are setting the delivery option and refreshing pricing.

- If the user picked an option (2h / 4h / next_day / standard) that is in the
  cart's serviceable_options, call set_delivery_option(option). If they did not
  pick one and none is set yet, do nothing and stop (the writer will ask).
- Once a delivery option is set (now or on a previous turn) but shipping/tax are
  stale, call quote_shipping() THEN compute_tax() so the grand total is fresh.
- When pricing is fresh, load_skill('collect_payment').
"""

COLLECT_PAYMENT = """\
You are capturing the payment method.

- If the user gave a method (card / cash / wallet), call attach_payment(method, …).
- For CARD: pass card_token ONLY if the user actually provided one (e.g. "my token
  is tok_77"). If they chose card but gave NO token, call attach_payment("card")
  with no token and stop — the writer will ask for it. NEVER invent or guess a
  token; do not copy any example value.
- If the message gives no payment method, do nothing and stop (the writer will ask).
- Once payment is set and the order is ready, load_skill('place_order').
"""

PLACE_ORDER = """\
You are placing the order — the final, irreversible step.

- Call confirm_checkout() ONLY if the user's LATEST message is an explicit
  approval: "yes", "y", "confirm", "place the order", "go ahead", "do it".
- If the cart is ready but the user has NOT said yes, do NOTHING — the writer will
  present the summary and ask for confirmation.
- If the user pushes back ("wait", "no", "change…"), handle that instead.
"""

# Skill config (validates against CustomSkillSpec). ``unlocks`` gates each step's
# tools behind its skill, so loading the skill is the only way to act on the step.
CHECKOUT_SKILLS: list[dict] = [
    {
        "kind": "custom",
        "name": "collect_identity",
        "description": "Capture the customer's first and last name.",
        "skill": COLLECT_IDENTITY,
        "unlocks": ["set_customer"],
    },
    {
        "kind": "custom",
        "name": "collect_address",
        "description": "Capture the shipping address (street, city, state, zip).",
        "skill": COLLECT_ADDRESS,
        "unlocks": ["set_address"],
    },
    {
        "kind": "custom",
        "name": "lookup_serviceability",
        "description": "Check which delivery options the address supports.",
        "skill": LOOKUP_SERVICEABILITY,
        "unlocks": ["lookup_serviceability"],
    },
    {
        "kind": "custom",
        "name": "collect_delivery",
        "description": "Pick a serviceable delivery option and quote shipping + tax.",
        "skill": COLLECT_DELIVERY,
        "unlocks": ["set_delivery_option", "quote_shipping", "compute_tax"],
    },
    {
        "kind": "custom",
        "name": "collect_payment",
        "description": "Capture the payment method.",
        "skill": COLLECT_PAYMENT,
        "unlocks": ["attach_payment"],
    },
    {
        "kind": "custom",
        "name": "place_order",
        "description": "Place the order on the user's explicit approval.",
        "skill": PLACE_ORDER,
        "unlocks": ["confirm_checkout"],
    },
]

# Which skill to load for each cart step. The cart_anchor surfaces this so the agent
# always knows which skill the current step needs.
STEP_SKILL: dict[str, str] = {
    "collecting_identity": "collect_identity",
    "collecting_address": "collect_address",
    "awaiting_serviceability": "lookup_serviceability",
    "collecting_delivery": "collect_delivery",
    "awaiting_pricing": "collect_delivery",  # recompute pricing — same tools
    "collecting_payment": "collect_payment",
    "ready_to_confirm": "place_order",
}

__all__ = ["CHECKOUT_SKILLS", "STEP_SKILL"]
