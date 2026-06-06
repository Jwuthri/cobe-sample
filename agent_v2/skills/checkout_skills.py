"""The five sub-skills the checkout subagent loads in order.

Loading order is encoded in the skill descriptions, not enforced by
code — but each skill's ``unlocks`` ensures the model can't run ahead
of itself. Combined with ``Cart.step``, the agent has clear forward
progress and can't skip a step.
"""

from __future__ import annotations

from agent_v2.skills.base import Skill

COLLECT_IDENTITY_PROMPT = """\
You are collecting the customer's identity.

1. Ask for the customer's first and last name (and optionally email).
2. Call set_customer(first_name, last_name, email?) once you have them.
3. Acknowledge briefly, then load_skill('collect_address').

Do NOT proceed to the address step until set_customer has been called.
"""

COLLECT_ADDRESS_PROMPT = """\
You are collecting the shipping address.

1. If the runtime has a saved address (visible in the system prompt
   under 'Saved addresses'), offer it as the default. Otherwise ask
   for street, city, state (US only) and zip code.
2. Call set_address(street, city, zip_code, state?, country?) once
   you have a complete address.
3. Then load_skill('lookup_serviceability') to verify we ship there.

Do NOT proceed until set_address has been called and you've loaded
the next skill.
"""

LOOKUP_SERVICEABILITY_PROMPT = """\
You are verifying the address is serviceable.

1. Call lookup_serviceability(). The result tells you which delivery
   options are available for this zip.
2. If the address is not serviceable, apologize and ask the user for
   a different address — go back and call set_address again, then
   re-lookup.
3. Once serviceable, load_skill('collect_delivery').
"""

COLLECT_DELIVERY_PROMPT = """\
You are picking a delivery option.

1. Present the available options from the cart (only those in
   serviceable_options are valid). Briefly note speed vs cost.
2. Call set_delivery_option(option) with the user's pick.
3. Call quote_shipping() and compute_tax() so we have a full grand
   total to show.
4. Load_skill('collect_payment').
"""

COLLECT_PAYMENT_PROMPT = """\
You are collecting payment.

1. Ask the user how they'd like to pay (card, cash, or wallet). If
   card, ask for a token (mocked — any string like 'tok_42' is fine).
2. Call attach_payment(method, card_token?).
3. Call get_cart_summary() and present the grand total to the user.
4. Once the user explicitly confirms they want to place the order,
   call confirm_checkout().

The confirm_checkout call WILL pause execution for the user to
approve the charge. That is expected behavior — make sure you call
it only after the user has agreed to place the order.
"""

CHECKOUT_SKILLS: list[Skill] = [
    {
        "name": "collect_identity",
        "description": "Capture the customer's first and last name.",
        "content": COLLECT_IDENTITY_PROMPT,
        "unlocks": ["set_customer"],
    },
    {
        "name": "collect_address",
        "description": "Capture the shipping address (street, city, state, zip).",
        "content": COLLECT_ADDRESS_PROMPT,
        "unlocks": ["set_address"],
    },
    {
        "name": "lookup_serviceability",
        "description": "Check which delivery options the address supports.",
        "content": LOOKUP_SERVICEABILITY_PROMPT,
        "unlocks": ["lookup_serviceability"],
    },
    {
        "name": "collect_delivery",
        "description": "Pick a serviceable delivery option and quote shipping + tax.",
        "content": COLLECT_DELIVERY_PROMPT,
        "unlocks": ["set_delivery_option", "quote_shipping", "compute_tax"],
    },
    {
        "name": "collect_payment",
        "description": "Capture payment method and place the order on user approval.",
        "content": COLLECT_PAYMENT_PROMPT,
        "unlocks": ["attach_payment", "confirm_checkout"],
    },
]
