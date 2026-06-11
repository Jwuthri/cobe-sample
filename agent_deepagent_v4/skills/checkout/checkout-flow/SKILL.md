---
name: checkout-flow
description: >-
  The end-to-end checkout procedure for the shopping assistant. Use whenever the
  checkout agent is moving an order forward — capturing identity, address,
  serviceability, delivery, payment — or placing the order. Explains step order,
  which steps are internal vs need the user, and the confirmation safety rules.
---

# checkout-flow

## Overview

Checkout advances ONE order, one field at a time, in a fixed order. The cart is
the source of truth and persists every captured field across turns. Always start
by calling `checkout_progress()` to see what is already done (✓) and what is next.

## The step order

```
identity → address → serviceability → delivery → payment → confirm
```

Resume from the first field that is not yet ✓. Never re-capture a ✓ field — that
is the single most common mistake.

## Step → tool

| Step           | Tool(s)                                                        | Needs the user? |
| -------------- | ------------------------------------------------------------- | --------------- |
| identity       | `set_customer(first_name, last_name, email?)`                 | yes             |
| address        | `set_address(street, city, zip_code, state?, country?)`       | yes             |
| serviceability | `lookup_serviceability()`                                     | no — internal   |
| delivery       | `set_delivery_option(option)` → `quote_shipping()` → `compute_tax()` | yes (the option) |
| payment        | `attach_payment(method, card_token?)`  (card needs a token)   | yes             |
| confirm        | `confirm_checkout()`                                          | yes (explicit)  |

## Internal vs user-driven steps

- **Internal** steps need no input — perform them automatically the moment you
  reach them: `lookup_serviceability()` right after an address is set, and
  `quote_shipping()` + `compute_tax()` right after a delivery option is set.
- **User-driven** steps (delivery option, payment method, final confirmation) use
  the user's LATEST message. If the message does not provide the answer, STOP and
  do nothing further this turn — the writer will ask. Never invent the choice.

## Serviceability

If `lookup_serviceability()` reports the address is NOT serviceable, do not pick a
delivery option. The order is blocked until a serviceable address is set — ask the
user for a different address (the product/checkout split means the writer relays
this; you just stop).

## Confirmation safety (read carefully)

- Call `confirm_checkout()` ONLY when the user's latest message is an explicit
  approval: "yes", "y", "confirm", "place the order", "go ahead", "do it".
- If the cart is ready but the user has not said yes, do nothing — the writer
  presents the summary and asks for a yes.
- `confirm_checkout()` will refuse if any blocker remains, and then PAUSE for a
  final human approval before charging. That pause is expected. The order is only
  placed — and a receipt id only exists — after that approval. Never claim the
  order is placed before then.

See `references` in the payments skill for payment-method specifics.
