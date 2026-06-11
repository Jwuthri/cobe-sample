---
name: payments
description: >-
  How to collect payment and apply promo codes during checkout. Use when the
  checkout agent reaches the payment step, when the user picks a payment method,
  or when a promo/discount code is mentioned.
---

# payments

## Overview

Payment is the last field before confirmation. Capture it with
`attach_payment(method, card_token?)`.

## Payment methods

- Accepted methods: `card`, `cash`, `wallet`.
- `card` REQUIRES a `card_token`. It is mocked — any non-empty string works
  (e.g. `tok_42`). If the user chooses card but gives no token, ask for one;
  do not pretend a token exists.
- `cash` and `wallet` need no token.

Examples:

- "I'll pay by card, token tok_42" → `attach_payment("card", "tok_42")`
- "cash" → `attach_payment("cash")`

## Promo codes

If the user mentions a discount or promo code, apply it with `apply_promo(code)`.

- `WELCOME10` — 10% off the whole order.
- `SHOES20` — 20% off shoes; it fails if there are no shoes in the cart.

A promo only changes the total; it is not a checkout step and is optional. If
`apply_promo` returns an error (unknown code, or no qualifying items), relay that
the code could not be applied and continue — do not block the order on a promo.

## After payment

Once payment is attached and the cart has no blockers, the order is
ready_to_confirm. Do NOT auto-place it — the writer presents the total and the
user must explicitly approve. See the checkout-flow skill's confirmation rules.
