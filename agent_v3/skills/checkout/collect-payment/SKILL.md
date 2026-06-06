---
name: collect-payment
description: Capture payment method and place the order on explicit user approval. The final checkout step.
---

# Collect payment

You are collecting payment — the final step.

1. Ask the user how they'd like to pay (card, cash, or wallet). If card,
   ask for a token (mocked — any string like 'tok_42' is fine).
2. Call `attach_payment(method, card_token?)`.
3. Call `get_cart_summary()` so the writer can present the grand total.
4. STOP and let the writer present the order summary and ask the user to
   confirm.

## Confirmation rule (read carefully)

NEVER call `confirm_checkout` automatically when the cart becomes
ready_to_confirm. Only call `confirm_checkout` on a SUBSEQUENT turn, once
the user's most recent message is an explicit approval like "yes", "y",
"confirm", "place the order", "go ahead". If the user pushes back ("wait",
"no", "change…"), do NOT confirm — handle their request instead.
