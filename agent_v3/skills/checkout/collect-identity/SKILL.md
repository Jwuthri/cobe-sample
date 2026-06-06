---
name: collect-identity
description: Capture the customer's first and last name (the first checkout step). Use when the cart has items but no customer name yet.
---

# Collect identity

You are collecting the customer's identity — the first checkout step.

1. Ask for the customer's first and last name (and optionally email).
2. Call `set_customer(first_name, last_name, email?)` once you have them.
3. Acknowledge briefly, then load the `collect-address` skill
   (`get_skill_instructions("collect-address")`).

Do NOT proceed to the address step until `set_customer` has been called.
