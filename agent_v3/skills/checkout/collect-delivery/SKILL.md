---
name: collect-delivery
description: Pick a serviceable delivery option and quote shipping + tax. Use after serviceability is confirmed.
---

# Collect delivery

You are picking a delivery option.

1. Present the available options from the cart (only those in
   `serviceable_options` are valid). Briefly note speed vs cost.
2. Call `set_delivery_option(option)` with the user's pick.
3. Call `quote_shipping()` and `compute_tax()` so we have a full grand
   total to show.
4. Load the `collect-payment` skill
   (`get_skill_instructions("collect-payment")`).
