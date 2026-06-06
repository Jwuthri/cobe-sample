---
name: collect-address
description: Capture the shipping address (street, city, state, zip). Use after the customer's name is captured.
---

# Collect address

You are collecting the shipping address.

1. If the runtime has a saved address (visible in the system prompt under
   "Saved addresses"), offer it as the default. Otherwise ask for street,
   city, state (US only) and zip code.
2. Call `set_address(street, city, zip_code, state?, country?)` once you
   have a complete address.
3. Then load the `lookup-serviceability` skill
   (`get_skill_instructions("lookup-serviceability")`) to verify we ship there.

Do NOT proceed until `set_address` has been called and you've loaded the
next skill.
