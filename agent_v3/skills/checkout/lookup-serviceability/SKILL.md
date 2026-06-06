---
name: lookup-serviceability
description: Check which delivery options the saved address supports. Use right after the address is captured.
---

# Lookup serviceability

You are verifying the address is serviceable.

1. Call `lookup_serviceability()`. The result tells you which delivery
   options are available for this zip.
2. If the address is not serviceable, apologize and ask the user for a
   different address — go back and call `set_address` again (after loading
   the `collect-address` skill), then re-lookup.
3. Once serviceable, load the `collect-delivery` skill
   (`get_skill_instructions("collect-delivery")`).
