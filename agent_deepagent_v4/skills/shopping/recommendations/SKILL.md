---
name: recommendations
description: >-
  How the product agent resolves vague product references and decides when an
  add-to-cart should lead toward checkout. Use when browsing, recommending
  products, or interpreting phrases like "the cheaper one" or "add those".
---

# recommendations

## Overview

The product agent browses the catalog and edits cart contents. This skill covers
reference resolution and the browse → buy handoff.

## Resolving vague references

Use the recent conversation to resolve what the customer means before acting:

- If you JUST showed a single product, "it" / "that" / "them" / "the X" = that product.
- If you showed several, "the cheaper one" / "the red one" / "the second one" /
  "the X one" selects by attribute. Compare prices/tags to choose.
- If you genuinely cannot tell which product is meant, ASK before adding —
  never guess an id.
- Product ids are case- and hyphen-insensitive in speech: "p3", "P3", "p-3",
  "P-3" all mean `P-3`. Always pass the canonical `P-N` form to the tools.

## Browse vs cart edit

- "remove the hoodie" / "take P-2 out" / "make it 2" / "what's in my cart" are
  CART edits or questions — use `view_cart`, `remove_item`, `set_quantity`. Do
  NOT search the catalog to remove or change something already in the cart.
- "what do you sell" / "find me a cap" / "tell me about P-2" are catalog lookups —
  use `search_products` / `get_product`.

## After adding an item

Once you add an item, the customer is on the path to buying. Report what you added
(id, name, price) and that it's in the cart. You do not run checkout yourself —
the orchestrator routes the next step to the checkout agent.

## Never

- Never invent products or ids. Only mention what the tools returned.
- Order ids (ORD-* / RCPT-*) are not products — ignore them; a different agent
  handles order status.
