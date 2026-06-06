"""Stateless serviceability tool — answers 'do you deliver to X?'

This is the *anonymous* lookup, available to product_rec for
pre-purchase questions. There's also a constrained version inside
checkout (``lookup_serviceability``) that mutates cart state. They
share the same backing table (``agent_v2.checkout.serviceability``)
but live in different places because their concerns differ:

  - this tool answers the user's question and exits.
  - the checkout tool sets ``cart.serviceable`` + ``serviceable_options``
    so the cart's blockers logic can act on it.
"""

from __future__ import annotations

from agent_v2.checkout.serviceability import lookup
from langchain_core.tools import tool


@tool
def check_serviceability(zip_code: str) -> str:
    """Check whether we ship to a given zip code and which delivery
    options are available there. Use this for questions like 'do you
    deliver to San Francisco?' or 'what shipping do you have for 94110?'."""
    z = (zip_code or "").strip()
    if not z:
        return "I need a zip code to check serviceability."
    result = lookup(z)
    if result is None:
        return f"We don't currently ship to zip {z}."
    options = ", ".join(result.options)
    return (
        f"Yes, we ship to zip {z} ({result.city}, {result.country}). "
        f"Available delivery options: {options}."
    )
