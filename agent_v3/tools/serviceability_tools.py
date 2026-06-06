"""Stateless serviceability tool — answers 'do you deliver to X?'

The *anonymous* lookup, available to product_rec for pre-purchase
questions. The checkout flow has its own constrained version
(``lookup_serviceability`` in checkout_tools) that mutates cart state;
both share the same backing table (``agent_v3.checkout.serviceability``).
"""

from __future__ import annotations

from agent_v3.checkout.serviceability import lookup


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
