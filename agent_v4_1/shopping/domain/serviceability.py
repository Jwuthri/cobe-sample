"""Mock serviceability lookup. Self-contained copy.

Given a zip code, returns which delivery options we serve.
"""

from __future__ import annotations

from typing import NamedTuple

from agent_v4_1.shopping.domain.cart import DeliveryOption


class Serviceability(NamedTuple):
    zip_code: str
    city: str
    country: str
    options: tuple[DeliveryOption, ...]


# zip prefix → details
_TABLE: dict[str, Serviceability] = {
    "941": Serviceability("941", "San Francisco", "US", ("2h", "4h", "next_day", "standard")),
    "100": Serviceability("100", "New York", "US", ("4h", "next_day", "standard")),
    "750": Serviceability("750", "Paris", "FR", ("next_day", "standard")),
    "900": Serviceability("900", "Los Angeles", "US", ("4h", "next_day", "standard")),
}


def lookup(zip_code: str) -> Serviceability | None:
    """Return Serviceability for the zip's region, or None if we don't ship there."""
    if not zip_code:
        return None
    prefix = zip_code[:3]
    return _TABLE.get(prefix)
