"""Mock serviceability lookup: a zip code → which delivery options we serve there."""

from __future__ import annotations

from typing import NamedTuple

from google_adk_agent_v1.domain.cart import DeliveryOption


class Serviceability(NamedTuple):
    zip_code: str
    city: str
    country: str
    options: tuple[DeliveryOption, ...]


# zip prefix → region details
_TABLE: dict[str, Serviceability] = {
    "941": Serviceability("941", "San Francisco", "US", ("2h", "4h", "next_day", "standard")),
    "100": Serviceability("100", "New York", "US", ("4h", "next_day", "standard")),
    "750": Serviceability("750", "Paris", "FR", ("next_day", "standard")),
    "900": Serviceability("900", "Los Angeles", "US", ("4h", "next_day", "standard")),
}


def lookup(zip_code: str) -> Serviceability | None:
    """Return the Serviceability for a zip's region, or None if we don't ship there."""
    if not zip_code:
        return None
    return _TABLE.get(zip_code[:3])
