"""Mock serviceability lookup: given a zip, which delivery options we serve."""

from __future__ import annotations

from typing import NamedTuple

# Delivery options we support, narrowest type lives here to avoid a cart import.
DeliveryOptionT = str


class Serviceability(NamedTuple):
    zip_code: str
    city: str
    country: str
    options: tuple[str, ...]


# zip prefix -> region details
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
    return _TABLE.get(zip_code[:3])
