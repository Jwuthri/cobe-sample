"""Pricing tables (stubs). Pure-functional, no mutation."""

from __future__ import annotations

from decimal import Decimal

from lg_agent.shopping.domain.cart import CartItem, DeliveryOption

# ------ shipping ------
SHIPPING_TIERS: dict[tuple[str, DeliveryOption], tuple[Decimal, int]] = {
    ("941", "2h"): (Decimal("19.99"), 2),
    ("941", "4h"): (Decimal("9.99"), 4),
    ("941", "next_day"): (Decimal("4.99"), 24),
    ("941", "standard"): (Decimal("0.00"), 72),
    ("750", "next_day"): (Decimal("14.99"), 24),
    ("750", "standard"): (Decimal("3.99"), 96),
}
DEFAULT_SHIPPING_BY_OPTION: dict[DeliveryOption, tuple[Decimal, int]] = {
    "2h": (Decimal("29.99"), 2),
    "4h": (Decimal("14.99"), 4),
    "next_day": (Decimal("9.99"), 24),
    "standard": (Decimal("4.99"), 96),
}


def quote_shipping(zip_code: str, option: DeliveryOption, subtotal: Decimal) -> tuple[Decimal, int]:
    """Return (cost, eta_hours). Free shipping over $100 on standard."""
    prefix = zip_code[:3]
    cost, eta = SHIPPING_TIERS.get((prefix, option), DEFAULT_SHIPPING_BY_OPTION[option])
    if option == "standard" and subtotal >= Decimal("100"):
        cost = Decimal("0.00")
    return cost, eta


# ------ tax ------
TAX_RATES: dict[str, Decimal] = {
    "94": Decimal("0.0875"),  # CA-ish
    "75": Decimal("0.20"),  # FR VAT
    "00": Decimal("0.00"),  # non-serviceable
}


def quote_tax(zip_code: str, subtotal: Decimal) -> tuple[Decimal, Decimal]:
    """Return (rate, amount)."""
    rate = TAX_RATES.get(zip_code[:2], Decimal("0.05"))
    amount = (subtotal * rate).quantize(Decimal("0.01"))
    return rate, amount


# ------ promo ------
class PromoSpec:
    def __init__(
        self,
        code: str,
        discount_pct: Decimal,
        applies_to_tags: list[str] | None = None,
    ) -> None:
        self.code = code
        self.discount_pct = discount_pct
        self.applies_to_tags = applies_to_tags  # None == applies to entire subtotal

    def applicable_items(self, items: list[CartItem]) -> list[CartItem]:
        if self.applies_to_tags is None:
            return items
        tagset = set(self.applies_to_tags)
        return [i for i in items if tagset.intersection(i.tags)]


PROMOS: dict[str, PromoSpec] = {
    "WELCOME10": PromoSpec("WELCOME10", Decimal("0.10")),
    "SHOES20": PromoSpec("SHOES20", Decimal("0.20"), applies_to_tags=["shoes"]),
}


def quote_promo(code: str, items: list[CartItem]) -> tuple[Decimal, list[str]]:
    """Return (discount_amount, skus_the_discount_was_computed_against).

    Raises ``KeyError`` for unknown codes, ``ValueError`` when the code matches but
    no current items qualify (e.g. SHOES20 with no shoes).
    """
    spec = PROMOS[code.upper()]
    qualifying = spec.applicable_items(items)
    if not qualifying:
        raise ValueError(f"promo {code} requires items tagged {spec.applies_to_tags}")
    qualifying_subtotal = sum((i.line_total for i in qualifying), start=Decimal("0"))
    discount = (qualifying_subtotal * spec.discount_pct).quantize(Decimal("0.01"))
    return discount, [i.product_id for i in qualifying]
