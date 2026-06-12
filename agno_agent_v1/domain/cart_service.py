"""All cart mutations funnel through ``CartService``.

Each mutator updates the targeted fields, invalidates any derived quotes whose
inputs changed, and returns a short event string. The *invalidation policy* lives
here; the Agno tools (``agno_agent_v1/agent/tools.py``) are dumb wrappers around
these methods. This separation is what keeps totals/taxes correct under
backtracking: edit the cart and the stale shipping/tax quotes are dropped, so the
cart's ``step`` becomes ``awaiting_pricing`` and the grand total goes ``None``
until re-quoted.
"""

from __future__ import annotations

import itertools

from agno_agent_v1.domain.cart import (
    Cart,
    DeliveryOption,
    PaymentMethod,
    PromoApplication,
    ShippingQuote,
    TaxQuote,
)
from agno_agent_v1.domain.catalog import get as get_product
from agno_agent_v1.domain.pricing import quote_promo, quote_shipping, quote_tax
from agno_agent_v1.domain.serviceability import lookup as lookup_zip

_CART_COUNTER = itertools.count(1000)
_RECEIPT_COUNTER = itertools.count(9000)


def _next_cart_id() -> str:
    return f"CART-{next(_CART_COUNTER)}"


def _next_receipt_id() -> str:
    return f"RCPT-{next(_RECEIPT_COUNTER)}"


class CartError(Exception):
    """Raised by service mutators when a requested mutation is illegal."""


# Words that appear in checkout prompts / routing labels but are never a person's
# name. The LLM occasionally confabulates identity from these (e.g. relabels an
# address message "Shipping address: …" and sets first/last = "Shipping"/"address").
# This deterministic guard is the LLM-agnostic backstop for that failure mode.
_NON_NAME_TOKENS = frozenset(
    {
        "shipping", "address", "delivery", "payment", "checkout", "name", "first",
        "last", "cart", "order", "customer", "billing", "email", "phone", "zip",
        "zipcode", "city", "street", "state", "country", "method", "none", "unknown",
    }
)


def _looks_like_name(value: str) -> bool:
    """A cheap, deterministic 'is this plausibly a name?' gate (not validation)."""
    v = (value or "").strip()
    if not v:
        return False
    if any(ch.isdigit() for ch in v):  # names have no digits; addresses do
        return False
    tokens = v.lower().split()
    if tokens and all(t in _NON_NAME_TOKENS for t in tokens):  # all label words → not a name
        return False
    return True


class CartService:
    def __init__(self, cart: Cart | None = None) -> None:
        self.cart = cart or Cart()
        if self.cart.cart_id is None:
            self.cart.cart_id = _next_cart_id()

    # ----- products -----
    def add_item(self, product_id: str, quantity: int = 1) -> str:
        if quantity < 1:
            raise CartError("quantity must be >= 1")
        product = get_product(product_id)
        if not product:
            raise CartError(f"unknown product: {product_id}")
        for existing in self.cart.items:
            if existing.product_id == product.id:
                existing.quantity += quantity
                break
        else:
            self.cart.items.append(product.to_cart_item(quantity=quantity))
        self._invalidate_items_dependent()
        self._revalidate_promo()
        return f"Added {quantity} × {product.name} (cart now ${self.cart.subtotal:.2f})."

    def remove_item(self, product_id: str) -> str:
        before = len(self.cart.items)
        self.cart.items = [i for i in self.cart.items if i.product_id != product_id.upper()]
        if len(self.cart.items) == before:
            raise CartError(f"product not in cart: {product_id}")
        self._invalidate_items_dependent()
        self._revalidate_promo()
        return f"Removed {product_id} (cart now ${self.cart.subtotal:.2f})."

    def set_quantity(self, product_id: str, quantity: int) -> str:
        if quantity < 0:
            raise CartError("quantity must be >= 0")
        if quantity == 0:
            return self.remove_item(product_id)
        for existing in self.cart.items:
            if existing.product_id == product_id.upper():
                existing.quantity = quantity
                self._invalidate_items_dependent()
                self._revalidate_promo()
                return f"Set {product_id} quantity to {quantity}."
        raise CartError(f"product not in cart: {product_id}")

    # ----- identity -----
    def set_customer(self, first_name: str, last_name: str, email: str | None = None) -> str:
        if not _looks_like_name(first_name) or not _looks_like_name(last_name):
            raise CartError(
                f"'{first_name} {last_name}' is not a valid customer name (it looks "
                "like a field label or address, not a person). Do NOT guess a name — "
                "leave identity empty and ask the user for their actual first and last "
                "name."
            )
        self.cart.customer.first_name = first_name.strip()
        self.cart.customer.last_name = last_name.strip()
        if email is not None:
            self.cart.customer.email = email.strip()
        return f"Customer set to {first_name} {last_name}."

    # ----- address (structured) -----
    def set_address(
        self,
        street: str,
        city: str,
        zip_code: str,
        state: str | None = None,
        country: str = "US",
    ) -> str:
        prev_zip = self.cart.address.zip_code
        self.cart.address.street = street.strip()
        self.cart.address.city = city.strip()
        self.cart.address.state = (state or "").strip() or None
        self.cart.address.zip_code = zip_code.strip()
        self.cart.address.country = country.strip().upper()
        # Changing the zip invalidates shipping, tax, AND serviceability.
        if prev_zip != zip_code:
            self.cart.shipping = None
            self.cart.tax = None
            self.cart.serviceable = None
            self.cart.serviceable_options = []
            self.cart.delivery_option = None
        return (
            f"Address set: {street}, {city} {zip_code} {country}. "
            f"Serviceability + shipping + tax invalidated."
        )

    # ----- serviceability -----
    def lookup_serviceability(self) -> str:
        if not self.cart.address.zip_code:
            raise CartError("need a zip code before checking serviceability")
        result = lookup_zip(self.cart.address.zip_code)
        if result is None:
            self.cart.serviceable = False
            self.cart.serviceable_options = []
            return f"Zip {self.cart.address.zip_code} is not serviceable."
        self.cart.serviceable = True
        self.cart.serviceable_options = list(result.options)
        return (
            f"Zip {self.cart.address.zip_code} is serviceable. "
            f"Available delivery options: {result.options}."
        )

    # ----- delivery -----
    def set_delivery_option(self, option: DeliveryOption) -> str:
        if self.cart.serviceable_options and option not in self.cart.serviceable_options:
            raise CartError(
                f"{option} not available for {self.cart.address.zip_code}. "
                f"Pick one of {self.cart.serviceable_options}."
            )
        if self.cart.delivery_option != option:
            self.cart.shipping = None
        self.cart.delivery_option = option
        return f"Delivery option set to {option}. Shipping invalidated."

    # ----- quotes -----
    def quote_shipping(self) -> str:
        if not self.cart.address.zip_code or not self.cart.delivery_option:
            raise CartError("need zip and delivery_option before quoting shipping")
        cost, eta = quote_shipping(
            self.cart.address.zip_code, self.cart.delivery_option, self.cart.subtotal
        )
        self.cart.shipping = ShippingQuote(
            cost=cost, eta_hours=eta, quoted_for=self.cart.shipping_fingerprint()
        )
        return f"Shipping quoted: ${cost:.2f}, ETA ~{eta}h."

    def compute_tax(self) -> str:
        if not self.cart.address.zip_code:
            raise CartError("need zip before computing tax")
        rate, amount = quote_tax(self.cart.address.zip_code, self.cart.subtotal)
        self.cart.tax = TaxQuote(rate=rate, amount=amount, quoted_for=self.cart.tax_fingerprint())
        return f"Tax computed: ${amount:.2f} ({rate * 100:.2f}%)."

    # ----- promo -----
    def apply_promo(self, code: str) -> str:
        try:
            discount, skus = quote_promo(code, self.cart.items)
        except KeyError:
            raise CartError(f"unknown promo code: {code}") from None
        except ValueError as e:
            raise CartError(str(e)) from None
        self.cart.promo = PromoApplication(
            code=code.upper(), discount=discount, applied_to_skus=skus
        )
        return f"Promo {code.upper()} applied: -${discount:.2f}."

    def clear_promo(self) -> str:
        if self.cart.promo is None:
            return "No promo to clear."
        code = self.cart.promo.code
        self.cart.promo = None
        return f"Promo {code} cleared."

    # ----- payment -----
    def attach_payment(self, method: PaymentMethod, card_token: str | None = None) -> str:
        prev = self.cart.payment_method
        self.cart.payment_method = method
        if method != "card":
            self.cart.card_token = None
        elif card_token:
            self.cart.card_token = card_token
        notes = []
        notes.append(f"switched from {prev or 'none'} to {method}" if prev != method else f"payment {method}")
        if method == "card" and not self.cart.card_token:
            notes.append("still need a card_token")
        return "Payment: " + ", ".join(notes) + "."

    # ----- confirmation -----
    def confirm(self) -> str:
        blockers = self.cart.blockers()
        if blockers:
            raise CartError("cannot confirm — blockers: " + "; ".join(b.code for b in blockers))
        self.cart.confirmed = True
        self.cart.receipt_id = _next_receipt_id()
        return f"Confirmed! Receipt {self.cart.receipt_id}. Total ${self.cart.grand_total:.2f}."

    # ----- internal invalidation policy -----
    def _invalidate_items_dependent(self) -> None:
        self.cart.shipping = None
        self.cart.tax = None

    def _revalidate_promo(self) -> None:
        if self.cart.promo and not self.cart.promo_is_valid():
            self.cart.promo = None
