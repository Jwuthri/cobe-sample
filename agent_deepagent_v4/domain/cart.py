"""Cart model: step tracker, quote-freshness fingerprints, and the blocker gate.

The cart is the **source of truth** for checkout safety:

* ``Cart.step`` derives the single next action (collect identity → address →
  serviceability → delivery → payment → confirm).
* Shipping/tax quotes carry the fingerprint of the inputs they were computed
  for, so any cart mutation makes a stale quote detectable.
* ``Cart.blockers()`` is the invariant safety net — ``CartService.confirm``
  refuses to place an order while any blocker remains, so ``cart.confirmed``
  (never the model's prose) is what the writer reports.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DeliveryOption = Literal["2h", "4h", "next_day", "standard"]
PaymentMethod = Literal["card", "cash", "wallet"]


# ---------- atoms ----------
class Blocker(BaseModel):
    code: str
    message: str


class CartItem(BaseModel):
    product_id: str
    name: str
    unit_price: Decimal
    quantity: int
    tags: list[str] = Field(default_factory=list)

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity


class Customer(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None

    def is_complete(self) -> bool:
        return bool(self.first_name and self.last_name)


class Address(BaseModel):
    street: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    country: str = "US"

    def is_complete(self) -> bool:
        return all([self.street, self.city, self.zip_code, self.country])


class ShippingFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)
    items_signature: str
    zip_code: str | None
    delivery_option: DeliveryOption | None


class TaxFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)
    items_signature: str
    zip_code: str | None


class ShippingQuote(BaseModel):
    cost: Decimal
    eta_hours: int
    quoted_for: ShippingFingerprint


class TaxQuote(BaseModel):
    rate: Decimal
    amount: Decimal
    quoted_for: TaxFingerprint


class PromoApplication(BaseModel):
    code: str
    discount: Decimal
    applied_to_skus: list[str]


def _items_signature(items: list[CartItem]) -> str:
    pairs = sorted((i.product_id, i.quantity) for i in items)
    return hashlib.sha1(json.dumps(pairs).encode()).hexdigest()[:12]


# ---------- step tracker ----------
class CheckoutStep(str, Enum):
    COLLECTING_PRODUCTS = "collecting_products"
    COLLECTING_IDENTITY = "collecting_identity"
    COLLECTING_ADDRESS = "collecting_address"
    AWAITING_SERVICEABILITY = "awaiting_serviceability"
    COLLECTING_DELIVERY = "collecting_delivery"
    COLLECTING_PAYMENT = "collecting_payment"
    READY_TO_CONFIRM = "ready_to_confirm"
    CONFIRMED = "confirmed"


# ---------- cart ----------
class Cart(BaseModel):
    cart_id: str | None = None
    items: list[CartItem] = Field(default_factory=list)

    customer: Customer = Field(default_factory=Customer)
    address: Address = Field(default_factory=Address)

    serviceable: bool | None = None
    serviceable_options: list[DeliveryOption] = Field(default_factory=list)

    delivery_option: DeliveryOption | None = None

    shipping: ShippingQuote | None = None
    tax: TaxQuote | None = None
    promo: PromoApplication | None = None

    payment_method: PaymentMethod | None = None
    card_token: str | None = None

    confirmed: bool = False
    receipt_id: str | None = None

    # ----- derived -----
    @property
    def subtotal(self) -> Decimal:
        return sum((i.line_total for i in self.items), start=Decimal("0"))

    def items_signature(self) -> str:
        return _items_signature(self.items)

    def shipping_fingerprint(self) -> ShippingFingerprint:
        return ShippingFingerprint(
            items_signature=self.items_signature(),
            zip_code=self.address.zip_code,
            delivery_option=self.delivery_option,
        )

    def tax_fingerprint(self) -> TaxFingerprint:
        return TaxFingerprint(items_signature=self.items_signature(), zip_code=self.address.zip_code)

    @property
    def promo_discount(self) -> Decimal:
        return self.promo.discount if self.promo and self.promo_is_valid() else Decimal("0")

    @property
    def grand_total(self) -> Decimal | None:
        if not (self.shipping_is_fresh() and self.tax_is_fresh()):
            return None
        return self.subtotal + self.shipping.cost + self.tax.amount - self.promo_discount

    @property
    def step(self) -> CheckoutStep:
        """Single source of truth for 'what the checkout agent should do next'."""
        if not self.items:
            return CheckoutStep.COLLECTING_PRODUCTS
        if not self.customer.is_complete():
            return CheckoutStep.COLLECTING_IDENTITY
        if not self.address.is_complete():
            return CheckoutStep.COLLECTING_ADDRESS
        if self.serviceable is None:
            return CheckoutStep.AWAITING_SERVICEABILITY
        if not self.delivery_option:
            return CheckoutStep.COLLECTING_DELIVERY
        if not self.payment_method or (self.payment_method == "card" and not self.card_token):
            return CheckoutStep.COLLECTING_PAYMENT
        if self.confirmed:
            return CheckoutStep.CONFIRMED
        return CheckoutStep.READY_TO_CONFIRM

    # ----- freshness -----
    def shipping_is_fresh(self) -> bool:
        return self.shipping is not None and self.shipping.quoted_for == self.shipping_fingerprint()

    def tax_is_fresh(self) -> bool:
        return self.tax is not None and self.tax.quoted_for == self.tax_fingerprint()

    def promo_is_valid(self) -> bool:
        if not self.promo:
            return True
        current = {i.product_id for i in self.items}
        return all(sku in current for sku in self.promo.applied_to_skus)

    # ----- the gate -----
    def blockers(self) -> list[Blocker]:
        out: list[Blocker] = []
        if not self.items:
            out.append(Blocker(code="empty_cart", message="Cart is empty."))
        if not self.customer.is_complete():
            out.append(Blocker(code="missing_identity", message="Need first and last name."))
        if not self.address.is_complete():
            out.append(Blocker(code="missing_address", message="Need full shipping address (street, city, zip)."))
        if self.address.is_complete() and self.serviceable is None:
            out.append(Blocker(code="missing_serviceability", message="Address not checked for serviceability."))
        if self.serviceable is False:
            out.append(Blocker(code="not_serviceable", message=f"We don't ship to zip {self.address.zip_code}."))
        if self.serviceable and not self.delivery_option:
            out.append(Blocker(code="missing_delivery_option", message="Need delivery option."))
        if (
            self.delivery_option
            and self.serviceable_options
            and self.delivery_option not in self.serviceable_options
        ):
            out.append(
                Blocker(
                    code="unserviceable_delivery_option",
                    message=(
                        f"{self.delivery_option} not available for {self.address.zip_code}. "
                        f"Available: {self.serviceable_options}."
                    ),
                )
            )
        if self.items and self.address.is_complete() and self.delivery_option and not self.shipping_is_fresh():
            out.append(Blocker(code="stale_shipping", message="Shipping quote stale/missing — call quote_shipping."))
        if self.items and self.address.is_complete() and not self.tax_is_fresh():
            out.append(Blocker(code="stale_tax", message="Tax not computed for current cart — call compute_tax."))
        if not self.payment_method:
            out.append(Blocker(code="missing_payment", message="Need payment method."))
        if self.payment_method == "card" and not self.card_token:
            out.append(Blocker(code="missing_card_token", message="Card selected but no card token attached."))
        if self.promo and not self.promo_is_valid():
            out.append(Blocker(code="invalid_promo", message=f"Promo {self.promo.code} no longer applies."))
        return out

    def ready_to_confirm(self) -> bool:
        return not self.blockers()
