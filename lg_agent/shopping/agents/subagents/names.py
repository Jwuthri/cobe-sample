"""The three sub-agent names — the single shared vocabulary.

These strings are used as the tool name, the ``StepResult.sop``, the block key, and
the frontend ``step`` event's ``sop`` field. Keeping them in one leaf module (no
other imports) lets every consumer agree without a circular import.
"""

from __future__ import annotations

PRODUCT_REC = "product_rec"
CHECKOUT = "checkout"
ORDER_STATUS = "order_status"

ALL = (PRODUCT_REC, CHECKOUT, ORDER_STATUS)

__all__ = ["PRODUCT_REC", "CHECKOUT", "ORDER_STATUS", "ALL"]
