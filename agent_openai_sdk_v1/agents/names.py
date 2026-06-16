"""The three worker names — one shared vocabulary, no imports.

Each string is used as the worker's delegate-tool name, its ``StepResult.sop``, its
writer-block key, and the frontend ``step`` event's ``sop`` field. Keeping them in a
leaf module lets every consumer agree without a circular import.
"""

from __future__ import annotations

PRODUCT_REC = "product_rec"
CHECKOUT = "checkout"
ORDER_STATUS = "order_status"

ALL = (PRODUCT_REC, CHECKOUT, ORDER_STATUS)
