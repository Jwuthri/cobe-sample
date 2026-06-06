"""SOP identifiers + the supervisor's structured-output decision shape.

Extracted into its own module (in agent_v2 these lived in ``supervisor``)
so that ``step_result`` and ``state`` can import them without pulling in
the model/agent machinery.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SOPName(str, Enum):
    CHECKOUT = "checkout"
    ORDER_STATUS = "order_status"
    PRODUCT_REC = "product_rec"


class SupervisorDecision(BaseModel):
    """Structured-output shape returned by the classifier agent."""

    done: bool = Field(description="True when no more SOP work is needed this turn.")
    next_sop: SOPName | None = Field(
        default=None, description="Required when done=False; ignored otherwise."
    )
    reason: str = Field(default="", description="One sentence justification.")
