"""Stable SOP (leaf) identifiers — the single low-level vocabulary.

In v2 these lived as a hard-coded ``SOPName`` enum inside ``supervisor``.
v4 is data-driven: the *set* of leaves comes from the ``LEAVES`` registry
in :mod:`agent_v4.leaves`, and the graph + classifier prompt are generated
from it. We still keep plain string constants here so the small amount of
unavoidable **domain routing policy** (e.g. "an empty cart routes 'buy X'
to product_rec, not checkout") can name specific leaves without importing
the heavy leaves module — and so low-level models (state, step_result)
have no dependency on the leaf definitions.

To add a new leaf you add a ``LeafSpec`` in :mod:`agent_v4.leaves`; you only
touch this file if a cross-leaf routing rule needs to name it.
"""

from __future__ import annotations

# Canonical leaf ids. Keep in sync with the LEAVES registry (leaves.py
# asserts that the registry names match these at import time).
CHECKOUT = "checkout"
PRODUCT_REC = "product_rec"
ORDER_STATUS = "order_status"

# Where the supervisor defaults when it knows work remains but the
# classifier failed to name a valid leaf.
DEFAULT_SOP = PRODUCT_REC

__all__ = ["CHECKOUT", "PRODUCT_REC", "ORDER_STATUS", "DEFAULT_SOP"]
