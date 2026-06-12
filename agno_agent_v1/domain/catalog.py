"""Tiny in-memory product catalog (mock)."""

from __future__ import annotations

from decimal import Decimal

from agno_agent_v1.domain.cart import CartItem


class Product:
    def __init__(self, id: str, name: str, price: Decimal, tags: list[str]) -> None:
        self.id = id
        self.name = name
        self.price = price
        self.tags = tags

    def to_cart_item(self, quantity: int = 1) -> CartItem:
        return CartItem(
            product_id=self.id,
            name=self.name,
            unit_price=self.price,
            quantity=quantity,
            tags=list(self.tags),
        )


CATALOG: dict[str, Product] = {
    "P-1": Product("P-1", "Classic White T-shirt", Decimal("19.99"), ["apparel", "shirt", "white"]),
    "P-2": Product("P-2", "Black Hoodie", Decimal("49.99"), ["apparel", "hoodie", "black"]),
    "P-3": Product("P-3", "Running Sneakers", Decimal("89.00"), ["shoes", "running"]),
    "P-4": Product("P-4", "Baseball Cap (Green)", Decimal("14.50"), ["apparel", "hat", "green"]),
    "P-5": Product("P-5", "Baseball Cap (Red)", Decimal("14.50"), ["apparel", "hat", "red"]),
}


def search(query: str, limit: int = 5) -> list[Product]:
    """Whole-word match against product-name tokens + tags.

    Tokenize both sides and require exact token equality or a proper-prefix match
    of length >= 3. (Substring scoring once let "ca" match "cap", which cascaded
    into spurious results for serviceability questions — so it is avoided here.)
    """
    q = (query or "").lower().strip()
    if not q:
        return list(CATALOG.values())[:limit]
    query_tokens = [tok.strip("()[],.!?") for tok in q.split() if tok.strip("()[],.!?")]
    if not query_tokens:
        return list(CATALOG.values())[:limit]
    norm_query_tokens = set(query_tokens) | {qt.replace("-", "") for qt in query_tokens}

    scored: list[tuple[int, Product]] = []
    for p in CATALOG.values():
        haystack_tokens = {tok.strip("()[],.!?").lower() for tok in p.name.split()} | {
            t.lower() for t in p.tags
        }
        pid = p.id.lower()
        haystack_tokens.add(pid)
        haystack_tokens.add(pid.replace("-", ""))
        haystack_tokens.discard("")

        score = 0
        for qt in norm_query_tokens:
            if qt in haystack_tokens:
                score += 10 if qt in (pid, pid.replace("-", "")) else 2
                continue
            if len(qt) >= 3 and any(
                ht.startswith(qt) or qt.startswith(ht) for ht in haystack_tokens if len(ht) >= 3
            ):
                score += 1
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:limit]]


def get(product_id: str) -> Product | None:
    return CATALOG.get(product_id.upper())
