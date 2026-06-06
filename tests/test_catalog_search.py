"""Catalog search must do whole-word matching, not substring.

The bug being prevented: queries like 'yes san francisco ca 94123'
returning baseball caps because "ca" was a substring of "cap".
"""

from __future__ import annotations

from agent_v2.checkout import catalog


def test_empty_query_returns_full_catalog():
    out = catalog.search("", limit=10)
    assert len(out) >= 5  # at least all our seeded products


def test_substring_ca_does_not_match_cap():
    """The regression — 'ca' (from California) must NOT match 'cap'."""
    out = catalog.search("yes san francisco ca 94123")
    assert out == []


def test_whole_word_query_works():
    out = catalog.search("hoodie")
    ids = [p.id for p in out]
    assert "P-2" in ids  # Black Hoodie


def test_query_matches_tag():
    out = catalog.search("shoes")
    ids = [p.id for p in out]
    assert "P-3" in ids  # tag includes 'shoes'


def test_query_matches_color_tag():
    out = catalog.search("green")
    ids = [p.id for p in out]
    assert "P-4" in ids  # green cap
    assert "P-5" not in ids  # red cap, no green tag


def test_prefix_match_three_or_more_chars():
    """'shoe' should match 'shoes' via proper-prefix logic."""
    out = catalog.search("shoe")
    ids = [p.id for p in out]
    assert "P-3" in ids


def test_two_letter_tokens_do_not_substring_match():
    out = catalog.search("ca")
    # Whole-word "ca" matches NO product. Proper-prefix requires len>=3.
    assert out == []


def test_multi_token_query_scores_higher():
    """A query that matches multiple tokens should rank above one-token matches."""
    out = catalog.search("baseball cap green")
    assert out  # at least one match
    assert out[0].id == "P-4"  # green cap should rank first


def test_search_by_product_id_with_hyphen():
    out = catalog.search("P-3")
    assert out and out[0].id == "P-3"


def test_search_by_product_id_lowercase_with_hyphen():
    out = catalog.search("p-3")
    assert out and out[0].id == "P-3"


def test_search_by_product_id_without_hyphen():
    """'p3' / 'P3' must find P-3 — users won't always type the hyphen."""
    out = catalog.search("p3")
    assert out and out[0].id == "P-3"
    out2 = catalog.search("P3")
    assert out2 and out2[0].id == "P-3"


def test_search_id_in_sentence_finds_it_first():
    """User types 'add the p3 to cart' → P-3 must be the top hit."""
    out = catalog.search("add the p3 to cart")
    assert out and out[0].id == "P-3"


def test_search_by_id_outranks_tag_match():
    """If the query has BOTH a product id and a tag, the id-match
    product should rank above tag-only matches."""
    out = catalog.search("p3 cap")
    # P-3 is sneakers (no 'cap' tag) but matches by id.
    # P-4, P-5 are caps (cap matches).
    # P-3 should still come first because id matches are weighted higher.
    assert out and out[0].id == "P-3"
