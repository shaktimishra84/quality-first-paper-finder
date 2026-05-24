"""Tests for the bundled SCImago quartile table and journal-name matching."""

from __future__ import annotations

from paper_finder import (
    load_bundled_quartiles,
    lookup_quartile,
    quartile_query_keys,
)


def test_bundled_table_loads() -> None:
    table = load_bundled_quartiles()
    assert len(table) > 5000  # the SCImago medicine set is several thousand journals
    assert "resuscitation" in table


def test_query_keys_bridge_the_prefix_and_suffix() -> None:
    keys = quartile_query_keys("Lancet")
    assert "lancet" in keys and "the lancet" in keys
    keys2 = quartile_query_keys("Lancet (London, England)")
    assert "lancet" in keys2  # location suffix stripped


def test_lookup_matches_despite_the_prefix() -> None:
    table = load_bundled_quartiles()
    # SCImago stores "The Lancet"; PubMed often gives "Lancet" or with a suffix.
    assert lookup_quartile("Lancet", table)["quartile"] == "Q1"
    assert lookup_quartile("Lancet (London, England)", table)["quartile"] == "Q1"
    assert lookup_quartile("Resuscitation", table)["quartile"] == "Q1"


def test_unknown_journal_is_not_verified() -> None:
    table = load_bundled_quartiles()
    assert lookup_quartile("Journal of Nonexistent Nonsense 9999", table)["quartile"] == (
        "quartile not verified"
    )


def test_user_override_wins_over_bundled() -> None:
    bundled = load_bundled_quartiles()
    merged = {**bundled, "resuscitation": {"quartile": "Q3", "source": "manual override"}}
    assert lookup_quartile("Resuscitation", merged)["quartile"] == "Q3"
