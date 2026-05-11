"""Unit tests for topic_primer.

Exercises the deterministic helpers (normalization, coercion, title overlap)
plus the cached prime_topic flow with mocked Gemini and NCBI calls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Make the project root importable when running `pytest` from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from topic_primer import (  # noqa: E402  (import after sys.path tweak)
    ExpectedPaper,
    PenaltyRule,
    TopicPrimer,
    _build_primer_from_payload,
    _coerce_penalty_rules,
    _coerce_str_list,
    _normalize_topic_key,
    _title_overlap,
    clear_primer_cache,
    prime_topic,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    clear_primer_cache()
    yield
    clear_primer_cache()


@pytest.mark.unit
def test_normalize_topic_key_lowercases_and_collapses_whitespace() -> None:
    assert _normalize_topic_key("  ARDS Treatment  ") == "ards treatment"
    assert _normalize_topic_key("Sepsis\t\nCare") == "sepsis care"
    assert _normalize_topic_key("") == ""


@pytest.mark.unit
def test_title_overlap_matches_substantial_token_overlap() -> None:
    claimed = "Driving Pressure and Survival in ARDS"
    actual = (
        "Driving pressure and survival in the acute respiratory distress syndrome"
    )

    overlap = _title_overlap(claimed, actual)

    assert overlap >= 0.5


@pytest.mark.unit
def test_title_overlap_rejects_unrelated_titles() -> None:
    overlap = _title_overlap("VILI mechanical power", "Sepsis bundle compliance")

    assert overlap < 0.5


@pytest.mark.unit
def test_title_overlap_handles_empty_inputs() -> None:
    assert _title_overlap("", "anything") == 0.0
    assert _title_overlap("anything", "") == 0.0


@pytest.mark.unit
def test_coerce_str_list_strips_dedups_and_caps() -> None:
    raw = ["A", "b ", " a", "c", "c", "d", "e"]

    out = _coerce_str_list(raw, max_items=3)

    assert out == ("a", "b", "c")


@pytest.mark.unit
def test_coerce_str_list_handles_non_list_input() -> None:
    assert _coerce_str_list(None, max_items=5) == ()
    assert _coerce_str_list("not a list", max_items=5) == ()


@pytest.mark.unit
def test_coerce_penalty_rules_clamps_score_and_skips_invalid() -> None:
    raw = [
        {"name": "valid", "match_terms": ["x"], "score": -8, "reason": "test"},
        {"name": "no terms", "match_terms": [], "score": -5},
        {"name": "", "match_terms": ["y"], "score": -5},
        {"name": "missing score", "match_terms": ["z"]},
        {"name": "out of bounds", "match_terms": ["w"], "score": -100},
    ]

    rules = _coerce_penalty_rules(raw)

    # Only "valid" and "out of bounds" pass; the latter is clamped to -25.
    assert len(rules) == 2
    assert rules[0].score == -8
    assert rules[0].match_terms == ("x",)
    assert rules[1].score == -25


@pytest.mark.unit
def test_topic_primer_to_profile_dict_has_expected_shape() -> None:
    primer = TopicPrimer(
        topic="VILI",
        expected_papers=(
            ExpectedPaper(
                pmid="12345",
                title="Test paper",
                reason="reason",
                category="Landmark RCT",
            ),
        ),
        must_include_concepts=("driving pressure",),
        penalize_rules=(PenaltyRule("animal", ("rat",), -10, "animal-only"),),
        query_expansion_terms=("ventilator induced lung injury",),
        population_priority="adult ICU",
        expected_categories=("Major guideline / consensus",),
        status="generated",
    )

    profile = primer.to_profile_dict()

    assert profile["_primed"] is True
    assert profile["_primer_status"] == "generated"
    assert profile["expected_papers"][0]["pmid"] == "12345"
    assert "vili" in profile["triggers"]
    assert profile["penalize"][0]["score"] == -10
    assert profile["query_expansion_terms"] == ["ventilator induced lung injury"]


@pytest.mark.unit
def test_prime_topic_returns_none_without_key() -> None:
    assert prime_topic("any topic", gemini_key="") is None
    assert prime_topic("", gemini_key="key") is None


@pytest.mark.unit
def test_prime_topic_caches_repeat_calls() -> None:
    fake_payload = {
        "expected_papers": [],
        "must_include_concepts": ["test"],
        "penalize_rules": [],
        "query_expansion_terms": ["alt"],
    }
    with patch("topic_primer._call_gemini", return_value=fake_payload) as mock_call:
        first = prime_topic("Sepsis bundles", gemini_key="key")
        second = prime_topic("sepsis bundles", gemini_key="key")  # different case

    assert first is not None
    assert second is not None
    assert first.status == "generated"
    assert second.status == "cached"
    assert mock_call.call_count == 1


@pytest.mark.unit
def test_prime_topic_returns_none_on_network_failure() -> None:
    with patch("topic_primer._call_gemini", side_effect=requests.HTTPError("rate limit")):
        result = prime_topic("Anything", gemini_key="key")

    assert result is None


@pytest.mark.unit
def test_prime_topic_returns_none_on_invalid_json() -> None:
    with patch("topic_primer._call_gemini", side_effect=ValueError("bad json")):
        result = prime_topic("Anything", gemini_key="key")

    assert result is None


@pytest.mark.unit
def test_build_primer_drops_pmids_that_fail_verification() -> None:
    payload = {
        "expected_papers": [
            {"pmid": "111", "title": "Real title", "reason": "r", "category": "Landmark RCT"},
            {"pmid": "222", "title": "Hallucinated", "reason": "r", "category": "Landmark RCT"},
        ],
        "must_include_concepts": ["foo"],
        "penalize_rules": [],
        "query_expansion_terms": ["bar"],
    }
    # Only PMID 111 verifies — 222 is "missing" from NCBI response
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "result": {
            "111": {"title": "Real title from NCBI"},
        }
    }
    fake_response.raise_for_status = MagicMock()

    with patch("topic_primer.requests.get", return_value=fake_response):
        primer = _build_primer_from_payload(
            topic="Test",
            payload=payload,
            email="",
            api_key="",
            status="generated",
        )

    assert len(primer.expected_papers) == 1
    assert primer.expected_papers[0].pmid == "111"
    assert "1 suggested PMIDs failed NCBI verification" in primer.notes[0]


@pytest.mark.unit
def test_build_primer_drops_pmid_when_title_mismatches() -> None:
    payload = {
        "expected_papers": [
            {"pmid": "999", "title": "Driving pressure ARDS", "category": "Landmark RCT"},
        ],
        "must_include_concepts": [],
        "penalize_rules": [],
        "query_expansion_terms": [],
    }
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "result": {"999": {"title": "Sepsis bundle implementation"}},
    }
    fake_response.raise_for_status = MagicMock()

    with patch("topic_primer.requests.get", return_value=fake_response):
        primer = _build_primer_from_payload(
            topic="Test", payload=payload, email="", api_key="", status="generated"
        )

    assert primer.expected_papers == ()


@pytest.mark.unit
def test_build_primer_normalizes_invalid_category_to_default() -> None:
    payload = {
        "expected_papers": [
            {
                "pmid": "1",
                "title": "Driving pressure and survival",
                "category": "Made-up category",
            },
        ],
        "must_include_concepts": [],
        "penalize_rules": [],
        "query_expansion_terms": [],
    }
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "result": {"1": {"title": "Driving pressure and survival in ARDS"}}
    }
    fake_response.raise_for_status = MagicMock()

    with patch("topic_primer.requests.get", return_value=fake_response):
        primer = _build_primer_from_payload(
            topic="t", payload=payload, email="", api_key="", status="generated"
        )

    assert primer.expected_papers[0].category == "Recent high-quality update"
