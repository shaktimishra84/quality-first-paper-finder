"""Tests for the optional Gemini-embedding semantic relevance guard."""

from __future__ import annotations

import semantic_relevance
from semantic_relevance import cosine, score_semantic_relevance


def test_cosine_basics() -> None:
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([], [1]) == 0.0


def test_skipped_without_key() -> None:
    papers = [{"title": "x", "total_score": 1}]
    assert score_semantic_relevance("topic", papers, "") == "skipped"
    assert "semantic_score" not in papers[0]


def _patch_embed(monkeypatch, query_vec, doc_vecs) -> None:
    def fake_embed(texts, key, model, task_type):
        return [query_vec] if task_type == "RETRIEVAL_QUERY" else list(doc_vecs)

    monkeypatch.setattr(semantic_relevance, "_resolve_embed_model", lambda k: "m")
    monkeypatch.setattr(semantic_relevance, "_embed", fake_embed)


def test_flags_high_tier_low_similarity_outlier(monkeypatch) -> None:
    # Order embedded = sorted by total_score desc: A (relevant), B (collision), C (low tier)
    _patch_embed(monkeypatch, [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    a = {"title": "hantavirus", "abstract": "hantavirus cardiopulmonary syndrome",
         "tier": "Tier 1: Must-read", "total_score": 90}
    b = {"title": "BPH HPS laser", "abstract": "benign prostatic hyperplasia greenlight laser",
         "tier": "Tier 1: Must-read", "total_score": 80}
    c = {"title": "unrelated", "abstract": "something", "tier": "Tier 3: Background", "total_score": 10}
    papers = [a, b, c]

    status = score_semantic_relevance("hantavirus cardiopulmonary syndrome", papers, "FAKE")

    assert status == "scored"
    assert a["semantic_score"] == 1.0 and not a["semantic_outlier"]   # relevant, kept
    assert b["semantic_score"] == 0.0 and b["semantic_outlier"]       # collision, flagged
    assert not c.get("semantic_outlier")                              # low tier -> not flagged


def test_protected_paper_not_flagged(monkeypatch) -> None:
    _patch_embed(monkeypatch, [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    a = {"title": "rel", "abstract": "hantavirus", "tier": "Tier 1: Must-read", "total_score": 90}
    landmark = {"title": "far", "abstract": "off topic", "tier": "Tier 1: Must-read",
                "total_score": 80, "mandatory_review_candidate": True}
    score_semantic_relevance("hantavirus", [a, landmark], "FAKE")
    assert not landmark["semantic_outlier"]   # protected landmarks are exempt


def test_no_abstract_not_flagged(monkeypatch) -> None:
    _patch_embed(monkeypatch, [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    a = {"title": "rel", "abstract": "hantavirus", "tier": "Tier 1: Must-read", "total_score": 90}
    no_abs = {"title": "far", "abstract": "", "tier": "Tier 1: Must-read", "total_score": 80}
    score_semantic_relevance("hantavirus", [a, no_abs], "FAKE")
    assert not no_abs["semantic_outlier"]   # can't judge without an abstract


def test_embed_failure_is_fail_soft(monkeypatch) -> None:
    monkeypatch.setattr(semantic_relevance, "_resolve_embed_model", lambda k: "m")
    monkeypatch.setattr(semantic_relevance, "_embed", lambda *a, **k: None)
    papers = [{"title": "x", "abstract": "y", "tier": "Tier 1: Must-read", "total_score": 5}]
    assert score_semantic_relevance("topic", papers, "FAKE") == "error"
    assert not papers[0].get("semantic_outlier")
