"""Tests for intent-aware ranking (LLM intent + embeddings + LLM rerank)."""

from __future__ import annotations

import intent_ranker
import topic_primer
from intent_ranker import rank_by_intent


def _patch(monkeypatch, query_vec, doc_vecs, fits) -> None:
    monkeypatch.setattr(intent_ranker, "_resolve_embed_model", lambda k: "m")

    def fake_embed(texts, key, model, task_type):
        return [query_vec] if task_type == "RETRIEVAL_QUERY" else list(doc_vecs)

    monkeypatch.setattr(intent_ranker, "_embed", fake_embed)
    monkeypatch.setattr(intent_ranker, "_llm_rerank", lambda intent, subset, key: fits)


def test_skipped_without_key() -> None:
    papers = [{"title": "x", "total_score": 1}]
    assert rank_by_intent("intent", "query", papers, "") == "skipped"
    assert "intent_fit" not in papers[0]


def test_intent_fit_from_embeddings_then_rerank(monkeypatch) -> None:
    a = {"title": "relevant", "abstract": "hantavirus management", "total_score": 90}
    b = {"title": "ecmo", "abstract": "cardiogenic shock ecmo", "total_score": 80}
    # embeddings: a close (1,0)->1.0, b far (0,1)->0.0; LLM rerank overrides both
    _patch(monkeypatch, [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], {0: 0.9, 1: 0.1})

    status = rank_by_intent("management of hantavirus", "query", [a, b], "FAKE")

    assert status == "scored"
    assert a["intent_similarity"] == 1.0
    assert a["intent_fit"] == 0.9 and a["intent_reranked"]    # LLM rerank wins
    assert b["intent_fit"] == 0.1 and b["intent_reranked"]


def test_falls_back_to_embeddings_when_rerank_empty(monkeypatch) -> None:
    a = {"title": "x", "abstract": "y", "total_score": 5}
    _patch(monkeypatch, [1.0, 0.0], [[1.0, 0.0]], {})   # rerank returns nothing
    rank_by_intent("intent", "query", [a], "FAKE")
    assert a["intent_fit"] == 1.0 and not a.get("intent_reranked")


def test_fail_soft_on_embed_failure(monkeypatch) -> None:
    monkeypatch.setattr(intent_ranker, "_resolve_embed_model", lambda k: "m")
    monkeypatch.setattr(intent_ranker, "_embed", lambda *a, **k: None)
    a = {"title": "x", "abstract": "y", "total_score": 5}
    assert rank_by_intent("intent", "query", [a], "FAKE") == "error"
    assert "intent_fit" not in a


def test_llm_rerank_parses_ratings(monkeypatch) -> None:
    payload = {"candidates": [{"content": {"parts": [
        {"text": '{"ratings":[{"id":0,"fit":80},{"id":1,"fit":10},{"id":9,"fit":50}]}'}
    ]}}]}

    class _Resp:
        def raise_for_status(self): return None
        def json(self): return payload

    monkeypatch.setattr(intent_ranker, "resolve_gemini_model", lambda k: "m")
    monkeypatch.setattr(intent_ranker.requests, "post", lambda *a, **k: _Resp())

    fits = intent_ranker._llm_rerank("intent", [{"title": "a"}, {"title": "b"}], "FAKE")
    assert fits == {0: 0.8, 1: 0.1}    # id 9 is out of range -> dropped


def test_primer_exposes_clinical_intent() -> None:
    # The primer schema + prompt must request a clinical_intent for ranking.
    assert "clinical_intent" in topic_primer.GEMINI_RESPONSE_SCHEMA["properties"]
    assert "clinical_intent" in topic_primer.PROMPT_INSTRUCTIONS
