"""Tests for the Feynman-derived AI synthesis, model discovery, LLM query
expansion, and the recall-dilution guard."""

from __future__ import annotations

import json

import evidence_engine
import paper_finder
import topic_primer
from evidence_engine import (
    _expansion_trace_note,
    _sanitize_gemini_error,
    generate_ai_evidence_synthesis,
    resolve_gemini_model,
)
from paper_finder import (
    SEARCH_PURPOSE_DEEP,
    SEARCH_PURPOSE_KNOWLEDGE,
    SEARCH_PURPOSE_RARE,
    SEARCH_PURPOSE_RESEARCH,
    SearchContext,
    build_search_layers,
    paper_sort_key,
)


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - mimics requests.Response
        return None

    def json(self) -> dict:
        return self._payload


def _gemini_text_payload(obj: dict) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}


# --------------------------------------------------------------------------- #
# AI evidence synthesis
# --------------------------------------------------------------------------- #
def test_synthesis_without_key_is_not_configured() -> None:
    out = generate_ai_evidence_synthesis({"sources": [{"source_id": "S1"}]}, "")
    assert out["status"] == "not_configured"
    assert out["themes"] == []


def test_synthesis_without_sources_is_blocked() -> None:
    out = generate_ai_evidence_synthesis({"sources": []}, "FAKE")
    assert out["status"] == "blocked"


def test_synthesis_strips_fabricated_source_ids(monkeypatch) -> None:
    payload = _gemini_text_payload(
        {
            "executive_summary": "Summary.",
            "themes": [
                {"theme": "Real", "summary": "ok", "source_ids": ["S1", "S99"],
                 "strength_of_evidence": "high", "is_inference": False},
                {"theme": "Ghost", "summary": "x", "source_ids": ["S99"],
                 "strength_of_evidence": "high"},
            ],
            "agreements": [{"statement": "agree", "source_ids": ["S2", "S99"]},
                           {"statement": "ghost", "source_ids": ["S99"]}],
            "conflicts": [],
            "uncertainties": ["u"],
        }
    )
    monkeypatch.setattr(evidence_engine, "resolve_gemini_model", lambda key: "gemini-x")
    monkeypatch.setattr(evidence_engine.requests, "post", lambda *a, **k: _FakeResp(payload))

    # Focused question so agreement/conflict are retained and we can assert on them.
    review = {"question": {"question_type": "Intervention or treatment"},
              "sources": [{"source_id": "S1", "title": "A"},
                          {"source_id": "S2", "title": "B"}]}
    out = generate_ai_evidence_synthesis(review, "FAKE")

    assert out["status"] == "generated"
    assert len(out["themes"]) == 1                       # fabricated theme dropped
    assert out["themes"][0]["source_ids"] == ["S1"]      # S99 stripped
    assert len(out["agreements"]) == 1
    assert out["agreements"][0]["source_ids"] == ["S2"]


def _synthesis_with_agreements(monkeypatch, question: dict):
    payload = _gemini_text_payload(
        {
            "executive_summary": "Summary.",
            "themes": [{"theme": "T", "summary": "s", "source_ids": ["S1"],
                        "strength_of_evidence": "moderate"}],
            "agreements": [{"statement": "studies concur", "source_ids": ["S1"]}],
            "conflicts": [{"statement": "studies diverge", "source_ids": ["S1"]}],
            "uncertainties": ["u"],
        }
    )
    monkeypatch.setattr(evidence_engine, "resolve_gemini_model", lambda key: "gemini-x")
    monkeypatch.setattr(evidence_engine.requests, "post", lambda *a, **k: _FakeResp(payload))
    review = {"question": question, "sources": [{"source_id": "S1", "title": "A"}]}
    return generate_ai_evidence_synthesis(review, "FAKE")


def test_broad_query_drops_agreements_and_conflicts(monkeypatch) -> None:
    out = _synthesis_with_agreements(monkeypatch, {"question_type": "General evidence map"})
    assert out["query_focus"] == "broad"
    assert out["agreements"] == []      # dropped even though the model returned them
    assert out["conflicts"] == []
    assert out["themes"]                # landscape themes still present


def test_focused_query_keeps_agreements_and_conflicts(monkeypatch) -> None:
    out = _synthesis_with_agreements(
        monkeypatch,
        {"question_type": "Intervention or treatment", "intervention": "corticosteroids"},
    )
    assert out["query_focus"] == "focused"
    assert out["agreements"] and out["agreements"][0]["source_ids"] == ["S1"]
    assert out["conflicts"]


def test_synthesis_sends_abstracts_to_the_model(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, json=None, **kwargs):  # noqa: A002 - matches requests API
        captured["body"] = json
        return _FakeResp(_gemini_text_payload({"executive_summary": "", "themes": []}))

    monkeypatch.setattr(evidence_engine, "resolve_gemini_model", lambda key: "gemini-x")
    monkeypatch.setattr(evidence_engine.requests, "post", fake_post)

    review = {
        "question": {},
        "sources": [{"source_id": "S1", "title": "A",
                     "abstract": "UNIQUEABSTRACTTOKEN findings reported here."}],
    }
    generate_ai_evidence_synthesis(review, "FAKE")
    sent = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "UNIQUEABSTRACTTOKEN" in sent


# --------------------------------------------------------------------------- #
# Model discovery + error sanitizing
# --------------------------------------------------------------------------- #
def test_error_sanitizer_redacts_api_key() -> None:
    msg = _sanitize_gemini_error(Exception("404 for url ...:generateContent?key=AIzaSECRET999"))
    assert "AIzaSECRET999" not in msg
    assert "key=REDACTED" in msg


def test_resolve_gemini_model_prefers_flash(monkeypatch) -> None:
    payload = {
        "models": [
            {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
        ]
    }
    evidence_engine._RESOLVED_GEMINI_MODEL = None
    monkeypatch.setattr(evidence_engine.requests, "get", lambda *a, **k: _FakeResp(payload))
    assert resolve_gemini_model("FAKE") == "gemini-2.5-flash"


def test_resolve_gemini_model_falls_back_on_failure(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("network down")

    evidence_engine._RESOLVED_GEMINI_MODEL = None
    monkeypatch.setattr(evidence_engine.requests, "get", boom)
    assert resolve_gemini_model("FAKE") == evidence_engine.GEMINI_MODEL


def test_resolve_gemini_model_honours_preferred() -> None:
    assert resolve_gemini_model("FAKE", preferred="custom-model") == "custom-model"


# --------------------------------------------------------------------------- #
# LLM query expansion -> search layers (mode gating)
# --------------------------------------------------------------------------- #
def _stub_layers_env(monkeypatch) -> None:
    monkeypatch.setattr(paper_finder, "build_topic_query", lambda topic, email="", api_key="": topic)
    monkeypatch.setattr(
        paper_finder,
        "topic_profile",
        lambda topic: {
            "query_expansion_terms": ["cvst", "dural sinus thrombosis", "sinus thrombosis"],
            "parent_topics": ["venous thromboembolism"],
            "mechanism_terms": ["hypercoagulability"],
        },
    )


def test_recall_layer_only_in_deep_and_rare(monkeypatch) -> None:
    _stub_layers_env(monkeypatch)

    def names(purpose, depth):
        return [l.name for l in build_search_layers(SearchContext(topic="zz topic", search_purpose=purpose), depth)]

    assert "LLM-expanded recall" in names(SEARCH_PURPOSE_DEEP, 200)
    assert "LLM-expanded recall" in names(SEARCH_PURPOSE_RARE, 160)
    assert "LLM-expanded recall" not in names(SEARCH_PURPOSE_RESEARCH, 130)
    assert "LLM-expanded recall" not in names(SEARCH_PURPOSE_KNOWLEDGE, 80)


def test_recall_layer_ors_expansion_terms(monkeypatch) -> None:
    _stub_layers_env(monkeypatch)
    deep = build_search_layers(SearchContext(topic="zz topic", search_purpose=SEARCH_PURPOSE_DEEP), 200)
    recall = next(l for l in deep if l.name == "LLM-expanded recall")
    assert "cvst" in recall.query.lower()
    assert " OR " in recall.query


# --------------------------------------------------------------------------- #
# Workflow trace visibility
# --------------------------------------------------------------------------- #
def test_expansion_trace_note_reports_terms_when_active() -> None:
    layers = [{"name": "LLM-expanded recall",
               "query": '("a"[Title/Abstract] OR "b"[Title/Abstract] OR "c"[Title/Abstract])'}]
    note = _expansion_trace_note({"topic_primer_status": "generated", "layers": layers})
    assert "3 recall terms" in note
    assert "LLM primer" in note


def test_expansion_trace_note_reports_off_states() -> None:
    assert "off (primer: error)" in _expansion_trace_note({"topic_primer_status": "error", "layers": []})
    assert "off (primer: unavailable)" in _expansion_trace_note(
        {"topic_primer_status": "unavailable", "layers": []}
    )


# --------------------------------------------------------------------------- #
# Recall-dilution guard (sort tiebreak)
# --------------------------------------------------------------------------- #
def test_recall_only_paper_sorts_after_equal_precise_paper() -> None:
    base = {
        "search_mode": SEARCH_PURPOSE_DEEP,
        "reading_section": "Extended evidence base",
        "expected_paper_order": 999,
        "tier": "Tier 2: Useful supporting",
        "topic_match_level": "direct",
        "evidence_family_rank": 1,
        "purpose_fit_score": 5,
        "total_score": 70,
        "year": 2022,
    }
    precise = dict(base, expansion_recall_only=False)
    recall_only = dict(base, expansion_recall_only=True)
    # identical on every meaningful axis; only the recall flag differs
    assert paper_sort_key(precise) < paper_sort_key(recall_only)


# --------------------------------------------------------------------------- #
# Eponym-aware primer prompt
# --------------------------------------------------------------------------- #
def test_primer_prompt_requests_eponyms_and_variants() -> None:
    prompt = topic_primer.PROMPT_INSTRUCTIONS.lower()
    assert "eponym" in prompt
    assert "spelling" in prompt or "variant" in prompt

