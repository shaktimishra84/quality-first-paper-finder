from __future__ import annotations

from evidence_engine import build_evidence_review, evidence_type_for_paper


def test_evidence_type_prefers_existing_study_design() -> None:
    paper = {
        "study_design": "Systematic review / meta-analysis",
        "title": "A narrative sounding title",
    }

    assert evidence_type_for_paper(paper) == "Systematic review / meta-analysis"


def test_evidence_type_infers_guideline_from_title() -> None:
    paper = {
        "study_design": "",
        "title": "Scientific statement for cerebral venous thrombosis management",
    }

    assert evidence_type_for_paper(paper) == "Guideline / consensus / society statement"


def test_build_evidence_review_groups_high_value_sources() -> None:
    result = {
        "search_date": "2026-05-11",
        "topic_used": "cerebral venous thrombosis",
        "retrieved_count": 3,
        "layers": [],
        "papers": [
            {
                "title": "Guideline statement",
                "journal": "Stroke",
                "year": 2024,
                "study_design": "Guideline / consensus / society statement",
                "reading_section": "Core reading pack",
                "tier": "Tier 1: Must-read",
                "topic_match_gate": "Direct topic match",
                "pmid": "111",
                "doi": "10.1/test",
                "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
                "citation_count": None,
                "quartile": "quartile not verified",
                "why_included": "guideline anchor",
            },
            {
                "title": "Randomized trial",
                "journal": "NEJM",
                "year": 2020,
                "study_design": "Randomized controlled trial",
                "reading_section": "Core reading pack",
                "tier": "Tier 2: Useful supporting",
                "topic_match_gate": "Direct topic match",
                "pmid": "222",
                "doi": "",
                "url": "https://pubmed.ncbi.nlm.nih.gov/222/",
                "citation_count": 120,
                "quartile": "Q1",
            },
            {
                "title": "Pulmonary embolism background",
                "study_design": "Narrative review",
                "reading_section": "Low-priority / indirect papers",
                "tier": "Noise / manual review",
                "topic_match_gate": "Noise / manual review",
                "pmid": "333",
            },
        ],
        "api_discovery": {"pmids": ["111"]},
        "missing_expected": [],
        "gap_map": [],
        "errors": [],
        "question_context": {
            "topic": "cerebral venous thrombosis",
            "question_type": "General evidence map",
        },
    }

    review = build_evidence_review(result)

    assert review["verification"]["records_reviewed"] == 2
    assert len(review["major_guidelines"]) == 1
    assert len(review["major_randomized_trials"]) == 1
    assert "Citation Verification" in review["markdown"]
    assert "PMID 111" in review["markdown"]
