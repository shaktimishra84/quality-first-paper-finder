from __future__ import annotations

from evidence_engine import build_evidence_review, evidence_type_for_paper
from paper_finder import (
    SEARCH_PURPOSE_DEEP,
    SEARCH_PURPOSE_KNOWLEDGE,
    SEARCH_PURPOSE_RARE,
    SEARCH_PURPOSE_RESEARCH,
    SearchContext,
    score_and_classify_paper,
    search_purpose_config,
)


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
                "reading_section": "Guidelines and consensus",
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
                "reading_section": "Randomized controlled trials",
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
                "reading_section": "Tier 4 / weak but related papers",
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
            "search_purpose": SEARCH_PURPOSE_RESEARCH,
        },
    }

    review = build_evidence_review(result, generate_ai_gaps=True)

    assert review["verification"]["records_reviewed"] == 2
    assert len(review["major_guidelines"]) == 1
    assert len(review["major_randomized_trials"]) == 1
    assert review["ai_gap_synthesis"]["status"] == "not_configured"
    assert "Citation Verification" in review["markdown"]
    assert "PMID 111" in review["markdown"]


def test_search_purpose_presets_are_researcher_facing() -> None:
    knowledge = search_purpose_config(SEARCH_PURPOSE_KNOWLEDGE)
    research = search_purpose_config(SEARCH_PURPOSE_RESEARCH)
    deep = search_purpose_config(SEARCH_PURPOSE_DEEP)
    rare = search_purpose_config(SEARCH_PURPOSE_RARE)

    assert knowledge["candidate_depth"] < research["candidate_depth"] < deep["candidate_depth"]
    assert rare["candidate_depth"] >= research["candidate_depth"]
    assert knowledge["ai_gap_analysis"] is False
    assert research["ai_gap_analysis"] is True
    assert "description" in deep


def test_same_review_ranks_higher_for_knowledge_than_research() -> None:
    paper = {
        "title": "Cerebral venous thrombosis: a narrative review",
        "abstract": "Cerebral venous thrombosis review for broad clinical background.",
        "publication_types": ["Review"],
        "journal": "Neurology Review",
        "pmid": "111",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "citation_count": 150,
        "citation_source": "OpenAlex",
        "year": 2020,
    }

    knowledge = score_and_classify_paper(
        paper,
        SearchContext(topic="cerebral venous thrombosis", search_purpose=SEARCH_PURPOSE_KNOWLEDGE),
        {},
    )
    research = score_and_classify_paper(
        paper,
        SearchContext(topic="cerebral venous thrombosis", search_purpose=SEARCH_PURPOSE_RESEARCH),
        {},
    )

    assert knowledge["tier"] in {"Tier 1: Must-read", "Tier 2: Useful supporting"}
    assert research["tier"] == "Tier 3: Background"
    assert knowledge["reading_section"] == "Best review articles"
    assert research["reading_section"] == "Background reviews"


def test_case_report_ranks_highest_for_rare_mode() -> None:
    paper = {
        "title": "Rare cerebral venous thrombosis complication: a case report",
        "abstract": "A rare cerebral venous thrombosis complication is described in this case report.",
        "publication_types": ["Case Reports"],
        "journal": "Case Reports in Neurology",
        "pmid": "222",
        "url": "https://pubmed.ncbi.nlm.nih.gov/222/",
        "citation_count": 2,
        "citation_source": "OpenAlex",
        "year": 2022,
    }

    rare = score_and_classify_paper(
        paper,
        SearchContext(topic="cerebral venous thrombosis complication", search_purpose=SEARCH_PURPOSE_RARE),
        {},
    )
    deep = score_and_classify_paper(
        paper,
        SearchContext(topic="cerebral venous thrombosis complication", search_purpose=SEARCH_PURPOSE_DEEP),
        {},
    )

    assert rare["tier"] == "Tier 1: Must-read"
    assert rare["reading_section"] == "Closest matching case reports"
    assert deep["tier"] in {"Tier 2: Useful supporting", "Tier 3: Background", "Tier 4: Low priority"}
    assert rare["search_mode"] == SEARCH_PURPOSE_RARE
    assert rare["relation_type"] == "Directly related"
