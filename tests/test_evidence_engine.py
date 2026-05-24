from __future__ import annotations

import paper_finder
from evidence_engine import build_evidence_review, evidence_type_for_paper
from paper_finder import (
    SEARCH_PURPOSE_DEEP,
    SEARCH_PURPOSE_KNOWLEDGE,
    SEARCH_PURPOSE_RARE,
    SEARCH_PURPOSE_RESEARCH,
    SearchContext,
    build_search_layers,
    classify_topic_match,
    expand_acronyms,
    score_and_classify_paper,
    search_purpose_config,
    user_intent_terms,
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
    assert knowledge["reading_section"] == "Best narrative reviews"
    assert research["reading_section"] == "Background reviews"


def test_learning_mode_prioritizes_narrative_reviews_over_meta_analysis() -> None:
    narrative_paper = {
        "title": "Cerebral venous thrombosis: a narrative review",
        "abstract": "Cerebral venous thrombosis review for broad clinical learning.",
        "publication_types": ["Review"],
        "journal": "Neurology Review",
        "pmid": "111",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "citation_count": 80,
        "citation_source": "OpenAlex",
        "year": 2021,
    }
    meta_paper = {
        "title": "Cerebral venous thrombosis: a systematic review and meta-analysis",
        "abstract": "Cerebral venous thrombosis evidence synthesis and pooled outcomes.",
        "publication_types": ["Systematic Review", "Meta-Analysis"],
        "journal": "Stroke",
        "pmid": "222",
        "url": "https://pubmed.ncbi.nlm.nih.gov/222/",
        "citation_count": 200,
        "citation_source": "OpenAlex",
        "year": 2021,
    }
    context = SearchContext(
        topic="cerebral venous thrombosis",
        search_purpose=SEARCH_PURPOSE_KNOWLEDGE,
    )

    narrative = score_and_classify_paper(narrative_paper, context, {})
    meta = score_and_classify_paper(meta_paper, context, {})

    assert narrative["purpose_fit_score"] > meta["purpose_fit_score"]
    assert narrative["reading_section"] == "Best narrative reviews"
    assert meta["reading_section"] == "Evidence synthesis"
    assert meta["tier"] == "Tier 3: Background"
    assert meta["total_score"] <= 59


def test_profile_topic_search_layers_preserve_user_intent_modifiers() -> None:
    context = SearchContext(
        topic="cerebral venous thrombosis anticoagulation recurrence",
        search_purpose=SEARCH_PURPOSE_KNOWLEDGE,
    )

    layers = build_search_layers(context, candidate_depth=30)
    intent_layer = next(layer for layer in layers if layer.name == "Intent-focused review")
    focused_layer = next(layer for layer in layers if layer.name == "Focused")

    assert user_intent_terms(context) == ["anticoagulation", "recurrence"]
    assert "anticoagulation" in intent_layer.query
    assert "recurrence" in intent_layer.query
    assert "anticoagulation" in focused_layer.query
    assert "recurrence" in focused_layer.query


def test_requested_intent_modifiers_downrank_generic_profile_matches() -> None:
    context = SearchContext(
        topic="cerebral venous thrombosis anticoagulation recurrence",
        search_purpose=SEARCH_PURPOSE_KNOWLEDGE,
    )
    generic = {
        "title": "Cerebral venous thrombosis: a narrative review",
        "abstract": "Cerebral venous thrombosis diagnosis and clinical presentation are reviewed.",
        "publication_types": ["Review"],
        "journal": "Stroke",
        "pmid": "111",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "citation_count": 300,
        "citation_source": "OpenAlex",
        "year": 2021,
    }
    focused = {
        "title": "Cerebral venous thrombosis anticoagulation and recurrence: a narrative review",
        "abstract": "Cerebral venous thrombosis anticoagulation duration and recurrence risk are reviewed.",
        "publication_types": ["Review"],
        "journal": "Stroke",
        "pmid": "222",
        "url": "https://pubmed.ncbi.nlm.nih.gov/222/",
        "citation_count": 80,
        "citation_source": "OpenAlex",
        "year": 2021,
    }

    generic_scored = score_and_classify_paper(generic, context, {})
    focused_scored = score_and_classify_paper(focused, context, {})

    assert "Does not match requested modifiers" in "; ".join(generic_scored["penalty_notes"])
    assert generic_scored["tier"] == "Tier 3: Background"
    assert focused_scored["intent_match_score"] == 6
    assert focused_scored["total_score"] > generic_scored["total_score"]


def test_learning_mode_demotes_veterinary_systematic_reviews() -> None:
    paper = {
        "title": "Acute Respiratory Distress Syndrome in Veterinary Medicine-The ARDSVet Definitions.",
        "abstract": "Veterinary medicine consensus definitions for acute respiratory distress syndrome in dogs and cats.",
        "publication_types": ["Consensus Statement", "Journal Article", "Systematic Review"],
        "journal": "Journal of veterinary emergency and critical care",
        "pmid": "40838381",
        "url": "https://pubmed.ncbi.nlm.nih.gov/40838381/",
        "citation_count": 5,
        "citation_source": "OpenAlex",
        "year": 2025,
    }

    result = score_and_classify_paper(
        paper,
        SearchContext(
            topic="acute respiratory distress syndrome",
            search_purpose=SEARCH_PURPOSE_KNOWLEDGE,
        ),
        {},
    )

    assert result["study_design"] == "Systematic review / meta-analysis"
    assert result["tier"] == "Tier 4: Low priority"
    assert result["total_score"] <= 39
    assert "veterinary" in result["non_human_signal"]
    assert "non-human" in result["score_cap_reason"]


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


def test_editorial_correspondence_is_kept_in_deep_and_rare_modes() -> None:
    paper = {
        "title": "Correspondence: rare cerebral venous thrombosis complication after infection",
        "abstract": "A rare cerebral venous thrombosis complication is discussed with clinical implications.",
        "publication_types": ["Letter", "Comment"],
        "journal": "Stroke",
        "pmid": "333",
        "url": "https://pubmed.ncbi.nlm.nih.gov/333/",
        "citation_count": 1,
        "citation_source": "OpenAlex",
        "year": 2024,
    }

    deep = score_and_classify_paper(
        paper,
        SearchContext(topic="cerebral venous thrombosis", search_purpose=SEARCH_PURPOSE_DEEP),
        {},
    )
    rare = score_and_classify_paper(
        paper,
        SearchContext(topic="cerebral venous thrombosis", search_purpose=SEARCH_PURPOSE_RARE),
        {},
    )

    assert deep["study_design"] == "Editorial / correspondence / commentary"
    assert deep["reading_section"] == "Editorials/correspondence"
    assert rare["study_design"] == "Editorial / correspondence / commentary"
    assert rare["tier"] in {"Tier 1: Must-read", "Tier 2: Useful supporting"}
    assert rare["publication_type"] == "Letter, Comment"


def test_semantic_topic_gate_does_not_promote_component_overlap_to_full_concept() -> None:
    gate = classify_topic_match(
        "Pulmonary embolism presenting with shock: a case report",
        "Massive pulmonary embolism with hemodynamic collapse and right ventricular failure.",
        SearchContext(topic="cardiogenic shock in pulmonary embolism"),
    )

    assert gate["level"] == "strong_component"
    assert gate["relevance_cap"] == 34
    assert "component" in gate["gate"].lower()


def test_semantic_topic_gate_treats_negated_component_as_parallel_fallback() -> None:
    gate = classify_topic_match(
        "Cardiogenic shock: current concepts and management",
        "A broad review of cardiogenic shock without pulmonary embolism as a focus.",
        SearchContext(topic="cardiogenic shock in pulmonary embolism"),
    )

    assert gate["level"] == "parallel"
    assert gate["relevance_cap"] == 22
    assert "parallel" in gate["gate"].lower()


def test_research_mode_keeps_direct_case_reports_low_priority() -> None:
    paper = {
        "title": "Pulmonary embolism presenting with shock: a case report",
        "abstract": "Massive pulmonary embolism with shock, hemodynamic collapse, and right ventricular failure is described.",
        "publication_types": ["Case Reports"],
        "journal": "Journal of Medical Case Reports",
        "pmid": "444",
        "url": "https://pubmed.ncbi.nlm.nih.gov/444/",
        "citation_count": 15,
        "citation_source": "OpenAlex",
        "year": 2024,
    }

    result = score_and_classify_paper(
        paper,
        SearchContext(
            topic="cardiogenic shock in pulmonary embolism",
            search_purpose=SEARCH_PURPOSE_RESEARCH,
        ),
        {},
    )

    assert result["topic_match_level"] == "strong_component"
    assert result["study_design"] == "Case series / case report"
    assert result["tier"] == "Tier 4: Low priority"


def _hantavirus_primed_profile() -> dict:
    # Mirrors topic_primer.to_profile_dict() output for a primed topic.
    return {
        "key": "primed:x",
        "display_name": "Critical care management of hanta virus cardiopulmonary syndrome",
        "triggers": ["critical care management of hanta virus cardiopulmonary syndrome"],
        "direct_phrases": ["critical care management of hanta virus cardiopulmonary syndrome"],
        "direct_synonyms": ["hantavirus cardiopulmonary syndrome", "hantavirus pulmonary syndrome"],
        "component_concepts": ["hantavirus", "cardiopulmonary syndrome"],
        "parent_topics": ["viral hemorrhagic fever"],
        "parallel_topics": [],
        "mechanism_terms": ["capillary leak"],
        "direct_acronyms": ["HPS", "HCPS"],
        "acronym_context": ["hantavirus", "cardiopulmonary syndrome", "cardiopulmonary"],
        "must_include_concepts": ["hantavirus", "cardiopulmonary"],
        "query_expansion_terms": ["HPS", "HCPS", "hantavirus pulmonary syndrome"],
    }


def test_ambiguous_acronym_does_not_force_direct_match(monkeypatch) -> None:
    # Regression: the primer abbreviation "HPS" (Hantavirus Pulmonary Syndrome)
    # must not make an unrelated "GreenLight HPS laser" BPH paper a direct match.
    monkeypatch.setattr(paper_finder, "topic_profile", lambda topic: _hantavirus_primed_profile())
    context = SearchContext(topic="Critical care management of hanta virus cardiopulmonary syndrome")

    bph = classify_topic_match(
        "Comparative efficacy and safety of 180 W XPS vs. 120 W HPS GreenLight laser "
        "therapy for benign prostatic hyperplasia: a systematic review and meta-analysis.",
        "A systematic review comparing GreenLight laser therapy for benign prostatic hyperplasia.",
        context,
    )
    assert bph["level"] not in {"direct", "direct_synonym", "strong_component", "abstract_only"}
    assert bph["relevance_cap"] <= 14


def test_acronym_with_disease_context_still_matches(monkeypatch) -> None:
    monkeypatch.setattr(paper_finder, "topic_profile", lambda topic: _hantavirus_primed_profile())
    context = SearchContext(topic="Critical care management of hanta virus cardiopulmonary syndrome")

    real = classify_topic_match(
        "Critical care management of hantavirus cardiopulmonary syndrome (HPS): a review",
        "Hantavirus pulmonary syndrome causes capillary leak and cardiopulmonary collapse.",
        context,
    )
    assert real["level"] in {"direct", "direct_synonym", "strong_component", "abstract_only"}


def test_cam_icu_typo_expands_to_specific_delirium_tool() -> None:
    expanded = expand_acronyms("complianse of cam icu")

    assert expanded == "compliance of confusion assessment method for the intensive care unit"


def test_cam_icu_compliance_search_layers_keep_specific_intent() -> None:
    context = SearchContext(
        topic=expand_acronyms("complianse of cam icu"),
        search_purpose=SEARCH_PURPOSE_RESEARCH,
    )

    layers = build_search_layers(context, candidate_depth=30)
    joined_queries = " ".join(layer.query for layer in layers)

    assert "CAM-ICU" in joined_queries
    assert "Confusion Assessment Method for the Intensive Care Unit" in joined_queries
    assert "compliance" in joined_queries
    assert "delirium" in joined_queries
    assert "compliance" in user_intent_terms(context)


def test_cam_icu_compliance_downranks_generic_icu_papers() -> None:
    context = SearchContext(
        topic=expand_acronyms("complianse of cam icu"),
        search_purpose=SEARCH_PURPOSE_RESEARCH,
    )
    relevant = {
        "title": "Improving CAM-ICU compliance through nurse education in the intensive care unit",
        "abstract": (
            "CAM-ICU delirium screening compliance and adherence improved after "
            "implementation of nurse education in critical care."
        ),
        "publication_types": ["Journal Article", "Observational Study"],
        "journal": "Critical Care Nurse",
        "pmid": "111",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "citation_count": 25,
        "citation_source": "OpenAlex",
        "year": 2021,
    }
    off_topic = {
        "title": "Measurement of irradiation doses secondary to bedside radiographs in a medical intensive care unit",
        "abstract": (
            "A quality audit measured radiograph exposure among nurses and patients "
            "in a medical intensive care unit."
        ),
        "publication_types": ["Journal Article"],
        "journal": "Intensive Care Medicine",
        "pmid": "222",
        "url": "https://pubmed.ncbi.nlm.nih.gov/222/",
        "citation_count": 60,
        "citation_source": "OpenAlex",
        "year": 2020,
    }

    relevant_scored = score_and_classify_paper(relevant, context, {})
    off_topic_scored = score_and_classify_paper(off_topic, context, {})

    assert relevant_scored["topic_match_level"] in {"direct", "direct_synonym", "strong_component", "abstract_only"}
    assert off_topic_scored["topic_match_level"] == "noise"
    assert off_topic_scored["tier"] == "Noise / manual review"
    assert relevant_scored["total_score"] > off_topic_scored["total_score"]


def test_cam_icu_compliance_requires_cam_or_compliance_anchor() -> None:
    context = SearchContext(
        topic=expand_acronyms("complianse of cam icu"),
        search_purpose=SEARCH_PURPOSE_RESEARCH,
    )
    generic_delirium = {
        "title": "Nursing Understanding and Perceptions of Delirium in a Burn ICU",
        "abstract": "This survey assessed current knowledge and beliefs about delirium among nurses in a burn ICU.",
        "publication_types": ["Journal Article"],
        "journal": "Journal of burn care & research",
        "pmid": "333",
        "url": "https://pubmed.ncbi.nlm.nih.gov/333/",
        "citation_count": 21,
        "citation_source": "OpenAlex",
        "year": 2019,
    }

    scored = score_and_classify_paper(generic_delirium, context, {})

    assert scored["topic_match_level"] in {"background", "noise"}
    assert scored["tier"] in {"Tier 3: Background", "Tier 4: Low priority", "Noise / manual review"}


def test_cam_icu_compliance_rejects_generic_icu_compliance_trials() -> None:
    context = SearchContext(
        topic=expand_acronyms("complianse of cam icu"),
        search_purpose=SEARCH_PURPOSE_RESEARCH,
    )
    restraint_trial = {
        "title": (
            "Stepped wedge cluster randomised controlled trial to assess the impact "
            "of a decision support tool for physical restraint use in intensive care units"
        ),
        "abstract": (
            "The trial tested a nursing management strategy for physical restraint use "
            "in ICU patients. Physical restraints have been associated with delirium, "
            "and the study measured compliance with the intervention."
        ),
        "publication_types": ["Journal Article", "Randomized Controlled Trial"],
        "journal": "BMJ Open",
        "pmid": "444",
        "url": "https://pubmed.ncbi.nlm.nih.gov/444/",
        "citation_count": 10,
        "citation_source": "OpenAlex",
        "year": 2024,
    }

    scored = score_and_classify_paper(restraint_trial, context, {})

    assert scored["topic_match_level"] in {"background", "noise"}
    assert scored["tier"] in {"Tier 4: Low priority", "Noise / manual review"}
    assert scored["reading_section"] == "Low-priority/background papers"
