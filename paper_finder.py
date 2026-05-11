from __future__ import annotations

import copy
import csv
import functools
import io
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace as dataclass_replace
from datetime import date
from pathlib import Path
from typing import Any, Callable

import requests

from evidence_engine import build_evidence_review
from topic_primer import TopicPrimer, prime_topic


TOPICS_DIR = Path(__file__).resolve().parent / "topics"


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_LINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
MESH_SEARCH_URL = PUBMED_SEARCH_URL  # same endpoint, db=mesh
MESH_FETCH_URL = PUBMED_FETCH_URL
MESH_MAX_DESCRIPTORS = 8
MESH_MAX_QUERY_CLAUSES = 40
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

REQUEST_TIMEOUT = (5, 12)
DEFAULT_HEADERS = {
    "User-Agent": "QualityFirstPaperFinder/1.0; verified-metadata-literature-tool"
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "can",
    "care",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "patient",
    "patients",
    "should",
    "study",
    "the",
    "their",
    "to",
    "using",
    "what",
    "when",
    "with",
    "without",
}

ICU_TERMS = [
    "icu",
    "intensive care",
    "critical care",
    "critically ill",
    "critical illness",
    "intensive care unit",
    "intensive care units",
    "mechanical ventilation",
    "ventilator",
    "vasopressor",
    "shock",
]

LMIC_TERMS = [
    "india",
    "indian",
    "lmic",
    "low income",
    "middle income",
    "resource limited",
    "resource-limited",
    "developing country",
    "developing countries",
]

GAP_TERMS = [
    "gap",
    "uncertain",
    "uncertainty",
    "controversy",
    "controversial",
    "unresolved",
    "implementation",
    "feasibility",
    "cost",
    "external validation",
    "subgroup",
    "trial protocol",
]

RARE_CASE_TERMS = [
    "case report",
    "case reports",
    "case series",
    "rare",
    "unusual",
    "uncommon",
    "atypical",
    "complication",
    "complications",
    "adverse event",
    "adverse events",
    "adverse drug",
    "device-related",
    "presentation",
    "presenting",
    "association",
    "associations",
    "diagnostic finding",
    "imaging finding",
    "laboratory finding",
]

DATABASE_TERMS = [
    "mimic",
    "eicu",
    "physionet",
    "icnarc",
    "anzics",
    "nethermap",
    "database",
    "registry",
    "national inpatient sample",
]

JOURNAL_SCORE = {"Q1": 12, "Q2": 9, "Q3": 4, "Q4": 1}
JOURNAL_MAJOR_BONUS = 5
TIER_ORDER = {
    "Tier 1: Must-read": 1,
    "Tier 2: Useful supporting": 2,
    "Tier 3: Background": 3,
    "Tier 4: Low priority": 4,
    "Noise / manual review": 5,
}
TIER_BY_ORDER = {order: tier for tier, order in TIER_ORDER.items()}
TOPIC_LEVEL_ORDER = {
    "direct": 0,
    "abstract_only": 1,
    "partial": 2,
    "background": 3,
    "noise": 4,
}
SEARCH_PURPOSE_KNOWLEDGE = "Knowledge / Learning"
SEARCH_PURPOSE_RESEARCH = "Research"
SEARCH_PURPOSE_DEEP = "Deep Search"
SEARCH_PURPOSE_RARE = "Rare / Case Report"
SEARCH_PURPOSE_DEFAULT = SEARCH_PURPOSE_RESEARCH
SEARCH_PURPOSE_OPTIONS = [
    SEARCH_PURPOSE_KNOWLEDGE,
    SEARCH_PURPOSE_RESEARCH,
    SEARCH_PURPOSE_DEEP,
    SEARCH_PURPOSE_RARE,
]
SEARCH_PURPOSE_PRESETS: dict[str, dict[str, Any]] = {
    SEARCH_PURPOSE_KNOWLEDGE: {
        "candidate_depth": 80,
        "enrichment_limit": 80,
        "review_max_sources": 60,
        "semantic_scholar": False,
        "ai_gap_analysis": False,
        "runtime_label": "Usually fastest",
        "description": "Best reviews and conceptual papers.",
    },
    SEARCH_PURPOSE_RESEARCH: {
        "candidate_depth": 130,
        "enrichment_limit": 120,
        "review_max_sources": 100,
        "semantic_scholar": False,
        "ai_gap_analysis": True,
        "runtime_label": "Deeper search",
        "description": "Original studies, RCTs, cohorts, and gap-defining evidence.",
    },
    SEARCH_PURPOSE_DEEP: {
        "candidate_depth": 200,
        "enrichment_limit": 150,
        "review_max_sources": 200,
        "semantic_scholar": False,
        "ai_gap_analysis": False,
        "runtime_label": "Most exhaustive",
        "description": "Broad exhaustive collection.",
    },
    SEARCH_PURPOSE_RARE: {
        "candidate_depth": 160,
        "enrichment_limit": 100,
        "review_max_sources": 160,
        "semantic_scholar": False,
        "ai_gap_analysis": False,
        "runtime_label": "Broad rare-event search",
        "description": "Unusual cases, rare complications, and low-frequency reports.",
    },
}
SEARCH_PURPOSE_ALIASES = {
    "Learn / teach topic": SEARCH_PURPOSE_KNOWLEDGE,
    "Find research gaps": SEARCH_PURPOSE_RESEARCH,
    "Systematic review pool": SEARCH_PURPOSE_DEEP,
}


def search_purpose_config(search_purpose: str) -> dict[str, Any]:
    normalized = normalized_search_purpose(search_purpose)
    config = SEARCH_PURPOSE_PRESETS.get(normalized) or SEARCH_PURPOSE_PRESETS[SEARCH_PURPOSE_DEFAULT]
    return dict(config)


def normalized_search_purpose(search_purpose: str) -> str:
    if search_purpose in SEARCH_PURPOSE_PRESETS:
        return search_purpose
    return SEARCH_PURPOSE_ALIASES.get(search_purpose, SEARCH_PURPOSE_DEFAULT)


@functools.lru_cache(maxsize=1)
def load_topic_profiles() -> tuple[dict[str, Any], ...]:
    if not TOPICS_DIR.is_dir():
        return ()
    profiles: list[dict[str, Any]] = []
    for path in sorted(TOPICS_DIR.glob("*.json")):
        if path.name.startswith(".") or path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("triggers"):
            profiles.append(data)
    return tuple(profiles)


@functools.lru_cache(maxsize=1)
def load_acronyms() -> dict[str, str]:
    path = TOPICS_DIR / "_acronyms.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k).lower().strip(): str(v).lower().strip()
        for k, v in data.items()
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
    }


def expand_acronyms(topic: str) -> str:
    if not topic:
        return topic
    acronyms = load_acronyms()
    if not acronyms:
        return topic
    text = normalize_space(topic).lower()
    if text in acronyms:
        return acronyms[text]
    words = text.split()
    if not words:
        return topic
    expanded_words = [acronyms.get(word, word) for word in words]
    expanded = " ".join(expanded_words)
    return expanded if expanded != text else topic


def expected_paper_order(profile: dict[str, Any] | None) -> dict[str, int]:
    if not profile:
        return {}
    return {item["pmid"]: index for index, item in enumerate(profile.get("expected_papers", []))}

MAJOR_JOURNAL_TERMS = [
    "new england journal of medicine",
    "n engl j med",
    "lancet",
    "jama",
    "bmj",
    "stroke",
    "european respiratory review",
    "eur respir rev",
    "journal of thrombosis and haemostasis",
    "j thromb haemost",
    "intensive care medicine",
    "critical care",
]


@dataclass(frozen=True)
class SearchContext:
    topic: str
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    question_type: str = "General evidence map"
    search_purpose: str = SEARCH_PURPOSE_DEFAULT
    current_year: int = date.today().year
    gemini_api_key: str = ""


@dataclass(frozen=True)
class SearchLayer:
    name: str
    purpose: str
    query: str
    retmax: int | None = None


def build_search_layers(
    context: SearchContext,
    candidate_depth: int = 50,
    email: str = "",
    api_key: str = "",
) -> list[SearchLayer]:
    purpose = normalized_search_purpose(context.search_purpose)
    topic = context.topic.strip()
    topic_query = build_topic_query(topic, email=email, api_key=api_key)
    recent_start_year = max(1900, context.current_year - 2)
    ico_clause = (
        '"intensive care units"[MeSH Terms] OR "critical care"[MeSH Terms] '
        'OR ICU OR "intensive care" OR "critical care" OR "critically ill"'
    )
    broad_design_clause = (
        '"systematic review"[Publication Type] OR "meta-analysis"[Publication Type] '
        'OR "randomized controlled trial"[Publication Type] OR guideline[Publication Type] '
        'OR cohort OR observational OR database OR registry'
    )

    pico_terms = [topic_query]
    pico_terms.extend(
        part.strip()
        for part in [
            context.population,
            context.intervention,
            context.comparator,
            context.outcome,
        ]
        if part.strip()
    )
    focused_terms = " AND ".join(f"({part})" for part in pico_terms)

    question_type_terms = {
        "Intervention or treatment": '"randomized controlled trial" OR trial OR treatment OR therapy',
        "Diagnosis": '"diagnostic accuracy" OR sensitivity OR specificity OR diagnosis',
        "Prognosis or prediction": 'prognosis OR prediction OR cohort OR "risk model"',
        "Implementation or cost": 'implementation OR feasibility OR cost OR audit OR protocol',
        "General evidence map": broad_design_clause,
    }.get(context.question_type, broad_design_clause)

    gap_clause = (
        'India OR Indian OR LMIC OR "resource limited" OR "low income" '
        'OR implementation OR cost OR feasibility OR "external validation" '
        'OR subgroup OR "trial protocol" OR uncertainty OR "research gap"'
    )
    review_guideline_clause = (
        'guideline[Publication Type] OR practice guideline[Publication Type] OR '
        '"scientific statement" OR statement OR consensus OR "society statement" OR '
        'AHA OR ASA OR ESO OR ESICM OR SCCM OR "Neurocritical Care Society" OR '
        'review[Publication Type] OR "systematic review"[Publication Type] OR '
        '"meta-analysis"[Publication Type] OR review OR "practical review" OR '
        '"comprehensive review" OR "state of the art" OR update OR seminar OR primer OR '
        '"clinical review"'
    )
    landmark_clause = (
        'landmark OR classic OR "highly cited" OR "current concepts" OR '
        'review[Publication Type] OR "New England Journal of Medicine" OR Lancet OR JAMA OR BMJ OR Stroke'
    )
    recent_update_clause = (
        f'("{recent_start_year}"[dp] : "3000"[dp]) AND '
        '(review OR update OR guideline OR statement OR consensus OR trial OR cohort)'
    )
    if purpose == SEARCH_PURPOSE_KNOWLEDGE:
        broad_retmax = max(candidate_depth // 2, 50)
        review_retmax = max(candidate_depth, 80)
        focused_retmax = max(25, candidate_depth // 3)
        gap_retmax = max(15, candidate_depth // 4)
    elif purpose == SEARCH_PURPOSE_DEEP:
        broad_retmax = max(candidate_depth, 200)
        review_retmax = max(candidate_depth, 120)
        focused_retmax = max(candidate_depth, 150)
        gap_retmax = max(50, candidate_depth // 2)
    elif purpose == SEARCH_PURPOSE_RARE:
        broad_retmax = max(candidate_depth, 140)
        review_retmax = max(40, candidate_depth // 3)
        focused_retmax = max(candidate_depth, 120)
        gap_retmax = max(candidate_depth, 120)
    else:
        broad_retmax = max(candidate_depth, 100)
        review_retmax = max(50, candidate_depth // 2)
        focused_retmax = max(50, candidate_depth // 2)
        gap_retmax = max(40, candidate_depth // 2)

    layers = [
        SearchLayer(
            name="Broad",
            purpose="Capture the field without ICU-only narrowing before scoring.",
            query=f"({topic_query}) AND ({broad_design_clause})",
            retmax=broad_retmax,
        ),
        SearchLayer(
            name="Review/guideline",
            purpose="Mandatory discovery layer for statements, guidelines, consensus papers, and major reviews.",
            query=f"({topic_query}) AND ({review_guideline_clause})",
            retmax=review_retmax,
        ),
        SearchLayer(
            name="Landmark/classic",
            purpose="Mandatory discovery layer for older classic and highly cited review papers.",
            query=f"({topic_query}) AND ({landmark_clause})",
            retmax=review_retmax,
        ),
        SearchLayer(
            name="Recent update",
            purpose="Mandatory discovery layer for recent reviews, updates, trials, and cohorts.",
            query=f"({topic_query}) AND ({recent_update_clause})",
            retmax=review_retmax,
        ),
        SearchLayer(
            name="Focused",
            purpose="Answer the exact question using PICO and question-type terms.",
            query=f"{focused_terms} AND ({question_type_terms})",
            retmax=focused_retmax,
        ),
        SearchLayer(
            name="ICU/gap",
            purpose="Find ICU, local, implementation, subgroup, and future-study gaps.",
            query=f"({topic_query}) AND ({ico_clause}) AND ({gap_clause})",
            retmax=gap_retmax,
        ),
    ]
    if purpose == SEARCH_PURPOSE_DEEP:
        layers.insert(
            0,
            SearchLayer(
                name="Screening pool",
                purpose="Broad candidate pool for systematic-review style title/abstract screening.",
                query=f"({topic_query})",
                retmax=max(candidate_depth, 200),
            ),
        )
    if purpose == SEARCH_PURPOSE_RARE:
        rare_clause = (
            '"case reports"[Publication Type] OR "case report" OR "case series" OR '
            'rare OR unusual OR uncommon OR atypical OR complication OR complications OR '
            '"adverse event" OR "adverse drug" OR presentation OR association OR imaging OR laboratory'
        )
        layers.insert(
            0,
            SearchLayer(
                name="Rare/case reports",
                purpose="Find rare presentations, complications, adverse events, case reports, and case series.",
                query=f"({topic_query}) AND ({rare_clause})",
                retmax=max(candidate_depth, 150),
            ),
        )
    return layers


def build_topic_query(topic: str, email: str = "", api_key: str = "") -> str:
    profile = topic_profile(topic)
    profile_query = str(profile.get("pubmed_query", "")).strip() if profile else ""

    mesh_records = discover_mesh(topic, email=email, api_key=api_key)
    mesh_clauses = mesh_query_clauses(mesh_records)
    mesh_query = " OR ".join(mesh_clauses)

    expansion_query = ""
    if profile:
        expansion_clauses: list[str] = []
        for raw_term in profile.get("query_expansion_terms", []) or []:
            term = str(raw_term).strip().replace('"', "")
            if not term:
                continue
            expansion_clauses.append(f'"{term}"[Title/Abstract]')
            if len(expansion_clauses) >= 15:
                break
        expansion_query = " OR ".join(expansion_clauses)

    parts: list[str] = []
    if profile_query:
        parts.append(f"({profile_query})")
    if mesh_query:
        parts.append(f"({mesh_query})")
    if expansion_query:
        parts.append(f"({expansion_query})")
    if parts:
        return " OR ".join(parts)

    translation = pubmed_translation(topic, email=email, api_key=api_key)
    if translation:
        return translation
    return topic


def run_quality_first_search(
    context: SearchContext,
    max_results_per_layer: int = 25,
    email: str = "",
    use_openalex: bool = True,
    use_semantic_scholar: bool = False,
    enrichment_limit: int = 10,
    quartile_overrides: dict[str, dict[str, str]] | None = None,
    manual_google_scholar_notes: str = "",
    progress_callback: Callable[[str, int, int], None] | None = None,
    ncbi_api_key: str = "",
) -> dict[str, Any]:
    context = dataclass_replace(
        context,
        search_purpose=normalized_search_purpose(context.search_purpose),
    )
    purpose_config = search_purpose_config(context.search_purpose)
    original_topic = context.topic
    expanded_topic = expand_acronyms(original_topic)
    if expanded_topic.strip().lower() != original_topic.strip().lower():
        context = dataclass_replace(context, topic=expanded_topic)

    primer_status = register_primer_if_needed(
        context.topic,
        gemini_api_key=context.gemini_api_key,
        email=email,
        api_key=ncbi_api_key,
    )

    layers = build_search_layers(
        context,
        max_results_per_layer,
        email=email,
        api_key=ncbi_api_key,
    )
    discovered_mesh = discover_mesh(context.topic, email=email, api_key=ncbi_api_key)
    all_papers: list[dict[str, Any]] = []
    errors: list[str] = []
    automatically_retrieved_pmids: set[str] = set()
    expected_papers = expected_papers_for_topic(context.topic)
    api_discovery: dict[str, Any] = {
        "pmids": [],
        "related_pmids": [],
        "sources": [],
        "errors": [],
        "warnings": [],
        "pmid_layers": {},
        "pmid_reasons": {},
    }

    def _notify(message: str, completed: int, total: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(message, completed, total)
        except Exception:
            pass

    def _fetch_layer(layer: SearchLayer) -> tuple[SearchLayer, list[str], list[dict[str, Any]], str | None]:
        try:
            ids = search_pubmed(
                layer.query,
                layer.retmax or max_results_per_layer,
                email=email,
                api_key=ncbi_api_key,
            )
            papers = fetch_pubmed_records(ids, email=email, api_key=ncbi_api_key) if ids else []
            for paper in papers:
                paper["search_layers"] = [layer.name]
                paper["search_origin"] = "PubMed"
                paper["source_records"] = ["PMID"]
            return layer, ids, papers, None
        except requests.RequestException as exc:
            return layer, [], [], f"{layer.name} PubMed search failed: {friendly_request_error(exc)}"
        except ET.ParseError as exc:
            return layer, [], [], f"{layer.name} PubMed XML parsing failed: {exc}"

    total_layers = len(layers)
    completed_layers = 0
    layer_workers = 4 if ncbi_api_key.strip() else 2
    _notify(f"Starting {total_layers} parallel PubMed searches", 0, total_layers)
    with ThreadPoolExecutor(max_workers=layer_workers) as executor:
        futures = [executor.submit(_fetch_layer, layer) for layer in layers]
        for future in as_completed(futures):
            layer, ids, papers, error = future.result()
            completed_layers += 1
            if error:
                errors.append(error)
                _notify(f"Layer '{layer.name}' failed", completed_layers, total_layers)
            else:
                automatically_retrieved_pmids.update(ids)
                all_papers.extend(papers)
                _notify(
                    f"Layer '{layer.name}' done — {len(papers)} candidates",
                    completed_layers,
                    total_layers,
                )

    _notify("Running API discovery supervisor", total_layers, total_layers)
    try:
        api_discovery = run_api_discovery_supervisor(
            context,
            email=email,
            ncbi_api_key=ncbi_api_key,
            per_query_limit=max(10, min(25, max_results_per_layer // 3)),
        )
        api_pmids = list(api_discovery.get("pmids", []))
        if api_pmids:
            api_records = fetch_pubmed_records(
                api_pmids,
                email=email,
                api_key=ncbi_api_key,
            )
            api_layers = api_discovery.get("pmid_layers", {})
            api_reasons = api_discovery.get("pmid_reasons", {})
            for paper in api_records:
                pmid = str(paper.get("pmid", ""))
                paper["search_layers"] = api_layers.get(pmid) or ["API discovery supervisor"]
                paper["search_origin"] = "API discovery supervisor"
                paper["source_records"] = ["PMID", "API supervisor"]
                paper["api_discovery_reason"] = "; ".join(api_reasons.get(pmid, []))
            all_papers.extend(api_records)
            automatically_retrieved_pmids.update(api_pmids)
            _notify(
                f"API discovery found {len(api_pmids)} verified PubMed candidates",
                total_layers,
                total_layers,
            )
    except requests.RequestException as exc:
        api_discovery.setdefault("errors", []).append(f"API discovery failed: {friendly_request_error(exc)}")
    except ET.ParseError as exc:
        api_discovery.setdefault("errors", []).append(f"API discovery PubMed XML parsing failed: {exc}")

    recovered_expected: list[dict[str, str]] = []
    if expected_papers:
        expected_pmids = [item["pmid"] for item in expected_papers]
        missing_from_automatic = [
            item for item in expected_papers if item["pmid"] not in automatically_retrieved_pmids
        ]
        try:
            expected_records = fetch_pubmed_records(
                expected_pmids,
                email=email,
                api_key=ncbi_api_key,
            )
            for paper in expected_records:
                paper["search_layers"] = ["Expected landmark seed"]
                paper["search_origin"] = "PubMed expected-paper sanity seed"
                paper["source_records"] = ["PMID"]
                expected_meta = next(
                    (item for item in expected_papers if item["pmid"] == paper.get("pmid")),
                    None,
                )
                if expected_meta:
                    paper["expected_paper_reason"] = expected_meta["reason"]
                    if paper.get("pmid") not in automatically_retrieved_pmids:
                        recovered_expected.append(expected_meta)
            all_papers.extend(expected_records)
        except requests.RequestException as exc:
            errors.append(f"Expected landmark seed PubMed fetch failed: {friendly_request_error(exc)}")
            recovered_expected = []
        except ET.ParseError as exc:
            errors.append(f"Expected landmark seed PubMed XML parsing failed: {exc}")
            recovered_expected = []
    else:
        missing_from_automatic = []

    deduped = deduplicate_papers(all_papers)
    accepted = [paper for paper in deduped if is_verified(paper)]
    rejected = [paper for paper in deduped if not is_verified(paper)]

    quartile_overrides = quartile_overrides or {}
    enrichment_candidates = rank_for_enrichment(accepted, context, quartile_overrides)
    enrichment_candidates = enrichment_candidates[: max(0, enrichment_limit)]

    def _enrich_openalex(paper: dict[str, Any]) -> None:
        try:
            enrich_with_openalex(paper, email=email)
        except requests.RequestException as exc:
            paper.setdefault("enrichment_warnings", []).append(
                f"OpenAlex unavailable: {friendly_request_error(exc)}"
            )

    def _enrich_semantic_scholar(paper: dict[str, Any]) -> None:
        try:
            enrich_with_semantic_scholar(paper)
        except requests.RequestException as exc:
            paper.setdefault("enrichment_warnings", []).append(
                f"Semantic Scholar unavailable: {friendly_request_error(exc)}"
            )

    if use_openalex and enrichment_candidates:
        _notify(
            f"Enriching {len(enrichment_candidates)} papers with OpenAlex citations",
            total_layers,
            total_layers,
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(_enrich_openalex, enrichment_candidates))

    if use_semantic_scholar and enrichment_candidates:
        _notify(
            f"Cross-checking {len(enrichment_candidates)} papers with Semantic Scholar",
            total_layers,
            total_layers,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(_enrich_semantic_scholar, enrichment_candidates))

    scored = [score_and_classify_paper(paper, context, quartile_overrides) for paper in accepted]
    apply_evidence_family_ranks(scored)
    expected_order = expected_paper_order(topic_profile(context.topic))
    for paper in scored:
        paper["expected_paper_order"] = expected_order.get(str(paper.get("pmid", "")), 999)
    scored.sort(key=paper_sort_key)
    missing_expected = missing_expected_papers(expected_papers, scored)

    summary = generate_knowledge_summary(scored, context, manual_google_scholar_notes)
    gap_map = generate_gap_map(scored, context)
    subtopic_coverage = compute_subtopic_coverage(scored, context)

    result = {
        "search_date": date.today().isoformat(),
        "layers": layers,
        "papers": scored,
        "retrieved_count": len(all_papers),
        "deduped_count": len(deduped),
        "rejected_unverified": rejected,
        "errors": errors,
        "summary": summary,
        "gap_map": gap_map,
        "subtopic_coverage": subtopic_coverage,
        "expected_papers": expected_papers,
        "recovered_expected": recovered_expected,
        "missing_expected": missing_expected,
        "missing_from_automatic": missing_from_automatic,
        "manual_google_scholar_notes": manual_google_scholar_notes.strip(),
        "enrichment_limit": enrichment_limit,
        "topic_used": context.topic,
        "topic_original": original_topic,
        "topic_expanded": context.topic if context.topic.strip().lower() != original_topic.strip().lower() else "",
        "mesh_discovered": discovered_mesh,
        "topic_primer_status": primer_status,
        "api_discovery": api_discovery,
        "search_purpose": context.search_purpose,
        "search_purpose_config": {
            key: value
            for key, value in purpose_config.items()
            if key in {"description", "runtime_label", "review_max_sources", "ai_gap_analysis"}
        },
        "question_context": {
            "topic": context.topic,
            "original_topic": original_topic,
            "population": context.population,
            "intervention": context.intervention,
            "comparator": context.comparator,
            "outcome": context.outcome,
            "question_type": context.question_type,
            "search_purpose": context.search_purpose,
        },
    }
    result["evidence_review"] = build_evidence_review(
        result,
        max_sources=int(purpose_config.get("review_max_sources", 80)),
        gemini_key=context.gemini_api_key if purpose_config.get("ai_gap_analysis") else "",
        generate_ai_gaps=bool(purpose_config.get("ai_gap_analysis")),
    )
    return result


def expected_papers_for_topic(topic: str) -> list[dict[str, str]]:
    profile = topic_profile(topic)
    if not profile:
        return []
    return [dict(item) for item in profile.get("expected_papers", [])]


def run_api_discovery_supervisor(
    context: SearchContext,
    email: str = "",
    ncbi_api_key: str = "",
    per_query_limit: int = 15,
) -> dict[str, Any]:
    """Use public scholarly APIs as a retrieval supervisor before scoring.

    The supervisor does not admit unverified citations. It gathers candidate
    PMIDs from narrow PubMed searches, Europe PMC, OpenAlex, and PubMed related
    articles, then the main pipeline fetches the actual PubMed records.
    """
    pmids: list[str] = []
    pmid_layers: dict[str, list[str]] = defaultdict(list)
    pmid_reasons: dict[str, list[str]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def add_pmids(found_pmids: list[str], layer: str, query: str, reason: str) -> None:
        clean_pmids = []
        for pmid in found_pmids:
            normalized = normalize_pmid(pmid)
            if not normalized:
                continue
            clean_pmids.append(normalized)
            if normalized not in pmids:
                pmids.append(normalized)
            if layer not in pmid_layers[normalized]:
                pmid_layers[normalized].append(layer)
            if reason and reason not in pmid_reasons[normalized]:
                pmid_reasons[normalized].append(reason)
        sources.append(
            {
                "source": layer,
                "query": query,
                "count": len(dict.fromkeys(clean_pmids)),
                "pmids": list(dict.fromkeys(clean_pmids))[:25],
            }
        )

    for layer_name, query, retmax, reason in api_supervisor_pubmed_queries(context, per_query_limit):
        try:
            found = search_pubmed(query, retmax, email=email, api_key=ncbi_api_key)
            add_pmids(found, layer_name, query, reason)
        except requests.RequestException as exc:
            errors.append(f"{layer_name} failed: {friendly_request_error(exc)}")

    for layer_name, query, page_size, reason in api_supervisor_europe_pmc_queries(context, per_query_limit):
        try:
            found = search_europe_pmc_pmids(query, page_size=page_size)
            add_pmids(found, layer_name, query, reason)
        except requests.RequestException as exc:
            errors.append(f"{layer_name} failed: {friendly_request_error(exc)}")

    for layer_name, query, page_size, reason in api_supervisor_openalex_queries(context, per_query_limit):
        try:
            found = search_openalex_pmids(query, per_page=page_size, email=email)
            add_pmids(found, layer_name, query, reason)
        except requests.RequestException as exc:
            errors.append(f"{layer_name} failed: {friendly_request_error(exc)}")

    related_seed_pmids = pmids[:3]
    related_pmids: list[str] = []
    if related_seed_pmids:
        try:
            related_pmids = pubmed_related_pmids(
                related_seed_pmids,
                retmax=max(20, per_query_limit),
                email=email,
                api_key=ncbi_api_key,
            )
            add_pmids(
                related_pmids,
                "API supervisor - PubMed related",
                ", ".join(related_seed_pmids),
                "PubMed related-article expansion from API-discovered seed papers",
            )
        except requests.RequestException as exc:
            warnings.append(f"API supervisor - PubMed related skipped: {friendly_request_error(exc)}")
        except ET.ParseError as exc:
            warnings.append(f"API supervisor - PubMed related XML parsing skipped: {exc}")

    return {
        "pmids": pmids[:200],
        "related_pmids": related_pmids[:100],
        "sources": sources,
        "errors": errors,
        "warnings": warnings,
        "pmid_layers": {pmid: layers for pmid, layers in pmid_layers.items()},
        "pmid_reasons": {pmid: reasons for pmid, reasons in pmid_reasons.items()},
    }


def api_supervisor_pubmed_queries(
    context: SearchContext,
    per_query_limit: int,
) -> list[tuple[str, str, int, str]]:
    phrase = normalize_space(context.topic)
    quoted_phrase = pubmed_quote(phrase)
    concept_query = api_supervisor_concept_query(context, field="Title/Abstract")
    recent_start_year = max(1900, context.current_year - 2)
    review_terms = (
        'review[Publication Type] OR review OR "clinical review" OR '
        '"comprehensive review" OR "state of the art" OR update OR primer OR seminar'
    )
    queries: list[tuple[str, str, int, str]] = []
    if quoted_phrase:
        queries.append(
            (
                "API supervisor - PubMed exact",
                f'"{quoted_phrase}"[Title] OR "{quoted_phrase}"[Title/Abstract]',
                max(5, per_query_limit),
                "Exact title/topic phrase search",
            )
        )
    if concept_query:
        queries.append(
            (
                "API supervisor - PubMed focused review",
                f"({concept_query}) AND ({review_terms})",
                per_query_limit,
                "Focused review/update search from topic concepts",
            )
        )
        queries.append(
            (
                "API supervisor - PubMed recent focused",
                f"({concept_query}) AND (\"{recent_start_year}\"[dp] : \"3000\"[dp])",
                per_query_limit,
                "Recent focused-topic search from topic concepts",
            )
        )
    return queries


def api_supervisor_europe_pmc_queries(
    context: SearchContext,
    per_query_limit: int,
) -> list[tuple[str, str, int, str]]:
    phrase = normalize_space(context.topic)
    keyword_query = " ".join(api_supervisor_keywords(context)[:8])
    queries: list[tuple[str, str, int, str]] = []
    if phrase:
        queries.append(
            (
                "API supervisor - Europe PMC exact",
                f'"{phrase}"',
                max(5, per_query_limit),
                "Europe PMC exact phrase search",
            )
        )
    if keyword_query:
        queries.append(
            (
                "API supervisor - Europe PMC recent review",
                f"{keyword_query} review sort_date:y",
                per_query_limit,
                "Europe PMC recent review search",
            )
        )
    return queries


def api_supervisor_openalex_queries(
    context: SearchContext,
    per_query_limit: int,
) -> list[tuple[str, str, int, str]]:
    phrase = normalize_space(context.topic)
    keyword_query = " ".join(api_supervisor_keywords(context)[:8])
    queries: list[tuple[str, str, int, str]] = []
    if phrase:
        queries.append(
            (
                "API supervisor - OpenAlex exact",
                phrase,
                max(5, per_query_limit),
                "OpenAlex title/abstract/full-text search",
            )
        )
    if keyword_query and keyword_query.lower() != phrase.lower():
        queries.append(
            (
                "API supervisor - OpenAlex concept",
                keyword_query,
                per_query_limit,
                "OpenAlex concept search from topic keywords",
            )
        )
    return queries


def api_supervisor_concept_query(context: SearchContext, field: str = "Title/Abstract") -> str:
    keywords_for_query = api_supervisor_keywords(context)
    if not keywords_for_query:
        return ""
    clauses = []
    phrase = normalize_space(context.topic)
    if len(phrase.split()) >= 3:
        clauses.append(f'"{pubmed_quote(phrase)}"[{field}]')
    for term in keywords_for_query[:8]:
        clauses.append(pubmed_term_clause(term, field))
    # Exact phrase OR all core terms. The phrase catches titles; the AND chain
    # catches near-misses with morphology or reordered terms.
    if len(clauses) == 1:
        return clauses[0]
    and_terms = " AND ".join(clauses[1:])
    return f"({clauses[0]}) OR ({and_terms})" if and_terms else clauses[0]


def api_supervisor_keywords(context: SearchContext) -> list[str]:
    topic = normalize_space(context.topic)
    profile = topic_profile(topic)
    raw_terms: list[str] = []
    if profile:
        raw_terms.extend(str(term) for term in profile.get("query_expansion_terms", [])[:6])
        raw_terms.extend(str(term) for term in profile.get("must_include_concepts", [])[:6])
    raw_terms.extend(keywords(topic))
    raw_terms.extend(keywords(context.population))
    raw_terms.extend(keywords(context.outcome))

    weak_terms = {
        "effect",
        "effects",
        "impact",
        "management",
        "treatment",
        "therapy",
        "review",
        "clinical",
        "study",
        "analysis",
        "outcome",
        "outcomes",
    }
    out: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        cleaned = normalize_space(str(term).lower())
        if not cleaned or cleaned in weak_terms or len(cleaned) < 3:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= 12:
            break
    return out


def pubmed_quote(text: str) -> str:
    return normalize_space(text).replace('"', "")


def pubmed_term_clause(term: str, field: str) -> str:
    cleaned = pubmed_quote(term)
    if not cleaned:
        return ""
    if " " in cleaned or "-" in cleaned:
        return f'"{cleaned}"[{field}]'
    if len(cleaned) >= 5:
        return f"{cleaned}*[{field}]"
    return f'"{cleaned}"[{field}]'


def normalize_pmid(value: str | int | None) -> str:
    if value is None:
        return ""
    match = re.search(r"\b\d{5,10}\b", str(value))
    return match.group(0) if match else ""


def missing_expected_papers(
    expected_papers: list[dict[str, str]],
    scored_papers: list[dict[str, Any]],
) -> list[dict[str, str]]:
    retrieved_pmids = {paper.get("pmid", "") for paper in scored_papers}
    retrieved_titles = {normalize_title(paper.get("title", "")) for paper in scored_papers}
    missing = []
    for expected in expected_papers:
        expected_title = normalize_title(expected["title"])
        if expected.get("pmid"):
            if expected["pmid"] in retrieved_pmids:
                continue
        elif expected_title in retrieved_titles:
            continue
        missing.append(expected)
    return missing


def rank_for_enrichment(
    papers: list[dict[str, Any]],
    context: SearchContext,
    quartile_overrides: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    preliminary = [
        score_and_classify_paper(paper, context, quartile_overrides)
        for paper in papers
    ]
    preliminary.sort(key=paper_sort_key)
    rank_by_identity = {
        paper_identity(paper): index
        for index, paper in enumerate(preliminary)
    }
    return sorted(papers, key=lambda paper: rank_by_identity.get(paper_identity(paper), 10_000))


def paper_identity(paper: dict[str, Any]) -> str:
    if paper.get("pmid"):
        return f"pmid:{paper['pmid']}"
    if paper.get("doi"):
        return f"doi:{clean_doi(paper['doi']).lower()}"
    return f"title:{normalize_title(paper.get('title', ''))}"


def friendly_request_error(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        detail = ""
        try:
            body = (response.text or "").strip()
            if body:
                snippet = re.sub(r"\s+", " ", body)[:240]
                snippet = re.sub(
                    r'("api[-_]?key"\s*:\s*")([^"]+)(")',
                    r"\1[redacted]\3",
                    snippet,
                    flags=re.IGNORECASE,
                )
                snippet = re.sub(
                    r'(api[-_]?key=)([^&\s"]+)',
                    r"\1[redacted]",
                    snippet,
                    flags=re.IGNORECASE,
                )
                detail = f" — {snippet}"
        except Exception:
            detail = ""
        return f"HTTP {response.status_code} from source API{detail}"
    text = str(exc)
    if "NameResolutionError" in text or "Failed to resolve" in text:
        return "network/DNS unavailable from the app process"
    if "Read timed out" in text or "timed out" in text:
        return "source API timed out"
    if "Connection refused" in text:
        return "source API connection refused"
    return text[:240]


def _pubmed_get(url: str, params: dict[str, str], max_retries: int = 4) -> requests.Response:
    delay = 0.8
    last_exc: requests.RequestException | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries - 1:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            time.sleep(min(wait, 8.0))
            delay *= 2
            continue
        response.raise_for_status()
        return response
    if last_exc:
        raise last_exc
    raise requests.RequestException("PubMed request failed without response")


@functools.lru_cache(maxsize=256)
def _cached_search_pubmed(query: str, retmax: int, email: str, api_key: str) -> tuple[str, ...]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "sort": "relevance",
        "tool": "quality_first_paper_finder",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    response = _pubmed_get(PUBMED_SEARCH_URL, params)
    payload = response.json()
    return tuple(payload.get("esearchresult", {}).get("idlist", []))


def search_pubmed(query: str, retmax: int, email: str = "", api_key: str = "") -> list[str]:
    return list(_cached_search_pubmed(query, retmax, (email or "").strip(), (api_key or "").strip()))


@functools.lru_cache(maxsize=256)
def _cached_search_europe_pmc_pmids(query: str, page_size: int) -> tuple[str, ...]:
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": str(page_size),
    }
    response = requests.get(
        EUROPE_PMC_SEARCH_URL,
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("resultList", {}).get("result", [])
    pmids: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        pmid = normalize_pmid(item.get("pmid") or item.get("id"))
        source = str(item.get("source") or "").upper()
        if pmid and (source in {"MED", "PMC", ""} or item.get("pmid")):
            pmids.append(pmid)
    return tuple(dict.fromkeys(pmids))


def search_europe_pmc_pmids(query: str, page_size: int = 15) -> list[str]:
    return list(_cached_search_europe_pmc_pmids(query.strip(), max(1, min(page_size, 100))))


@functools.lru_cache(maxsize=256)
def _cached_search_openalex_pmids(query: str, per_page: int, email: str) -> tuple[str, ...]:
    params = {
        "search": query,
        "filter": "has_pmid:true",
        "per-page": str(per_page),
    }
    if email:
        params["mailto"] = email
    response = requests.get(
        OPENALEX_WORKS_URL,
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    pmids: list[str] = []
    for work in results:
        if not isinstance(work, dict):
            continue
        ids = work.get("ids") or {}
        pmid = normalize_pmid(ids.get("pmid") or "")
        if pmid:
            pmids.append(pmid)
    return tuple(dict.fromkeys(pmids))


def search_openalex_pmids(query: str, per_page: int = 15, email: str = "") -> list[str]:
    return list(
        _cached_search_openalex_pmids(
            query.strip(),
            max(1, min(per_page, 50)),
            (email or "").strip(),
        )
    )


@functools.lru_cache(maxsize=128)
def _cached_pubmed_related_pmids(
    pmids_key: tuple[str, ...], retmax: int, email: str, api_key: str
) -> tuple[str, ...]:
    params = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "id": ",".join(pmids_key),
        "cmd": "neighbor_score",
        "retmode": "xml",
        "retmax": str(retmax),
        "tool": "quality_first_paper_finder",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    response = requests.get(
        PUBMED_LINK_URL,
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=(3, 5),
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    related: list[str] = []
    seed_set = set(pmids_key)
    for linkset_db in root.findall(".//LinkSetDb"):
        linkname = text_of(linkset_db.find("./LinkName"))
        if "pubmed_pubmed" not in linkname:
            continue
        for link in linkset_db.findall("./Link/Id"):
            pmid = normalize_pmid(text_of(link))
            if pmid and pmid not in seed_set:
                related.append(pmid)
    if not related:
        for link in root.findall(".//Link/Id"):
            pmid = normalize_pmid(text_of(link))
            if pmid and pmid not in seed_set:
                related.append(pmid)
    return tuple(dict.fromkeys(related))


def pubmed_related_pmids(
    pmids: list[str], retmax: int = 20, email: str = "", api_key: str = ""
) -> list[str]:
    clean_pmids = tuple(dict.fromkeys(normalize_pmid(pmid) for pmid in pmids if normalize_pmid(pmid)))
    if not clean_pmids:
        return []
    return list(
        _cached_pubmed_related_pmids(
            clean_pmids,
            max(1, min(retmax, 100)),
            (email or "").strip(),
            (api_key or "").strip(),
        )
    )


@functools.lru_cache(maxsize=128)
def _cached_fetch_pubmed_records(
    pmids_key: tuple[str, ...], email: str, api_key: str
) -> tuple[dict[str, Any], ...]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids_key),
        "retmode": "xml",
        "tool": "quality_first_paper_finder",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    response = _pubmed_get(PUBMED_FETCH_URL, params)
    root = ET.fromstring(response.text)
    return tuple(parse_pubmed_article(node) for node in root.findall(".//PubmedArticle"))


def fetch_pubmed_records(
    pmids: list[str], email: str = "", api_key: str = ""
) -> list[dict[str, Any]]:
    if not pmids:
        return []
    cached = _cached_fetch_pubmed_records(tuple(pmids), (email or "").strip(), (api_key or "").strip())
    return [copy.deepcopy(record) for record in cached]


def clear_pubmed_caches() -> None:
    _cached_search_pubmed.cache_clear()
    _cached_fetch_pubmed_records.cache_clear()
    _cached_discover_mesh.cache_clear()
    _cached_pubmed_translation.cache_clear()
    _cached_search_europe_pmc_pmids.cache_clear()
    _cached_search_openalex_pmids.cache_clear()
    _cached_pubmed_related_pmids.cache_clear()


@functools.lru_cache(maxsize=128)
def _cached_pubmed_translation(topic: str, email: str, api_key: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        return ""
    params = {
        "db": "pubmed",
        "term": topic,
        "retmode": "json",
        "retmax": "0",
        "tool": "quality_first_paper_finder",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    try:
        response = _pubmed_get(PUBMED_SEARCH_URL, params)
        return str(response.json().get("esearchresult", {}).get("querytranslation", "")).strip()
    except (requests.RequestException, ValueError):
        return ""


def pubmed_translation(topic: str, email: str = "", api_key: str = "") -> str:
    return _cached_pubmed_translation(
        (topic or "").strip(), (email or "").strip(), (api_key or "").strip()
    )


@functools.lru_cache(maxsize=128)
def _cached_discover_mesh(topic: str, email: str, api_key: str) -> tuple[dict[str, Any], ...]:
    topic = (topic or "").strip()
    if not topic:
        return ()
    search_params = {
        "db": "mesh",
        "term": topic,
        "retmode": "json",
        "retmax": str(MESH_MAX_DESCRIPTORS),
        "tool": "quality_first_paper_finder",
    }
    if email:
        search_params["email"] = email
    if api_key:
        search_params["api_key"] = api_key
    try:
        response = _pubmed_get(MESH_SEARCH_URL, search_params)
        uids = response.json().get("esearchresult", {}).get("idlist", [])
    except (requests.RequestException, ValueError):
        return ()
    if not uids:
        return ()

    fetch_params = {
        "db": "mesh",
        "id": ",".join(uids[:MESH_MAX_DESCRIPTORS]),
        "rettype": "full",
        "retmode": "text",
        "tool": "quality_first_paper_finder",
    }
    if email:
        fetch_params["email"] = email
    if api_key:
        fetch_params["api_key"] = api_key
    try:
        response = _pubmed_get(MESH_FETCH_URL, fetch_params)
    except requests.RequestException:
        return ()
    return tuple(_parse_mesh_text(response.text))


def discover_mesh(topic: str, email: str = "", api_key: str = "") -> list[dict[str, Any]]:
    raw_records = [
        copy.deepcopy(record)
        for record in _cached_discover_mesh(
            (topic or "").strip(), (email or "").strip(), (api_key or "").strip()
        )
    ]
    return _rank_mesh_records(raw_records, topic)


_MESH_FIELD_HEADER_RE = re.compile(r"^[A-Z][A-Za-z0-9 ()/-]{2,}:\s*$")


def _parse_mesh_text(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_entry_terms = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        new_record_match = re.match(r"^\d+:\s+(.+)$", line)
        if new_record_match:
            if current and current.get("descriptor"):
                records.append(current)
            current = {
                "descriptor": new_record_match.group(1).strip(),
                "ui": "",
                "tree_numbers": [],
                "entry_terms": [],
            }
            in_entry_terms = False
            continue
        if current is None:
            continue

        if line.startswith("MeSH Unique ID:"):
            current["ui"] = line.split(":", 1)[1].strip()
            in_entry_terms = False
            continue
        if line.startswith("Tree Number(s):"):
            tn_str = line.split(":", 1)[1].strip()
            current["tree_numbers"] = [t.strip() for t in tn_str.split(",") if t.strip()]
            in_entry_terms = False
            continue
        if re.match(r"^Entry Terms?:?\s*$", line, flags=re.IGNORECASE):
            in_entry_terms = True
            continue

        if in_entry_terms:
            indented = line.startswith((" ", "\t"))
            stripped = line.strip()
            if not stripped:
                continue
            if not indented or _MESH_FIELD_HEADER_RE.match(stripped):
                in_entry_terms = False
                continue
            current["entry_terms"].append(stripped)

    if current and current.get("descriptor"):
        records.append(current)
    return records


def _rank_mesh_records(
    records: list[dict[str, Any]], topic: str
) -> list[dict[str, Any]]:
    topic_lower = (topic or "").strip().lower()

    def sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
        name = (record.get("descriptor") or "").strip().lower()
        if not topic_lower:
            return (3, 0, name)
        if name == topic_lower:
            tier = 0
        elif name.startswith(topic_lower) or topic_lower in name:
            tier = 1
        elif any(topic_lower == term.strip().lower() for term in record.get("entry_terms", [])):
            tier = 1
        else:
            tier = 2
        # within a tier, prefer shorter (more specific) names
        return (tier, len(name), name)

    return sorted(records, key=sort_key)


def mesh_query_clauses(records: list[dict[str, Any]]) -> list[str]:
    clauses: list[str] = []
    seen: set[str] = set()
    for record in records:
        descriptor = (record.get("descriptor") or "").strip()
        if descriptor and descriptor.lower() not in seen:
            clauses.append(f'"{descriptor}"[Mesh]')
            clauses.append(f'"{descriptor}"[tw]')
            seen.add(descriptor.lower())
        for term in record.get("entry_terms", []):
            normalized = (term or "").strip()
            if normalized and normalized.lower() not in seen:
                clauses.append(f'"{normalized}"[tw]')
                seen.add(normalized.lower())
                if len(clauses) >= MESH_MAX_QUERY_CLAUSES:
                    return clauses
        if len(clauses) >= MESH_MAX_QUERY_CLAUSES:
            return clauses
    return clauses


def parse_pubmed_article(node: ET.Element) -> dict[str, Any]:
    article = node.find("./MedlineCitation/Article")
    medline = node.find("./MedlineCitation")
    pubmed_data = node.find("./PubmedData")

    pmid = text_of(medline.find("./PMID")) if medline is not None else ""
    title = text_of(article.find("./ArticleTitle")) if article is not None else ""
    journal = ""
    if article is not None:
        journal = text_of(article.find("./Journal/Title")) or text_of(
            article.find("./Journal/ISOAbbreviation")
        )

    abstract_parts = []
    if article is not None:
        for abstract_text in article.findall("./Abstract/AbstractText"):
            label = abstract_text.attrib.get("Label", "").strip()
            chunk = text_of(abstract_text)
            if chunk:
                abstract_parts.append(f"{label}: {chunk}" if label else chunk)
    abstract = " ".join(abstract_parts)

    authors = []
    if article is not None:
        for author in article.findall("./AuthorList/Author")[:8]:
            last = text_of(author.find("./LastName"))
            initials = text_of(author.find("./Initials"))
            collective = text_of(author.find("./CollectiveName"))
            if collective:
                authors.append(collective)
            elif last:
                authors.append(f"{last} {initials}".strip())

    year = extract_year(article)
    publication_types = []
    if article is not None:
        publication_types = [
            text_of(pub_type)
            for pub_type in article.findall("./PublicationTypeList/PublicationType")
            if text_of(pub_type)
        ]

    article_ids: dict[str, str] = {}
    if pubmed_data is not None:
        for article_id in pubmed_data.findall("./ArticleIdList/ArticleId"):
            id_type = article_id.attrib.get("IdType", "").lower()
            value = text_of(article_id)
            if id_type and value:
                article_ids[id_type] = value

    doi = article_ids.get("doi", "")
    if not doi and article is not None:
        for loc_id in article.findall("./ELocationID"):
            if loc_id.attrib.get("EIdType", "").lower() == "doi":
                doi = text_of(loc_id)
                break

    return {
        "title": normalize_space(title),
        "authors": "; ".join(authors),
        "year": year,
        "journal": normalize_space(journal),
        "pmid": pmid,
        "doi": doi.strip(),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "abstract": normalize_space(abstract),
        "publication_types": publication_types,
        "search_layers": [],
        "source_records": ["PMID"] if pmid else [],
        "openalex_id": "",
        "semantic_scholar_url": "",
        "citation_count": None,
        "citation_source": "citation count unavailable",
        "openalex_citations": None,
        "semantic_scholar_citations": None,
    }


def text_of(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return normalize_space("".join(node.itertext()))


def extract_year(article: ET.Element | None) -> int | None:
    if article is None:
        return None
    candidates = [
        article.find("./Journal/JournalIssue/PubDate/Year"),
        article.find("./ArticleDate/Year"),
    ]
    for candidate in candidates:
        year_text = text_of(candidate)
        if year_text and year_text.isdigit():
            return int(year_text)
    medline_date = text_of(article.find("./Journal/JournalIssue/PubDate/MedlineDate"))
    match = re.search(r"(19|20)\d{2}", medline_date)
    return int(match.group(0)) if match else None


def deduplicate_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}

    for paper in papers:
        keys = dedupe_keys(paper)
        existing = None
        for key in keys:
            candidate = index.get(key)
            if candidate is None or conflicting_identity(candidate, paper):
                continue
            existing = candidate
            break
        if existing:
            existing_layers = set(existing.get("search_layers", []))
            existing_layers.update(paper.get("search_layers", []))
            existing["search_layers"] = sorted(existing_layers)
            existing_sources = set(existing.get("source_records", []))
            existing_sources.update(paper.get("source_records", []))
            existing["source_records"] = sorted(existing_sources)
            if paper.get("expected_paper_reason") and not existing.get("expected_paper_reason"):
                existing["expected_paper_reason"] = paper["expected_paper_reason"]
            if paper.get("api_discovery_reason"):
                existing_reason = existing.get("api_discovery_reason", "")
                pieces = [
                    part.strip()
                    for part in f"{existing_reason}; {paper['api_discovery_reason']}".split(";")
                    if part.strip()
                ]
                existing["api_discovery_reason"] = "; ".join(dict.fromkeys(pieces))
            continue
        merged.append(paper)
        for key in keys:
            index[key] = paper

    return merged


def conflicting_identity(existing: dict[str, Any], paper: dict[str, Any]) -> bool:
    existing_pmid = existing.get("pmid", "").strip()
    paper_pmid = paper.get("pmid", "").strip()
    if existing_pmid and paper_pmid and existing_pmid != paper_pmid:
        return True

    existing_doi = clean_doi(existing.get("doi", "")).lower()
    paper_doi = clean_doi(paper.get("doi", "")).lower()
    if existing_doi and paper_doi and existing_doi != paper_doi:
        return True

    return False


def dedupe_keys(paper: dict[str, Any]) -> list[str]:
    keys = []
    pmid = paper.get("pmid", "").strip()
    doi = paper.get("doi", "").strip().lower()
    title = normalize_title(paper.get("title", ""))
    if pmid:
        keys.append(f"pmid:{pmid}")
    if doi:
        keys.append(f"doi:{doi}")
    if title:
        keys.append(f"title:{title}")
    return keys


def is_verified(paper: dict[str, Any]) -> bool:
    return bool(
        paper.get("pmid")
        or paper.get("doi")
        or paper.get("url")
        or paper.get("openalex_id")
        or paper.get("semantic_scholar_url")
    )


def enrich_with_openalex(paper: dict[str, Any], email: str = "") -> None:
    doi = clean_doi(paper.get("doi", ""))
    pmid = paper.get("pmid", "").strip()
    work: dict[str, Any] | None = None

    if doi:
        encoded_doi = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
        url = f"{OPENALEX_WORKS_URL}/{encoded_doi}"
        params = {"mailto": email} if email else None
        response = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            work = response.json()
        elif response.status_code not in {404, 403}:
            response.raise_for_status()

    if work is None and pmid:
        params = {
            "filter": f"ids.pmid:https://pubmed.ncbi.nlm.nih.gov/{pmid}",
            "per-page": "1",
        }
        if email:
            params["mailto"] = email
        response = requests.get(
            OPENALEX_WORKS_URL,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        work = results[0] if results else None

    if work is None:
        return

    paper["openalex_id"] = work.get("id", "") or ""
    cited_by = work.get("cited_by_count")
    if isinstance(cited_by, int):
        paper["openalex_citations"] = cited_by
        paper["citation_count"] = cited_by
        paper["citation_source"] = "OpenAlex"

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    if source.get("display_name") and not paper.get("journal"):
        paper["journal"] = source["display_name"]
    if work.get("publication_year") and not paper.get("year"):
        paper["year"] = work["publication_year"]


def enrich_with_semantic_scholar(paper: dict[str, Any]) -> None:
    identifier = ""
    doi = clean_doi(paper.get("doi", ""))
    pmid = paper.get("pmid", "").strip()
    if doi:
        identifier = f"DOI:{doi}"
    elif pmid:
        identifier = f"PMID:{pmid}"
    if not identifier:
        return

    fields = "title,url,citationCount,influentialCitationCount,year,journal,externalIds"
    url = f"{SEMANTIC_SCHOLAR_URL}/{urllib.parse.quote(identifier, safe=':')}"
    response = requests.get(
        url,
        params={"fields": fields},
        headers=DEFAULT_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 404:
        return
    response.raise_for_status()
    payload = response.json()
    paper["semantic_scholar_url"] = payload.get("url", "") or ""
    citations = payload.get("citationCount")
    if isinstance(citations, int):
        paper["semantic_scholar_citations"] = citations
        if paper.get("citation_count") is None:
            paper["citation_count"] = citations
            paper["citation_source"] = "Semantic Scholar"


def apply_topic_penalties(
    paper: dict[str, Any],
    context: SearchContext,
) -> tuple[int, list[str]]:
    profile = topic_profile(context.topic)
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    penalty_total = 0
    penalty_notes: list[str] = []

    if profile:
        for entry in profile.get("penalize", []):
            terms = [str(t).lower() for t in entry.get("match_terms", []) if t]
            if not terms:
                continue
            if any(term in text for term in terms):
                score = int(entry.get("score", 0))
                penalty_total += score
                reason = str(entry.get("reason") or entry.get("name") or "topic-profile penalty")
                penalty_notes.append(f"{reason} ({score:+d})")

    if not (paper.get("abstract") or "").strip():
        penalty_total -= 5
        penalty_notes.append("Missing abstract (-5)")

    return penalty_total, penalty_notes


def must_include_boost(paper: dict[str, Any], context: SearchContext) -> tuple[int, list[str]]:
    profile = topic_profile(context.topic)
    if not profile:
        return 0, []
    concepts = [str(c).lower() for c in profile.get("must_include_concepts", []) if c]
    if not concepts:
        return 0, []
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    matched = [c for c in concepts if c in text]
    if not matched:
        return 0, []
    bonus = min(6, len(matched) * 2)
    return bonus, matched


def ranking_confidence_for(paper: dict[str, Any]) -> str:
    citation_known = paper.get("citation_count") is not None
    quartile_known = paper.get("quartile") not in (None, "", "quartile not verified")
    abstract_present = bool((paper.get("abstract") or "").strip())
    expected_seed = bool(paper.get("expected_paper_reason"))

    if expected_seed:
        return "high"
    if citation_known and quartile_known and abstract_present:
        return "high"
    if (citation_known or quartile_known) and abstract_present:
        return "moderate"
    return "low"


def reason_for_tier(paper: dict[str, Any]) -> str:
    tier = paper.get("tier", "")
    bits: list[str] = []

    expected_reason = paper.get("expected_paper_reason")
    if expected_reason:
        bits.append(f"seeded landmark — {expected_reason}")

    if paper.get("mandatory_review_candidate"):
        protection = paper.get("mandatory_review_reason", "landmark / review protection")
        if protection:
            bits.append(protection)

    design = paper.get("study_design", "")
    if design:
        bits.append(f"design: {design}")

    journal = paper.get("journal", "")
    if journal:
        bits.append(f"journal: {journal}")

    citation_count = paper.get("citation_count")
    if citation_count is not None:
        bits.append(f"{int(citation_count)} citations")

    total = paper.get("total_score")
    if total is not None:
        bits.append(f"final score {int(total)}")

    if paper.get("purpose_fit_reason"):
        bits.append(f"goal fit: {paper['purpose_fit_reason']}")

    penalty_notes = paper.get("penalty_notes", [])
    if penalty_notes:
        bits.append("penalties: " + "; ".join(penalty_notes))

    head = f"{tier} because " if tier else ""
    return head + "; ".join(bits) if bits else tier


def score_and_classify_paper(
    paper: dict[str, Any],
    context: SearchContext,
    quartile_overrides: dict[str, dict[str, str]],
) -> dict[str, Any]:
    paper = dict(paper)
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    text = f"{title} {abstract} {' '.join(paper.get('publication_types', []))}".lower()

    quartile_data = lookup_quartile(paper.get("journal", ""), quartile_overrides)
    paper["quartile"] = quartile_data.get("quartile", "quartile not verified")
    paper["quartile_source"] = quartile_data.get("source", "quartile not verified")

    topic_gate = classify_topic_match(title, abstract, context)
    raw_relevance_score, relevance_reason = score_relevance(text, context)
    if topic_gate["level"] == "direct":
        raw_relevance_score = max(raw_relevance_score, 36)
        relevance_reason += "; Rule 0 direct topic gate raised relevance floor"
    relevance_score = min(raw_relevance_score, topic_gate["relevance_cap"])
    design, design_score = classify_design(paper, context)
    journal_score = score_journal_quality(paper["quartile"], paper.get("journal", ""))
    citation_score, citation_note = score_citations(paper.get("citation_count"))
    recency_score = score_recency(paper.get("year"), context.current_year)
    purpose_score, purpose_reason = search_purpose_adjustment(paper, design, context)

    concept_bonus, matched_concepts = must_include_boost(paper, context)
    if concept_bonus:
        relevance_score = min(40, relevance_score + concept_bonus)
        relevance_reason += f"; +{concept_bonus} for must-include concepts: {', '.join(matched_concepts)}"

    paper["quartile"] = paper.get("quartile", "")
    penalty_score, penalty_notes = apply_topic_penalties(paper, context)

    base_total = (
        relevance_score
        + design_score
        + journal_score
        + citation_score
        + recency_score
        + purpose_score
    )
    total_score = max(0, base_total + penalty_score)

    paper["topic_match_gate"] = topic_gate["gate"]
    paper["topic_match_level"] = topic_gate["level"]
    paper["topic_match_reason"] = topic_gate["reason"]
    paper["topic_match_max_tier"] = topic_gate["max_tier_label"]
    paper["raw_relevance_score"] = raw_relevance_score
    paper["relevance_score"] = relevance_score
    paper["clinical_relevance_score"] = relevance_score
    paper["relevance_cap"] = topic_gate["relevance_cap"]
    paper["design_strength_score"] = design_score
    paper["study_design_score"] = design_score
    paper["journal_quality_score"] = journal_score
    paper["citation_score"] = citation_score
    paper["citation_strength_score"] = citation_score
    paper["recency_score"] = recency_score
    paper["purpose_fit_score"] = purpose_score
    paper["purpose_fit_reason"] = purpose_reason
    paper["search_mode"] = normalized_search_purpose(context.search_purpose)
    paper["search_purpose"] = paper["search_mode"]
    paper["penalty_score"] = penalty_score
    paper["penalty_notes"] = penalty_notes
    paper["base_score"] = base_total
    paper["total_score"] = total_score
    paper["final_score"] = total_score
    paper["study_design"] = design
    paper["citation_note"] = citation_note
    paper["citation_count_missing"] = paper.get("citation_count") is None
    paper["relevance_reason"] = relevance_reason
    paper["recent_high_quality_note"] = recent_high_quality_note(paper, design, citation_score, context)
    review_protection = major_review_protection(paper, design, context)
    paper["mandatory_review_candidate"] = review_protection["candidate"]
    paper["mandatory_review_reason"] = review_protection["reason"]
    paper["landmark_seed_match"] = bool(paper.get("expected_paper_reason"))
    paper["score_only_tier"] = assign_mode_tier(paper, context)
    paper["tier"], paper["tier_cap_reason"] = apply_tier_caps(
        paper,
        paper["score_only_tier"],
        topic_gate["max_tier_order"],
        context,
    )
    paper["normalized_title"] = normalize_title(paper.get("title", ""))
    paper["publication_type"] = ", ".join(paper.get("publication_types", []))
    paper["evidence_group"] = classify_evidence_group(paper, design)
    paper["knowledge_roles"] = classify_knowledge_roles(paper, design, context)
    paper["tags"] = build_tags(paper, design, context)
    paper["verification"] = verification_label(paper)
    paper["why_included"] = why_included(paper)
    paper["why_related"] = paper["why_included"]
    paper["relation_type"] = relation_type_for_paper(paper)
    paper["gap_suggested"] = suggest_paper_gap(paper, context)
    paper["evidence_family"] = evidence_family(paper)
    paper["reading_section"] = assign_reading_section(paper, context)
    paper["ranking_confidence"] = ranking_confidence_for(paper)
    paper["reason_for_tier"] = reason_for_tier(paper)
    paper["search_layers"] = ", ".join(paper.get("search_layers", []))
    paper["publication_types"] = ", ".join(paper.get("publication_types", []))
    return paper


def major_review_protection(
    paper: dict[str, Any],
    design: str,
    context: SearchContext,
) -> dict[str, Any]:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    title = paper.get("title", "").lower()
    journal = paper.get("journal", "").lower()
    search_layers = " ".join(paper.get("search_layers", [])).lower()
    direct_or_seed = paper.get("topic_match_level") == "direct" or bool(
        paper.get("expected_paper_reason")
    )
    review_like = design in {
        "Guideline / consensus / society statement",
        "Systematic review / meta-analysis",
        "Narrative review",
        "Landmark physiological review",
    }
    major_journal = any(term in journal for term in MAJOR_JOURNAL_TERMS)
    title_signal = has_any(
        title,
        [
            "guideline",
            "statement",
            "consensus",
            "review",
            "update",
            "practical guide",
            "comprehensive",
            "state of the art",
            "current concepts",
            "seminar",
            "primer",
        ],
    )
    strong_review_signal = has_any(
        title,
        [
            "practical guide",
            "comprehensive",
            "state of the art",
            "current concepts",
            "seminar",
            "primer",
            "update",
            "advances",
            "current management",
            "changing face",
        ],
    )
    discovery_signal = any(
        layer in search_layers
        for layer in [
            "review/guideline",
            "landmark/classic",
            "expected landmark seed",
            "api supervisor",
        ]
    )
    landmark_discovery = any(
        layer in search_layers for layer in ["landmark/classic", "expected landmark seed"]
    )
    api_discovery = "api supervisor" in search_layers
    recent_comprehensive = (
        bool(paper.get("year") and context.current_year - paper["year"] <= 3)
        and strong_review_signal
    )
    recent_api_review = (
        api_discovery
        and bool(paper.get("year") and context.current_year - paper["year"] <= 3)
        and review_like
        and direct_or_seed
    )

    reasons = []
    if paper.get("expected_paper_reason"):
        reasons.append(paper["expected_paper_reason"])
    if major_journal:
        reasons.append("published in a major journal")
    if design == "Guideline / consensus / society statement":
        reasons.append("society/scientific statement or guideline-like paper")
    if recent_comprehensive:
        reasons.append("recent comprehensive/update review")
    if discovery_signal:
        reasons.append("found in mandatory review/guideline/landmark discovery layer")
    if api_discovery:
        reasons.append("found by API discovery supervisor")
    if title_signal and direct_or_seed:
        reasons.append("clearly focused review/update title")

    high_confidence_review = (
        design in {
            "Guideline / consensus / society statement",
            "Systematic review / meta-analysis",
            "Landmark physiological review",
        }
        or bool(paper.get("expected_paper_reason"))
        or major_journal
        or recent_comprehensive
        or recent_api_review
        or (landmark_discovery and strong_review_signal)
    )
    candidate = bool(review_like and direct_or_seed and reasons and high_confidence_review)
    return {
        "candidate": candidate,
        "reason": (
            "Mandatory review/landmark candidate - needs citation/quartile enrichment: "
            + "; ".join(dict.fromkeys(reasons))
            if candidate
            else ""
        ),
    }


def classify_topic_match(title: str, abstract: str, context: SearchContext) -> dict[str, Any]:
    title_text = normalize_space(title).lower()
    abstract_text = normalize_space(abstract).lower()
    text = f"{title_text} {abstract_text}"
    profile = topic_profile(context.topic)

    if profile:
        return classify_profile_topic_match(title_text, abstract_text, text, profile)

    topic_phrase = normalize_space(context.topic).lower()
    topic_terms = keywords(context.topic)
    title_coverage = coverage(topic_terms, title_text)
    text_coverage = coverage(topic_terms, text)

    if topic_phrase and topic_phrase in title_text:
        return topic_gate(
            "Direct topic match",
            "direct",
            40,
            1,
            "title contains the requested topic phrase",
        )
    if topic_terms and title_coverage >= 0.70:
        return topic_gate(
            "Direct topic match",
            "direct",
            40,
            1,
            "title contains most core topic terms",
        )
    if topic_phrase and topic_phrase in abstract_text:
        return topic_gate(
            "Abstract-only topic match",
            "abstract_only",
            30,
            3,
            "abstract contains the requested topic phrase but title is broader",
        )
    if topic_terms and text_coverage >= 0.70:
        return topic_gate(
            "Partial disease-family match",
            "partial",
            24,
            3,
            "abstract contains most topic terms, but title is not a clear match",
        )
    if topic_terms and text_coverage >= 0.35:
        return topic_gate(
            "General background match",
            "background",
            14,
            4,
            "only a minority of topic terms are present",
        )
    return topic_gate(
        "Noise / manual review",
        "noise",
        8,
        5,
        "title/abstract do not clearly match the requested topic",
    )


def classify_profile_topic_match(
    title_text: str,
    abstract_text: str,
    text: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    label = profile.get("display_name") or profile.get("key") or "topic"
    direct_phrases = profile.get("direct_phrases", [])
    direct_acronyms = profile.get("direct_acronyms", [])
    acronym_context = profile.get("acronym_context", [])
    wrong_terms = profile.get("wrong_terms", [])
    family_terms = profile.get("family_terms", [])
    background_terms = profile.get("background_terms", [])

    for phrase in direct_phrases:
        if phrase and phrase in title_text:
            return topic_gate(
                "Direct topic match",
                "direct",
                40,
                1,
                f"title contains core {label} term: {phrase}",
            )
    for acronym in direct_acronyms:
        if acronym and has_contextual_acronym(title_text, acronym, acronym_context):
            return topic_gate(
                "Direct topic match",
                "direct",
                40,
                1,
                f"title contains {acronym.upper()} with topical context",
            )
    for phrase in direct_phrases:
        if phrase and phrase in abstract_text:
            return topic_gate(
                "Abstract-only topic match",
                "abstract_only",
                30,
                3,
                f"abstract contains core {label} term but title is broader: {phrase}",
            )
    for acronym in direct_acronyms:
        if acronym and has_contextual_acronym(text, acronym, acronym_context):
            return topic_gate(
                "Abstract-only topic match",
                "abstract_only",
                30,
                3,
                f"abstract contains {acronym.upper()} with topical context but no core title phrase",
            )

    for term in wrong_terms:
        if term and term in text:
            return topic_gate(
                "Noise / manual review",
                "noise",
                8,
                5,
                f"off-topic exclusion signal: {term}",
            )
    for term in family_terms:
        if term and term in text:
            return topic_gate(
                "Partial disease-family match",
                "partial",
                24,
                3,
                f"family term without core {label} match: {term}",
            )
    for term in background_terms:
        if term and term in text:
            return topic_gate(
                "General background match",
                "background",
                14,
                4,
                f"background term without core {label} match: {term}",
            )
    return topic_gate(
        "Noise / manual review",
        "noise",
        8,
        5,
        f"no core {label} term found in title/abstract",
    )


def topic_gate(
    gate: str,
    level: str,
    relevance_cap: int,
    max_tier_order: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "level": level,
        "relevance_cap": relevance_cap,
        "max_tier_order": max_tier_order,
        "max_tier_label": TIER_BY_ORDER[max_tier_order],
        "reason": reason,
    }


_PRIMED_PROFILES: dict[str, dict[str, Any]] = {}


def _normalize_topic_key(topic: str) -> str:
    return re.sub(r"\s+", " ", topic or "").strip().lower()


def topic_profile(topic: str) -> dict[str, Any] | None:
    """Return the topic profile for `topic`.

    Hand-authored JSON profiles in topics/*.json take precedence over LLM-
    generated primers. The primed profile is a drop-in replacement with the
    same shape (triggers, expected_papers, must_include_concepts, penalize,
    plus an extra query_expansion_terms field).
    """
    topic_text = normalize_space(topic).lower()
    if not topic_text:
        return None
    for profile in load_topic_profiles():
        triggers = profile.get("triggers", [])
        if any(trigger and trigger in topic_text for trigger in triggers):
            return profile
    return _PRIMED_PROFILES.get(_normalize_topic_key(topic))


def register_primer_if_needed(
    topic: str,
    gemini_api_key: str,
    email: str = "",
    api_key: str = "",
) -> str:
    """Ensure a profile exists for `topic`. Returns one of:

    - 'profile'      — hand-authored topics/*.json profile matched
    - 'cached'       — primer already cached in this process
    - 'generated'    — primer just generated and registered
    - 'unavailable'  — no profile, no key, or LLM call failed
    """
    if not topic.strip():
        return "unavailable"

    topic_text = normalize_space(topic).lower()
    for profile in load_topic_profiles():
        triggers = profile.get("triggers", [])
        if any(trigger and trigger in topic_text for trigger in triggers):
            return "profile"

    if not gemini_api_key:
        return "unavailable"

    cache_key = _normalize_topic_key(topic)
    if cache_key in _PRIMED_PROFILES:
        return "cached"

    primer = prime_topic(topic, gemini_api_key, email=email, api_key=api_key)
    if primer is None:
        return "unavailable"
    _PRIMED_PROFILES[cache_key] = primer.to_profile_dict()
    return "generated"


def has_contextual_acronym(text: str, acronym: str, context_terms: list[str]) -> bool:
    if not re.search(rf"\b{re.escape(acronym)}\b", text, flags=re.IGNORECASE):
        return False
    return any(term in text for term in context_terms)


def score_relevance(text: str, context: SearchContext) -> tuple[int, str]:
    topic_terms = keywords(context.topic)
    direct_phrase = normalize_space(context.topic).lower()
    direct_score = 15 if direct_phrase and direct_phrase in text else round(15 * coverage(topic_terms, text))

    icu_score = 10 if any(term in text for term in ICU_TERMS) else 0

    population_terms = keywords(context.population) or topic_terms
    population_score = round(5 * coverage(population_terms, text)) if population_terms else 0

    outcome_terms = keywords(context.outcome)
    if outcome_terms:
        outcome_score = round(5 * coverage(outcome_terms, text))
        outcome_reason = "outcome terms scored"
    else:
        outcome_score = 3
        outcome_reason = "outcome not specified; neutral score"

    practical_terms = [
        "mortality",
        "ventilator-free",
        "length of stay",
        "protocol",
        "guideline",
        "randomized",
        "trial",
        "implementation",
        "practice",
    ]
    practical_score = 5 if any(term in text for term in practical_terms) else 3 if direct_score >= 8 else 0

    total = min(40, direct_score + icu_score + population_score + outcome_score + practical_score)
    reason = (
        f"topic {direct_score}/15, ICU {icu_score}/10, population {population_score}/5, "
        f"outcome {outcome_score}/5 ({outcome_reason}), practical {practical_score}/5"
    )
    return total, reason


def classify_design(paper: dict[str, Any], context: SearchContext) -> tuple[str, int]:
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    publication_types = [
        item.lower()
        for item in paper.get("publication_types", [])
        if isinstance(item, str)
    ]
    pub_type_text = " ".join(publication_types)
    title_and_types = f"{title} {pub_type_text}"
    text = f"{title} {abstract} {pub_type_text}"

    has_review_type = any(pub_type == "review" for pub_type in publication_types)

    citation_count = paper.get("citation_count") or 0
    year = paper.get("year") or 0
    journal = paper.get("journal", "").lower()
    is_major_journal = any(term in journal for term in MAJOR_JOURNAL_TERMS)
    landmark_age = year and year <= context.current_year - 5

    if has_any(pub_type_text, ["practice guideline", "guideline"]) or has_any(
        title, ["guideline", "consensus statement", "society statement", "scientific statement"]
    ):
        design, score = "Guideline / consensus / society statement", 23
    elif has_any(pub_type_text, ["systematic review", "meta-analysis"]) or has_any(
        title, ["systematic review", "meta-analysis", "meta analysis"]
    ):
        design, score = "Systematic review / meta-analysis", 20
    elif has_any(pub_type_text, ["randomized controlled trial"]) or (
        not has_review_type and looks_like_original_randomized_trial(title, abstract)
    ):
        if is_major_journal and citation_count >= 200:
            design, score = "Landmark randomized trial", 25
        else:
            design, score = "Randomized controlled trial", 18
    elif has_review_type:
        if is_major_journal and citation_count >= 200 and landmark_age:
            design, score = "Landmark physiological review", 22
        else:
            design, score = "Narrative review", 8
    elif has_any(title_and_types, ["diagnostic accuracy"]) or has_any(
        title, ["sensitivity", "specificity", "receiver operating"]
    ):
        design, score = "Diagnostic accuracy study", 15
    elif has_any(title_and_types, ["prospective cohort", "multicentre prospective", "multicenter prospective"]):
        design, score = "Large multicentre prospective cohort", 15
    elif has_any(title_and_types, ["retrospective", "database", "registry", "observational cohort"]):
        design, score = "Large retrospective / database study", 12
    elif has_any(title_and_types, ["cohort", "observational"]):
        design, score = "Single-centre observational study", 9
    elif has_any(text, ["in rats", "in mice", "in rabbits", "isolated lung", "ex vivo", "knockout mice"]):
        design, score = "Experimental / animal / basic science", 10
    elif has_any(text, ["transcriptomic", "proteomic", "rna-seq", "gene expression"]):
        design, score = "Molecular / mechanistic study", 8
    elif has_any(pub_type_text, ["editorial", "comment"]) or has_any(title, ["editorial", "commentary"]):
        design, score = "Editorial / commentary", 5
    elif has_any(pub_type_text, ["case reports"]) or has_any(title, ["case report", "case series"]):
        design, score = "Case series / case report", 4
    else:
        design, score = "Unclear", 3

    qtype = context.question_type.lower()
    if "intervention" in qtype and design in {
        "Randomized controlled trial",
        "Systematic review / meta-analysis",
        "Guideline / consensus / society statement",
    }:
        score += 1
    elif "diagnosis" in qtype and design in {
        "Diagnostic accuracy study",
        "Systematic review / meta-analysis",
        "Guideline / consensus / society statement",
    }:
        score += 2
    elif ("prognosis" in qtype or "prediction" in qtype) and design in {
        "Large multicentre prospective cohort",
        "Large retrospective / database study",
    }:
        score += 2
    elif ("implementation" in qtype or "cost" in qtype) and has_any(
        text, ["implementation", "feasibility", "cost", "audit", "quality improvement"]
    ):
        score += 2

    return design, min(25, max(0, score))


def looks_like_original_randomized_trial(title: str, abstract: str) -> bool:
    title_has_trial = re.search(
        r"\b(randomi[sz]ed|randomly assigned)\b.{0,80}\b(trial|study)\b",
        title,
    )
    if title_has_trial:
        return True
    abstract_signal = re.search(
        r"\b(we conducted|we performed|participants were|patients were)\b.{0,160}"
        r"\b(randomi[sz]ed|randomly assigned)\b.{0,80}\b(trial|study)\b",
        abstract,
    )
    return bool(abstract_signal)


def score_journal_quality(quartile: str, journal: str = "") -> int:
    base = JOURNAL_SCORE.get(quartile, 0)
    journal_lower = (journal or "").lower()
    if any(term in journal_lower for term in MAJOR_JOURNAL_TERMS):
        base += JOURNAL_MAJOR_BONUS
    return min(15, base)


def score_citations(citation_count: int | None) -> tuple[int, str]:
    if citation_count is None:
        return 0, "citation count unavailable"
    if citation_count > 1000:
        return 10, "more than 1000 citations"
    if citation_count >= 500:
        return 8, "500-999 citations"
    if citation_count >= 100:
        return 6, "100-499 citations"
    if citation_count >= 25:
        return 4, "25-99 citations"
    if citation_count >= 5:
        return 2, "5-24 citations"
    if citation_count >= 1:
        return 1, "less than 5 citations"
    return 0, "less than 5 citations"


def score_recency(year: int | None, current_year: int) -> int:
    if not year:
        return 0
    age = current_year - year
    if age <= 2:
        return 10
    if age <= 5:
        return 8
    if age <= 10:
        return 5
    return 2


def recent_high_quality_note(
    paper: dict[str, Any],
    design: str,
    citation_score: int,
    context: SearchContext,
) -> str:
    year = paper.get("year")
    if not year or context.current_year - year > 2:
        return ""
    major_design = design in {
        "Randomized controlled trial",
        "Systematic review / meta-analysis",
        "Guideline / consensus / society statement",
        "Large multicentre prospective cohort",
        "Diagnostic accuracy study",
    }
    if major_design and citation_score < 4:
        return "Recent high-quality paper - citation count not yet mature."
    return ""


def search_purpose_adjustment(
    paper: dict[str, Any],
    design: str,
    context: SearchContext,
) -> tuple[int, str]:
    purpose = normalized_search_purpose(context.search_purpose)
    title = paper.get("title", "").lower()
    text = f"{title} {paper.get('abstract', '')}".lower()
    year = paper.get("year") or 0
    recent = bool(year and context.current_year - year <= 3)
    direct = paper.get("topic_match_level") in {"direct", "abstract_only"}
    score = 0
    reasons: list[str] = []

    rare_signal = has_any(text, RARE_CASE_TERMS)

    if purpose == SEARCH_PURPOSE_KNOWLEDGE:
        if design in {"Guideline / consensus / society statement", "Systematic review / meta-analysis"}:
            score += 10
            reasons.append("learning mode prioritizes guidelines and systematic reviews")
        elif design in {"Narrative review", "Landmark physiological review"}:
            score += 8
            reasons.append("learning mode prioritizes readable review articles")
        elif design in {"Landmark randomized trial", "Randomized controlled trial"}:
            score += 4
            reasons.append("learning mode keeps major trials after review sources")
        if recent and has_any(title, ["review", "update", "practical", "state of the art"]):
            score += 3
            reasons.append("recent practical update")
        if design in {"Case series / case report", "Experimental / animal / basic science"}:
            score -= 3
            reasons.append("learning mode de-emphasizes narrow low-level evidence")

    elif purpose == SEARCH_PURPOSE_DEEP:
        if direct:
            score += 6
            reasons.append("deep search keeps directly matched records high")
        if design in {
            "Landmark randomized trial",
            "Randomized controlled trial",
            "Diagnostic accuracy study",
            "Large multicentre prospective cohort",
            "Large retrospective / database study",
            "Single-centre observational study",
            "Case series / case report",
            "Experimental / animal / basic science",
            "Molecular / mechanistic study",
        }:
            score += 5
            reasons.append("deep search keeps all relevant study designs")
        elif design in {"Systematic review / meta-analysis", "Guideline / consensus / society statement"}:
            score += 3
            reasons.append("deep search keeps reviews/guidelines as cross-reference sources")

    elif purpose == SEARCH_PURPOSE_RARE:
        if design == "Case series / case report":
            score += 12
            reasons.append("rare/case mode prioritizes case reports and case series")
        elif rare_signal:
            score += 10
            reasons.append("rare/case mode prioritizes rare presentations or complications")
        elif design in {
            "Large retrospective / database study",
            "Single-centre observational study",
            "Diagnostic accuracy study",
        }:
            score += 4
            reasons.append("descriptive study may contain rare-event data")
        elif design in {
            "Guideline / consensus / society statement",
            "Systematic review / meta-analysis",
            "Narrative review",
            "Landmark randomized trial",
            "Randomized controlled trial",
        }:
            score -= 4
            reasons.append("rare/case mode down-ranks broad guidance and common-topic evidence")
        if direct:
            score += 3
            reasons.append("direct rare-topic match")

    else:
        if design in {
            "Systematic review / meta-analysis",
            "Guideline / consensus / society statement",
            "Landmark randomized trial",
            "Randomized controlled trial",
        }:
            score += 8
            reasons.append("research-gap mode anchors on high-level evidence")
        elif design in {
            "Large multicentre prospective cohort",
            "Large retrospective / database study",
            "Diagnostic accuracy study",
        }:
            score += 5
            reasons.append("research-gap mode values strong primary evidence")
        if recent:
            score += 3
            reasons.append("recent evidence useful for current gap finding")
        if has_any(text, GAP_TERMS + ["limitation", "limitations", "future research", "further studies"]):
            score += 5
            reasons.append("explicit gap/uncertainty language")
        if has_any(text, LMIC_TERMS + ["implementation", "feasibility", "external validation"]):
            score += 3
            reasons.append("implementation, validation, or LMIC relevance")
        if design in {"Narrative review", "Editorial / commentary", "Case series / case report"}:
            score -= 4
            reasons.append("research mode down-ranks narrative, opinion, and small case evidence")

    return max(-8, min(15, score)), "; ".join(dict.fromkeys(reasons))


def relation_type_for_paper(paper: dict[str, Any]) -> str:
    level = paper.get("topic_match_level", "")
    mapping = {
        "direct": "Directly related",
        "abstract_only": "Related in abstract",
        "partial": "Partially related",
        "background": "Background / indirect",
        "noise": "Weak or uncertain relation",
    }
    return mapping.get(level, paper.get("topic_match_gate", "Relation not classified"))


def assign_tier(paper: dict[str, Any]) -> str:
    if paper.get("expected_paper_reason"):
        return "Tier 1: Must-read"

    total = paper.get("total_score", 0)
    relevance = paper.get("relevance_score", 0)

    if total >= 80 and relevance >= 24:
        return "Tier 1: Must-read"
    if total >= 60 and relevance >= 18:
        return "Tier 2: Useful supporting"
    if total >= 40 and relevance >= 12:
        return "Tier 3: Background"
    return "Tier 4: Low priority"


def assign_mode_tier(paper: dict[str, Any], context: SearchContext) -> str:
    if paper.get("expected_paper_reason"):
        return "Tier 1: Must-read"

    purpose = normalized_search_purpose(context.search_purpose)
    design = paper.get("study_design", "")
    total = paper.get("total_score", 0)
    relevance = paper.get("relevance_score", 0)
    topic_level = paper.get("topic_match_level", "")
    direct = topic_level in {"direct", "abstract_only"}
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    rare_signal = has_any(text, RARE_CASE_TERMS)
    major_review = bool(paper.get("mandatory_review_candidate"))

    if purpose == SEARCH_PURPOSE_KNOWLEDGE:
        if direct and (
            design in {
                "Guideline / consensus / society statement",
                "Systematic review / meta-analysis",
                "Narrative review",
                "Landmark physiological review",
            }
            or major_review
        ) and total >= 55:
            return "Tier 1: Must-read"
        if direct and design in {"Landmark randomized trial", "Randomized controlled trial"} and total >= 60:
            return "Tier 2: Useful supporting"
        if direct and total >= 45:
            return "Tier 2: Useful supporting"
        if topic_level in {"direct", "abstract_only", "partial", "background"}:
            return "Tier 3: Background"
        return "Tier 4: Low priority"

    if purpose == SEARCH_PURPOSE_RESEARCH:
        if (
            direct
            and design in {
                "Landmark randomized trial",
                "Randomized controlled trial",
                "Large multicentre prospective cohort",
                "Large retrospective / database study",
                "Diagnostic accuracy study",
                "Systematic review / meta-analysis",
            }
            and total >= 55
        ):
            return "Tier 1: Must-read"
        if design in {"Narrative review", "Landmark physiological review"}:
            return "Tier 2: Useful supporting" if major_review and direct else "Tier 3: Background"
        if direct and total >= 45:
            return "Tier 2: Useful supporting"
        if topic_level in {"direct", "abstract_only", "partial", "background"}:
            return "Tier 3: Background"
        return "Tier 4: Low priority"

    if purpose == SEARCH_PURPOSE_DEEP:
        if direct and (paper.get("expected_paper_reason") or major_review or total >= 65):
            return "Tier 1: Must-read"
        if topic_level in {"direct", "abstract_only"} and relevance >= 18:
            return "Tier 2: Useful supporting"
        if topic_level in {"partial", "background"}:
            return "Tier 3: Background"
        return "Tier 4: Low priority"

    if purpose == SEARCH_PURPOSE_RARE:
        if direct and (design == "Case series / case report" or rare_signal):
            return "Tier 1: Must-read"
        if topic_level in {"direct", "abstract_only", "partial"} and (
            design == "Case series / case report" or rare_signal
        ):
            return "Tier 2: Useful supporting"
        if topic_level in {"direct", "abstract_only", "partial", "background"}:
            return "Tier 3: Background"
        return "Tier 4: Low priority"

    return assign_tier(paper)


def apply_tier_caps(
    paper: dict[str, Any],
    score_only_tier: str,
    topic_max_tier_order: int,
    context: SearchContext,
) -> tuple[str, str]:
    if paper.get("expected_paper_reason"):
        return "Tier 1: Must-read", "landmark seed — promoted to Tier 1 regardless of gate/quality caps"

    purpose = normalized_search_purpose(context.search_purpose)
    cap_order = topic_max_tier_order
    topic_level = paper.get("topic_match_level", "")
    if purpose == SEARCH_PURPOSE_DEEP and topic_level in {"partial", "background"}:
        cap_order = min(cap_order, 3)
    if purpose == SEARCH_PURPOSE_RARE and topic_level == "partial":
        cap_order = min(cap_order, 2)
    elif purpose == SEARCH_PURPOSE_RARE and topic_level == "background":
        cap_order = min(cap_order, 3)
    cap_reasons = []
    if cap_order > 1:
        cap_reasons.append(
            f"Rule 0 topic gate caps this paper at {TIER_BY_ORDER[cap_order]}"
        )

    if purpose not in {SEARCH_PURPOSE_DEEP, SEARCH_PURPOSE_RARE}:
        quality_cap_order, quality_reason = quality_data_cap(paper)
        if quality_cap_order > cap_order:
            cap_order = quality_cap_order
        if quality_reason:
            cap_reasons.append(quality_reason)

    mode_cap_order, mode_reason = mode_specific_tier_cap(paper, context)
    if mode_cap_order > cap_order:
        cap_order = mode_cap_order
    if mode_reason:
        cap_reasons.append(mode_reason)

    score_order = TIER_ORDER.get(score_only_tier, 5)
    final_order = max(score_order, cap_order)
    final_tier = TIER_BY_ORDER[final_order]
    if final_tier != score_only_tier and not cap_reasons:
        cap_reasons.append(f"tier capped from {score_only_tier} to {final_tier}")
    return final_tier, "; ".join(cap_reasons)


def mode_specific_tier_cap(paper: dict[str, Any], context: SearchContext) -> tuple[int, str]:
    purpose = normalized_search_purpose(context.search_purpose)
    design = paper.get("study_design", "")
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    rare_signal = has_any(text, RARE_CASE_TERMS)

    if purpose == SEARCH_PURPOSE_RESEARCH and design in {
        "Narrative review",
        "Landmark physiological review",
    } and not (paper.get("mandatory_review_candidate") or paper.get("expected_paper_reason")):
        return 3, "research mode caps narrative/background reviews at Tier 3 unless landmark or gap-defining"

    if purpose == SEARCH_PURPOSE_KNOWLEDGE and design in {
        "Case series / case report",
        "Experimental / animal / basic science",
        "Molecular / mechanistic study",
    } and not paper.get("expected_paper_reason"):
        return 3, "knowledge mode keeps narrow low-level evidence as background"

    if purpose == SEARCH_PURPOSE_RARE and design in {
        "Guideline / consensus / society statement",
        "Systematic review / meta-analysis",
        "Narrative review",
        "Landmark randomized trial",
        "Randomized controlled trial",
    } and not rare_signal:
        return 3, "rare/case mode caps broad evidence unless it contains rare-event data"

    return 1, ""


def quality_data_cap(paper: dict[str, Any]) -> tuple[int, str]:
    quartile_unknown = paper.get("quartile") == "quartile not verified"
    citation_unknown = paper.get("citation_count") is None
    if not (quartile_unknown and citation_unknown):
        return 1, ""

    major_design = paper.get("study_design") in {
        "Guideline / consensus / society statement",
        "Systematic review / meta-analysis",
        "Randomized controlled trial",
    }
    direct_topic = paper.get("topic_match_level") == "direct"
    if major_design and direct_topic:
        return (
            2,
            "quartile and citations unavailable; direct major evidence capped at Tier 2 until quality metadata are verified",
        )
    if paper.get("mandatory_review_candidate"):
        return (
            2,
            paper.get("mandatory_review_reason", "")
            + "; quartile and citations unavailable, so this is protected for display but capped at Tier 2 until enrichment",
        )
    return (
        3,
        "quartile and citations unavailable; capped at Tier 3 until manual quality review",
    )


def assign_reading_section(paper: dict[str, Any], context: SearchContext) -> str:
    purpose = normalized_search_purpose(context.search_purpose)
    design = paper.get("study_design", "")
    topic_level = paper.get("topic_match_level", "")
    tier_order = TIER_ORDER.get(paper.get("tier", ""), 5)
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    rare_signal = has_any(text, RARE_CASE_TERMS)
    recent = bool(paper.get("year") and context.current_year - paper["year"] <= 3)

    if purpose == SEARCH_PURPOSE_KNOWLEDGE:
        if design == "Landmark physiological review":
            return "Foundational concepts"
        if design == "Narrative review" or (
            design == "Systematic review / meta-analysis" and tier_order <= 2
        ):
            return "Best review articles"
        if design == "Guideline / consensus / society statement":
            return "Guidelines and consensus"
        if design in {"Experimental / animal / basic science", "Molecular / mechanistic study"}:
            return "Foundational concepts"
        if design in {"Landmark randomized trial", "Randomized controlled trial"} and tier_order <= 2:
            return "Landmark clinical papers"
        if recent:
            return "Recent updates"
        return "Background papers"

    if purpose == SEARCH_PURPOSE_RESEARCH:
        if "gap" in str(paper.get("search_layers", "")).lower() or has_any(text, GAP_TERMS):
            return "Research gaps"
        if design in {"Landmark randomized trial", "Randomized controlled trial"}:
            return "Randomized controlled trials"
        if design in {
            "Large multicentre prospective cohort",
            "Large retrospective / database study",
            "Single-centre observational study",
        }:
            return "Observational/cohort studies"
        if design == "Systematic review / meta-analysis":
            return "Systematic reviews/meta-analyses"
        if design in {"Diagnostic accuracy study"} or has_any(text, ["outcome", "endpoint", "definition", "predict", "prognostic", "diagnostic"]):
            return "Methods/outcome-defining papers"
        if design in {"Narrative review", "Guideline / consensus / society statement", "Landmark physiological review"}:
            return "Background reviews"
        return "Key original research papers"

    if purpose == SEARCH_PURPOSE_DEEP:
        if paper.get("mandatory_review_candidate") or paper.get("expected_paper_reason") or tier_order == 1:
            return "Landmark/core papers"
        if design in {
            "Narrative review",
            "Landmark physiological review",
            "Systematic review / meta-analysis",
            "Guideline / consensus / society statement",
        }:
            return "Reviews and meta-analyses"
        if design in {"Landmark randomized trial", "Randomized controlled trial"}:
            return "Trials"
        if design in {
            "Large multicentre prospective cohort",
            "Large retrospective / database study",
            "Single-centre observational study",
            "Diagnostic accuracy study",
        }:
            return "Observational studies"
        if design in {"Experimental / animal / basic science", "Molecular / mechanistic study"}:
            return "Mechanistic/basic science papers"
        if has_any(text, ["pediatric", "paediatric", "pregnancy", "pregnant", "perioperative", "postoperative"]):
            return "Special populations"
        if design == "Case series / case report":
            return "Case reports/case series"
        return "Low-priority/background papers"

    if purpose == SEARCH_PURPOSE_RARE:
        if design == "Case series / case report" and topic_level in {"direct", "abstract_only"}:
            return "Closest matching case reports"
        if design == "Case series / case report":
            return "Case series"
        if has_any(text, ["complication", "complications", "adverse event", "adverse drug", "device-related"]):
            return "Rare complications"
        if has_any(text, ["association", "associated", "unusual", "uncommon", "atypical", "rare"]):
            return "Rare associations"
        if has_any(text, ["diagnostic", "imaging", "laboratory", "radiologic", "radiological"]):
            return "Unusual diagnostic findings"
        if tier_order >= 4 or not rare_signal:
            return "Tier 4 / weak but related papers"
        return "Background references"

    if paper.get("mandatory_review_candidate") or paper.get("expected_paper_reason"):
        return "Core reading pack"
    if topic_level in {"noise"}:
        return "Low-priority / indirect papers"
    if topic_level in {"direct", "abstract_only"} and tier_order <= 3:
        return "Core reading pack"
    if topic_level in {"direct", "abstract_only", "partial"}:
        return "Extended evidence base"
    return "Low-priority / indirect papers"


def reading_section_order(search_mode: str) -> dict[str, int]:
    sections = {
        SEARCH_PURPOSE_KNOWLEDGE: [
            "Best review articles",
            "Guidelines and consensus",
            "Foundational concepts",
            "Landmark clinical papers",
            "Recent updates",
            "Background papers",
        ],
        SEARCH_PURPOSE_RESEARCH: [
            "Key original research papers",
            "Randomized controlled trials",
            "Observational/cohort studies",
            "Systematic reviews/meta-analyses",
            "Research gaps",
            "Methods/outcome-defining papers",
            "Background reviews",
        ],
        SEARCH_PURPOSE_DEEP: [
            "Landmark/core papers",
            "Reviews and meta-analyses",
            "Trials",
            "Observational studies",
            "Mechanistic/basic science papers",
            "Special populations",
            "Case reports/case series",
            "Low-priority/background papers",
        ],
        SEARCH_PURPOSE_RARE: [
            "Closest matching case reports",
            "Case series",
            "Rare complications",
            "Rare associations",
            "Unusual diagnostic findings",
            "Background references",
            "Tier 4 / weak but related papers",
        ],
    }.get(normalized_search_purpose(search_mode), [])
    fallback = ["Core reading pack", "Extended evidence base", "Low-priority / indirect papers"]
    return {section: index for index, section in enumerate(sections or fallback)}


def classify_evidence_group(paper: dict[str, Any], design: str) -> str:
    has_gap_layer = any("gap" in layer.lower() for layer in paper.get("search_layers", []))
    if design in {"Randomized controlled trial", "Large multicentre prospective cohort", "Diagnostic accuracy study"}:
        return "Core evidence"
    if design in {"Large retrospective / database study", "Single-centre observational study"}:
        return "Core evidence"
    if design == "Systematic review / meta-analysis":
        return "Evidence synthesis"
    if design == "Guideline / consensus / society statement":
        return "Practice guidance"
    if has_gap_layer:
        return "Recent update" if paper.get("recent_high_quality_note") else "Local / LMIC evidence"
    return "Secondary evidence"


def classify_knowledge_roles(
    paper: dict[str, Any],
    design: str,
    context: SearchContext,
) -> str:
    roles = []
    citations = paper.get("citation_count")
    year = paper.get("year")
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

    if design in {
        "Guideline / consensus / society statement",
        "Systematic review / meta-analysis",
        "Randomized controlled trial",
    } and paper.get("relevance_score", 0) >= 25:
        roles.append("Established knowledge")
    if citations is not None and citations >= 500:
        roles.append("Landmark evidence")
    if year and context.current_year - year <= 2:
        roles.append("Recent updates")
    if has_any(text, ["controversy", "controversial", "conflicting", "inconsistent", "uncertain"]):
        roles.append("Unresolved controversy")
    if "Gap" in str(paper.get("search_layers", "")) or has_any(text, GAP_TERMS):
        roles.append("Research gaps")
        roles.append("Possible future study ideas")
    return ", ".join(dict.fromkeys(roles)) if roles else "Background knowledge"


def build_tags(paper: dict[str, Any], design: str, context: SearchContext) -> str:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    tags = []
    topic_tag = " ".join(keywords(context.topic)[:3]) or context.topic.strip().lower()
    if topic_tag:
        tags.append(topic_tag)
    tags.append(context.question_type.lower())
    tags.append(design.lower())

    if paper.get("recent_high_quality_note"):
        tags.append("recent update")
    if "randomized" in design.lower():
        tags.append("major RCT")
    if "systematic" in design.lower() or "meta-analysis" in design.lower():
        tags.append("systematic review")
    if "guideline" in design.lower() or "consensus" in design.lower():
        tags.append("guideline/consensus")
    if has_any(text, ICU_TERMS):
        tags.append("direct ICU evidence")
    else:
        tags.append("non-ICU but useful")
    if has_any(text, LMIC_TERMS):
        tags.append("India/LMIC evidence")
    else:
        tags.append("global")
    if has_any(text, ["controversy", "conflicting", "uncertain"]):
        tags.append("controversial")
    elif paper.get("tier") == "Tier 4: Low priority":
        tags.append("insufficient evidence")
    else:
        tags.append("evolving evidence")
    tags.append("manuscript introduction")
    if paper.get("tier") in {"Tier 1: Must-read", "Tier 2: Useful supporting"}:
        tags.append("protocol design")
    return " | ".join(dict.fromkeys(tags))


def suggest_paper_gap(paper: dict[str, Any], context: SearchContext) -> str:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    design = paper.get("study_design", "").lower()
    suggestions = []
    if not has_any(text, ICU_TERMS):
        suggestions.append("ICU-specific validation needed")
    if not has_any(text, LMIC_TERMS):
        suggestions.append("Indian/LMIC applicability not established")
    if has_any(design, ["observational", "retrospective", "database", "case"]):
        suggestions.append("prospective or randomized confirmation may be useful")
    if context.outcome and coverage(keywords(context.outcome), text) < 0.4:
        suggestions.append("target outcome underrepresented")
    if has_any(text, ["implementation", "feasibility", "cost"]):
        suggestions.append("implementation feasibility signal")
    if not suggestions:
        suggestions.append("use for knowledge base; no explicit gap inferred from metadata")
    return "; ".join(suggestions)


def why_included(paper: dict[str, Any]) -> str:
    reasons = [paper.get("verification", "verified source")]
    reasons.append(paper.get("topic_match_gate", "topic gate not recorded"))
    reasons.append(f"{paper['relevance_score']}/40 relevance")
    reasons.append(paper["study_design"])
    if paper.get("citation_count") is not None:
        reasons.append(
            f"{paper['citation_count']} citations from "
            f"{paper.get('citation_source', 'citation source unavailable')}"
        )
    else:
        reasons.append("citation count unavailable")
    if paper.get("quartile") in {"Q3", "Q4"}:
        reasons.append(f"{paper['quartile']} exception: {exception_reason(paper)}")
    elif paper.get("quartile") == "quartile not verified":
        reasons.append("quartile not verified")
    if paper.get("recent_high_quality_note"):
        reasons.append(paper["recent_high_quality_note"])
    if paper.get("tier_cap_reason"):
        reasons.append(paper["tier_cap_reason"])
    return "; ".join(reasons)


def exception_reason(paper: dict[str, Any]) -> str:
    if paper.get("citation_strength_score", 0) >= 6:
        return "highly cited"
    if paper.get("relevance_score", 0) >= 32:
        return "directly answers the ICU question"
    if "India/LMIC evidence" in paper.get("tags", ""):
        return "India/LMIC relevance"
    if paper.get("study_design_score", 0) >= 16:
        return "strong methods"
    return "kept but downgraded"


def evidence_family(paper: dict[str, Any]) -> str:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    nct = re.search(r"\bNCT\d{8}\b", text, flags=re.IGNORECASE)
    if nct:
        return f"Trial {nct.group(0).upper()}"
    for term in DATABASE_TERMS:
        if term in text:
            return f"Database/registry: {term.upper()}"
    if paper.get("doi"):
        return f"DOI {clean_doi(paper['doi']).lower()}"
    if paper.get("pmid"):
        return f"PMID {paper['pmid']}"
    return f"Title {normalize_title(paper.get('title', ''))[:80]}"


def apply_evidence_family_ranks(papers: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in papers:
        grouped[paper["evidence_family"]].append(paper)

    for family_papers in grouped.values():
        family_papers.sort(key=lambda item: item.get("total_score", 0), reverse=True)
        for index, paper in enumerate(family_papers, start=1):
            paper["evidence_family_rank"] = index
            if index > 1:
                paper["evidence_group"] = "Secondary evidence"
                paper["why_included"] += "; same evidence family as a higher-ranked paper"


def paper_sort_key(paper: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int, int]:
    section_order = reading_section_order(paper.get("search_mode", "")).get(
        paper.get("reading_section", ""),
        99,
    )
    expected_order = int(paper.get("expected_paper_order", 999))
    tier_order = TIER_ORDER.get(paper.get("tier", ""), 9)
    topic_order = TOPIC_LEVEL_ORDER.get(paper.get("topic_match_level", ""), 9)
    mandatory_order = 0 if paper.get("mandatory_review_candidate") else 1
    family_rank = int(paper.get("evidence_family_rank", 1))
    purpose_score = int(paper.get("purpose_fit_score", 0))
    total = int(paper.get("total_score", 0))
    year = int(paper.get("year") or 0)
    return (
        section_order,
        expected_order,
        tier_order,
        topic_order,
        mandatory_order,
        family_rank,
        -purpose_score,
        -total,
        -year,
    )


def generate_knowledge_summary(
    papers: list[dict[str, Any]],
    context: SearchContext,
    manual_google_scholar_notes: str = "",
) -> dict[str, Any]:
    tier_counts = Counter(paper["tier"] for paper in papers)
    design_counts = Counter(paper["study_design"] for paper in papers)
    recent_count = sum(1 for paper in papers if paper.get("year") and context.current_year - paper["year"] <= 2)
    landmark_count = sum(1 for paper in papers if (paper.get("citation_count") or 0) >= 500)
    india_lmic_count = sum(1 for paper in papers if "India/LMIC evidence" in paper.get("tags", ""))

    top_titles = [paper["title"] for paper in papers[:5]]
    what_we_know = [
        f"{len(papers)} verified PubMed records were admitted after deduplication.",
        f"Tier 1 papers: {tier_counts.get('Tier 1: Must-read', 0)}; Tier 2 papers: {tier_counts.get('Tier 2: Useful supporting', 0)}.",
        f"Evidence mix: {format_counter(design_counts)}.",
    ]
    if top_titles:
        what_we_know.append("Highest-ranked records: " + "; ".join(top_titles))

    uncertainty = []
    if tier_counts.get("Tier 1: Must-read", 0) == 0:
        uncertainty.append("No Tier 1 paper was identified with the current search limits.")
    if india_lmic_count == 0:
        uncertainty.append("No India/LMIC-tagged record was found in the accepted set.")
    if all("Randomized controlled trial" not in paper["study_design"] for paper in papers):
        uncertainty.append("No RCT was identified in the accepted set.")
    if all("Systematic review" not in paper["study_design"] for paper in papers):
        uncertainty.append("No systematic review/meta-analysis was identified in the accepted set.")
    if not uncertainty:
        uncertainty.append("Main uncertainty should be judged by reading full texts and extracted outcomes.")

    changing = [
        f"{recent_count} accepted records were published within the last 2 years.",
        f"{landmark_count} accepted records had at least 500 verified citations.",
    ]

    clinical_usefulness = [
        "Use Tier 1 and Tier 2 papers first for protocols, teaching, and manuscript background.",
        "Use Tier 3 papers for context and Tier 4 papers only when a niche or local reason is documented.",
        "This Version 1 app maps evidence quality and gaps; it does not infer clinical conclusions from unreviewed full texts.",
    ]
    if manual_google_scholar_notes.strip():
        clinical_usefulness.append(
            "Manual Google Scholar notes were recorded for cross-checking only and were not admitted as papers."
        )

    return {
        "what_we_know": what_we_know,
        "what_remains_uncertain": uncertainty,
        "what_is_changing": changing,
        "clinical_usefulness": clinical_usefulness,
    }


def generate_gap_map(papers: list[dict[str, Any]], context: SearchContext) -> list[dict[str, str]]:
    designs = Counter(paper["study_design"] for paper in papers)
    has_rct = any("Randomized controlled trial" == paper["study_design"] for paper in papers)
    has_synthesis = any("Systematic review" in paper["study_design"] for paper in papers)
    has_india_lmic = any("India/LMIC evidence" in paper.get("tags", "") for paper in papers)
    has_implementation = any(
        has_any(f"{paper.get('title', '')} {paper.get('abstract', '')}".lower(), ["implementation", "feasibility", "cost"])
        for paper in papers
    )
    has_external_validation = any(
        "external validation" in f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        for paper in papers
    )

    gaps: list[dict[str, str]] = []
    if not has_rct and context.question_type == "Intervention or treatment":
        gaps.append(
            gap_row(
                "Evidence gap",
                "No randomized controlled trial was identified within the accepted set.",
                "Treatment questions need stronger causal evidence before bedside adoption.",
                "Pragmatic multicentre RCT or registry-nested trial",
                "Moderate to high if the intervention is already used in ICU practice",
                "High",
            )
        )
    if not has_synthesis:
        gaps.append(
            gap_row(
                "Evidence gap",
                "No systematic review/meta-analysis was identified in the accepted set.",
                "The field may lack a current synthesis or the search should be broadened.",
                "Systematic review with ICU-specific subgroup extraction",
                "High",
                "High",
            )
        )
    if not has_india_lmic:
        gaps.append(
            gap_row(
                "Population gap",
                "No India/LMIC-tagged evidence was found in the accepted set.",
                "Generalizability to Indian or resource-limited ICUs remains uncertain.",
                "Prospective multicentre Indian ICU cohort or implementation study",
                "High if routine data capture is available",
                "High",
            )
        )
    if context.outcome and all(
        coverage(keywords(context.outcome), f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()) < 0.4
        for paper in papers
    ):
        gaps.append(
            gap_row(
                "Outcome gap",
                f"The requested outcome ({context.outcome}) was not prominent in accepted titles/abstracts.",
                "Patient-important endpoints may be missing or inconsistently reported.",
                "Prospective cohort or trial using standardized ICU outcomes",
                "Moderate",
                "Medium",
            )
        )
    if designs.get("Large retrospective / database study", 0) + designs.get("Single-centre observational study", 0) > max(
        1, len(papers) // 2
    ):
        gaps.append(
            gap_row(
                "Methodology gap",
                "The accepted set is dominated by observational or database studies.",
                "Confounding, inconsistent definitions, and missing external validation may limit conclusions.",
                "Prospective cohort, external validation study, or pragmatic trial",
                "Moderate",
                "Medium",
            )
        )
    if not has_implementation:
        gaps.append(
            gap_row(
                "Implementation gap",
                "Few or no accepted records directly addressed implementation, feasibility, or cost.",
                "Even strong evidence may be difficult to apply in resource-limited ICU workflows.",
                "Before-after implementation study or mixed-methods feasibility study",
                "High",
                "Medium",
            )
        )
    if context.question_type == "Prognosis or prediction" and not has_external_validation:
        gaps.append(
            gap_row(
                "Methodology gap",
                "No external validation signal was detected for prediction/prognosis evidence.",
                "Prediction models need transportability testing before clinical use.",
                "External validation across Indian ICU network data",
                "Moderate if historical ICU data are available",
                "High",
            )
        )

    profile = topic_profile(context.topic)
    if profile:
        accepted_blob = " ".join(
            f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            for paper in papers
        )
        for subtopic in profile.get("gap_subtopics", []):
            match_terms = [term.lower() for term in subtopic.get("match_terms", []) if term]
            if not match_terms:
                continue
            if any(term in accepted_blob for term in match_terms):
                continue
            gaps.append(
                gap_row(
                    f"Subtopic gap: {subtopic.get('name', 'unspecified')}",
                    subtopic.get("gap_statement", "Subtopic not represented in accepted set."),
                    subtopic.get("why_it_matters", ""),
                    subtopic.get("best_design", ""),
                    subtopic.get("feasibility", ""),
                    subtopic.get("priority", "Medium"),
                )
            )

    if not gaps:
        gaps.append(
            gap_row(
                "Research direction",
                "No obvious metadata-level gap was detected.",
                "Full-text review is still required to judge definitions, comparators, and outcomes.",
                "Focused evidence review with manual full-text extraction",
                "High",
                "Medium",
            )
        )
    return gaps


def compute_subtopic_coverage(
    papers: list[dict[str, Any]],
    context: SearchContext,
) -> list[dict[str, Any]]:
    profile = topic_profile(context.topic)
    if not profile:
        return []
    accepted_blob = " ".join(
        f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        for paper in papers
    )
    coverage_rows: list[dict[str, Any]] = []
    for subtopic in profile.get("gap_subtopics", []):
        match_terms = [term.lower() for term in subtopic.get("match_terms", []) if term]
        hits = sum(1 for term in match_terms if term in accepted_blob)
        coverage_rows.append(
            {
                "name": subtopic.get("name", "subtopic"),
                "covered": hits > 0,
                "match_count": hits,
                "term_total": len(match_terms),
            }
        )
    return coverage_rows


def gap_row(
    gap_type: str,
    gap_statement: str,
    why_it_matters: str,
    best_study_design: str,
    feasibility: str,
    priority: str,
) -> dict[str, str]:
    return {
        "Gap type": gap_type,
        "Gap statement": gap_statement,
        "Why it matters": why_it_matters,
        "Best study design": best_study_design,
        "Feasibility in ICU/network": feasibility,
        "Priority": priority,
    }


def parse_quartile_overrides(csv_text: str) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    if not csv_text.strip():
        return overrides
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        journal = row.get("journal") or row.get("Journal") or row.get("source") or row.get("Source")
        quartile = row.get("quartile") or row.get("Quartile") or row.get("q") or row.get("Q")
        source = row.get("quartile_source") or row.get("source_name") or row.get("source") or "manual override"
        if not journal or not quartile:
            continue
        normalized_quartile = normalize_quartile(quartile)
        if normalized_quartile:
            overrides[normalize_journal(journal)] = {
                "quartile": normalized_quartile,
                "source": source,
            }
    return overrides


def lookup_quartile(journal: str, overrides: dict[str, dict[str, str]]) -> dict[str, str]:
    key = normalize_journal(journal)
    return overrides.get(key, {"quartile": "quartile not verified", "source": "quartile not verified"})


def verification_label(paper: dict[str, Any]) -> str:
    labels = []
    if paper.get("pmid"):
        labels.append(f"PMID {paper['pmid']}")
    if paper.get("doi"):
        labels.append(f"DOI {paper['doi']}")
    if paper.get("openalex_id"):
        labels.append("OpenAlex record")
    if paper.get("semantic_scholar_url"):
        labels.append("Semantic Scholar record")
    return ", ".join(labels) if labels else "unverified"


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in counter.most_common())


def keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
    return [word for word in words if word not in STOPWORDS]


def coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return min(1.0, hits / len(set(terms)))


def has_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text.lower() for term in terms)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def normalize_journal(journal: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", journal.lower()).strip()


def normalize_quartile(quartile: str) -> str:
    match = re.search(r"q\s*([1-4])", quartile.strip().lower())
    return f"Q{match.group(1)}" if match else ""


def clean_doi(doi: str) -> str:
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi
