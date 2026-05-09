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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import requests


TOPICS_DIR = Path(__file__).resolve().parent / "topics"


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper"

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

JOURNAL_SCORE = {"Q1": 20, "Q2": 15, "Q3": 6, "Q4": 2}
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

@functools.lru_cache(maxsize=1)
def load_topic_profiles() -> tuple[dict[str, Any], ...]:
    if not TOPICS_DIR.is_dir():
        return ()
    profiles: list[dict[str, Any]] = []
    for path in sorted(TOPICS_DIR.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("triggers"):
            profiles.append(data)
    return tuple(profiles)


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
    current_year: int = date.today().year


@dataclass(frozen=True)
class SearchLayer:
    name: str
    purpose: str
    query: str
    retmax: int | None = None


def build_search_layers(context: SearchContext, candidate_depth: int = 50) -> list[SearchLayer]:
    topic = context.topic.strip()
    topic_query = build_topic_query(topic)
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
    broad_retmax = max(candidate_depth, 50)
    review_retmax = max(20, candidate_depth // 2)
    focused_retmax = max(25, candidate_depth // 2)

    return [
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
            retmax=max(20, candidate_depth // 3),
        ),
    ]


def build_topic_query(topic: str) -> str:
    profile = topic_profile(topic)
    if profile and profile.get("pubmed_query"):
        return str(profile["pubmed_query"])
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
) -> dict[str, Any]:
    layers = build_search_layers(context, max_results_per_layer)
    all_papers: list[dict[str, Any]] = []
    errors: list[str] = []
    automatically_retrieved_pmids: set[str] = set()
    expected_papers = expected_papers_for_topic(context.topic)

    def _notify(message: str, completed: int, total: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(message, completed, total)
        except Exception:
            pass

    def _fetch_layer(layer: SearchLayer) -> tuple[SearchLayer, list[str], list[dict[str, Any]], str | None]:
        try:
            ids = search_pubmed(layer.query, layer.retmax or max_results_per_layer, email=email)
            papers = fetch_pubmed_records(ids, email=email) if ids else []
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
    _notify(f"Starting {total_layers} parallel PubMed searches", 0, total_layers)
    with ThreadPoolExecutor(max_workers=3) as executor:
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

    recovered_expected: list[dict[str, str]] = []
    if expected_papers:
        expected_pmids = [item["pmid"] for item in expected_papers]
        missing_from_automatic = [
            item for item in expected_papers if item["pmid"] not in automatically_retrieved_pmids
        ]
        try:
            expected_records = fetch_pubmed_records(expected_pmids, email=email)
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

    return {
        "search_date": date.today().isoformat(),
        "layers": layers,
        "papers": scored,
        "retrieved_count": len(all_papers),
        "deduped_count": len(deduped),
        "rejected_unverified": rejected,
        "errors": errors,
        "summary": generate_knowledge_summary(scored, context, manual_google_scholar_notes),
        "gap_map": generate_gap_map(scored, context),
        "subtopic_coverage": compute_subtopic_coverage(scored, context),
        "expected_papers": expected_papers,
        "recovered_expected": recovered_expected,
        "missing_expected": missing_expected,
        "missing_from_automatic": missing_from_automatic,
        "manual_google_scholar_notes": manual_google_scholar_notes.strip(),
        "enrichment_limit": enrichment_limit,
    }


def expected_papers_for_topic(topic: str) -> list[dict[str, str]]:
    profile = topic_profile(topic)
    if not profile:
        return []
    return [dict(item) for item in profile.get("expected_papers", [])]


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
        return f"HTTP {response.status_code} from source API"
    text = str(exc)
    if "NameResolutionError" in text or "Failed to resolve" in text:
        return "network/DNS unavailable from the app process"
    if "Read timed out" in text or "timed out" in text:
        return "source API timed out"
    if "Connection refused" in text:
        return "source API connection refused"
    return text[:240]


@functools.lru_cache(maxsize=256)
def _cached_search_pubmed(query: str, retmax: int, email: str) -> tuple[str, ...]:
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
    response = requests.get(
        PUBMED_SEARCH_URL,
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return tuple(payload.get("esearchresult", {}).get("idlist", []))


def search_pubmed(query: str, retmax: int, email: str = "") -> list[str]:
    return list(_cached_search_pubmed(query, retmax, email))


@functools.lru_cache(maxsize=128)
def _cached_fetch_pubmed_records(
    pmids_key: tuple[str, ...], email: str
) -> tuple[dict[str, Any], ...]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids_key),
        "retmode": "xml",
        "tool": "quality_first_paper_finder",
    }
    if email:
        params["email"] = email
    response = requests.get(
        PUBMED_FETCH_URL,
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    return tuple(parse_pubmed_article(node) for node in root.findall(".//PubmedArticle"))


def fetch_pubmed_records(pmids: list[str], email: str = "") -> list[dict[str, Any]]:
    if not pmids:
        return []
    cached = _cached_fetch_pubmed_records(tuple(pmids), email)
    return [copy.deepcopy(record) for record in cached]


def clear_pubmed_caches() -> None:
    _cached_search_pubmed.cache_clear()
    _cached_fetch_pubmed_records.cache_clear()


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
    journal_score = score_journal_quality(paper["quartile"])
    citation_score, citation_note = score_citations(paper.get("citation_count"))
    recency_score = score_recency(paper.get("year"), context.current_year)

    total_score = relevance_score + design_score + journal_score + citation_score + recency_score

    paper["topic_match_gate"] = topic_gate["gate"]
    paper["topic_match_level"] = topic_gate["level"]
    paper["topic_match_reason"] = topic_gate["reason"]
    paper["topic_match_max_tier"] = topic_gate["max_tier_label"]
    paper["raw_relevance_score"] = raw_relevance_score
    paper["relevance_score"] = relevance_score
    paper["relevance_cap"] = topic_gate["relevance_cap"]
    paper["study_design_score"] = design_score
    paper["journal_quality_score"] = journal_score
    paper["citation_strength_score"] = citation_score
    paper["recency_score"] = recency_score
    paper["total_score"] = total_score
    paper["study_design"] = design
    paper["citation_note"] = citation_note
    paper["relevance_reason"] = relevance_reason
    paper["recent_high_quality_note"] = recent_high_quality_note(paper, design, citation_score, context)
    review_protection = major_review_protection(paper, design, context)
    paper["mandatory_review_candidate"] = review_protection["candidate"]
    paper["mandatory_review_reason"] = review_protection["reason"]
    paper["score_only_tier"] = assign_tier(paper)
    paper["tier"], paper["tier_cap_reason"] = apply_tier_caps(
        paper,
        paper["score_only_tier"],
        topic_gate["max_tier_order"],
    )
    paper["evidence_group"] = classify_evidence_group(paper, design)
    paper["knowledge_roles"] = classify_knowledge_roles(paper, design, context)
    paper["tags"] = build_tags(paper, design, context)
    paper["verification"] = verification_label(paper)
    paper["why_included"] = why_included(paper)
    paper["gap_suggested"] = suggest_paper_gap(paper, context)
    paper["evidence_family"] = evidence_family(paper)
    paper["reading_section"] = assign_reading_section(paper)
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
        for layer in ["review/guideline", "landmark/classic", "expected landmark seed"]
    )
    landmark_discovery = any(
        layer in search_layers for layer in ["landmark/classic", "expected landmark seed"]
    )
    recent_comprehensive = (
        bool(paper.get("year") and context.current_year - paper["year"] <= 3)
        and strong_review_signal
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
    if title_signal and direct_or_seed:
        reasons.append("clearly focused review/update title")

    high_confidence_review = (
        design in {"Guideline / consensus / society statement", "Systematic review / meta-analysis"}
        or bool(paper.get("expected_paper_reason"))
        or major_journal
        or recent_comprehensive
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


def topic_profile(topic: str) -> dict[str, Any] | None:
    topic_text = normalize_space(topic).lower()
    if not topic_text:
        return None
    for profile in load_topic_profiles():
        triggers = profile.get("triggers", [])
        if any(trigger and trigger in topic_text for trigger in triggers):
            return profile
    return None


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

    if has_any(pub_type_text, ["practice guideline", "guideline"]) or has_any(
        title, ["guideline", "consensus statement", "society statement", "scientific statement"]
    ):
        design, score = "Guideline / consensus / society statement", 19
    elif has_any(pub_type_text, ["systematic review", "meta-analysis"]) or has_any(
        title, ["systematic review", "meta-analysis", "meta analysis"]
    ):
        design, score = "Systematic review / meta-analysis", 18
    elif has_any(pub_type_text, ["randomized controlled trial"]) or (
        not has_review_type and looks_like_original_randomized_trial(title, abstract)
    ):
        design, score = "Randomized controlled trial", 18
    elif has_review_type:
        design, score = "Narrative review", 10
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
    elif has_any(pub_type_text, ["case reports"]) or has_any(title, ["case report", "case series"]):
        design, score = "Case series / case report", 4
    else:
        design, score = "Study design not clearly classified", 6

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

    return design, min(20, max(0, score))


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


def score_journal_quality(quartile: str) -> int:
    normalized = normalize_quartile(quartile)
    return JOURNAL_SCORE.get(normalized, 0)


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


def assign_tier(paper: dict[str, Any]) -> str:
    total = paper["total_score"]
    relevance = paper["relevance_score"]
    design = paper["study_design_score"]
    journal = paper["journal_quality_score"]
    citations = paper["citation_strength_score"]
    recent_major = bool(paper.get("recent_high_quality_note"))

    if (
        relevance >= 30
        and total >= 68
        and design >= 14
        and (journal >= 15 or citations >= 6 or recent_major or design >= 18)
    ):
        return "Tier 1: Must-read"
    if relevance >= 24 and total >= 50 and design >= 8:
        return "Tier 2: Useful supporting"
    if relevance >= 14 and total >= 30:
        return "Tier 3: Background"
    return "Tier 4: Low priority"


def apply_tier_caps(
    paper: dict[str, Any],
    score_only_tier: str,
    topic_max_tier_order: int,
) -> tuple[str, str]:
    cap_order = topic_max_tier_order
    cap_reasons = []
    if topic_max_tier_order > 1:
        cap_reasons.append(
            f"Rule 0 topic gate caps this paper at {TIER_BY_ORDER[topic_max_tier_order]}"
        )

    quality_cap_order, quality_reason = quality_data_cap(paper)
    if quality_cap_order > cap_order:
        cap_order = quality_cap_order
    if quality_reason:
        cap_reasons.append(quality_reason)

    score_order = TIER_ORDER.get(score_only_tier, 5)
    final_order = max(score_order, cap_order)
    final_tier = TIER_BY_ORDER[final_order]
    if final_tier != score_only_tier and not cap_reasons:
        cap_reasons.append(f"tier capped from {score_only_tier} to {final_tier}")
    return final_tier, "; ".join(cap_reasons)


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


def assign_reading_section(paper: dict[str, Any]) -> str:
    topic_level = paper.get("topic_match_level", "")
    tier_order = TIER_ORDER.get(paper.get("tier", ""), 5)
    design = paper.get("study_design", "")
    major_design = design in {
        "Guideline / consensus / society statement",
        "Systematic review / meta-analysis",
        "Randomized controlled trial",
        "Large multicentre prospective cohort",
        "Diagnostic accuracy study",
    }

    if paper.get("mandatory_review_candidate") or paper.get("expected_paper_reason"):
        return "Core reading pack"
    if topic_level in {"background", "noise"} or tier_order >= 4:
        return "Low-priority / indirect papers"
    if topic_level in {"direct", "abstract_only"} and major_design and tier_order <= 3:
        return "Core reading pack"
    return "Extended evidence base"


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
        reasons.append(f"{paper['citation_count']} citations from {paper['citation_source']}")
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


def paper_sort_key(paper: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int]:
    section_order = {
        "Core reading pack": 0,
        "Extended evidence base": 1,
        "Low-priority / indirect papers": 2,
    }.get(paper.get("reading_section", ""), 3)
    expected_order = int(paper.get("expected_paper_order", 999))
    tier_order = TIER_ORDER.get(paper.get("tier", ""), 9)
    topic_order = TOPIC_LEVEL_ORDER.get(paper.get("topic_match_level", ""), 9)
    mandatory_order = 0 if paper.get("mandatory_review_candidate") else 1
    family_rank = int(paper.get("evidence_family_rank", 1))
    total = int(paper.get("total_score", 0))
    year = int(paper.get("year") or 0)
    return (section_order, expected_order, tier_order, topic_order, mandatory_order, family_rank, -total, -year)


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
