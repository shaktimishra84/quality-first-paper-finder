"""LLM-powered topic primer for the literature search app.

Calls Gemini once per topic to generate a structured "primer" that helps the
deterministic ranking pipeline behave like a clinical-evidence librarian for
*any* medical topic — not just topics with a hand-authored profile JSON.

The primer returns:
- Expected landmark PMIDs (verified against NCBI before use)
- Must-include concepts (boost relevance score)
- Penalty rules (down-weight off-topic papers)
- Query expansion terms (synonyms, abbreviations)
- Population priority and expected paper categories

Hand-authored profiles in topics/*.json always take precedence. Disease-specific
profiles (VILI, CVT) act as test cases / curated overrides; the primer fills the
gap for un-profiled topics.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
PRIMER_TIMEOUT_S = 30
NCBI_VERIFY_TIMEOUT_S = 20
TITLE_OVERLAP_THRESHOLD = 0.5
MAX_EXPECTED_PAPERS = 12
MAX_EXPANSION_TERMS = 20
MAX_MUST_INCLUDE = 15
MAX_PENALTY_RULES = 8
MAX_SEMANTIC_TERMS = 12

VALID_CATEGORIES = (
    "Major guideline / consensus",
    "Landmark RCT",
    "Diagnostic landmark",
    "Systematic review / meta-analysis",
    "Foundational physiology / mechanism",
    "Large cohort / database study",
    "Recent high-quality update",
)


@dataclass(frozen=True)
class PenaltyRule:
    name: str
    match_terms: tuple[str, ...]
    score: int
    reason: str

    def to_profile_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "match_terms": list(self.match_terms),
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExpectedPaper:
    pmid: str
    title: str
    reason: str
    category: str

    def to_profile_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "reason": self.reason,
            "category": self.category,
        }


@dataclass(frozen=True)
class TopicPrimer:
    topic: str
    expected_papers: tuple[ExpectedPaper, ...]
    must_include_concepts: tuple[str, ...]
    penalize_rules: tuple[PenaltyRule, ...]
    query_expansion_terms: tuple[str, ...]
    population_priority: str
    expected_categories: tuple[str, ...]
    status: str  # "generated" | "cached" | "unavailable"
    notes: tuple[str, ...] = ()
    direct_synonyms: tuple[str, ...] = ()
    component_concepts: tuple[str, ...] = ()
    parent_topics: tuple[str, ...] = ()
    parallel_topics: tuple[str, ...] = ()
    mechanism_terms: tuple[str, ...] = ()
    abbreviations: tuple[str, ...] = ()

    def to_profile_dict(self) -> dict[str, Any]:
        normalized = _normalize_topic_key(self.topic)
        triggers = sorted({normalized, self.topic.strip().lower()})
        return {
            "key": f"primed:{normalized}",
            "display_name": self.topic.strip(),
            "triggers": [t for t in triggers if t],
            "direct_phrases": [self.topic.strip().lower()],
            "direct_synonyms": list(self.direct_synonyms),
            "component_concepts": list(self.component_concepts),
            "parent_topics": list(self.parent_topics),
            "parallel_topics": list(self.parallel_topics),
            "mechanism_terms": list(self.mechanism_terms),
            "direct_acronyms": list(self.abbreviations),
            "expected_papers": [p.to_profile_dict() for p in self.expected_papers],
            "must_include_concepts": list(self.must_include_concepts),
            "penalize": [r.to_profile_dict() for r in self.penalize_rules],
            "query_expansion_terms": list(self.query_expansion_terms),
            "expected_categories": list(self.expected_categories),
            "population_priority": self.population_priority,
            "_primed": True,
            "_primer_status": self.status,
            "_primer_notes": list(self.notes),
        }


def _normalize_topic_key(topic: str) -> str:
    return re.sub(r"\s+", " ", topic or "").strip().lower()


PROMPT_INSTRUCTIONS = (
    "You are a clinical-evidence librarian. The user is searching the medical "
    "literature on a topic. Return a JSON object that helps a downstream search "
    "engine score papers. Be conservative — only suggest things you are confident "
    "are clinically established. It is better to omit a field than to invent it.\n\n"
    "Required fields:\n"
    "- expected_papers: up to 8 landmark papers (guidelines, practice-changing "
    "  trials, foundational reviews, diagnostic landmarks). Provide PubMed ID "
    "  (pmid as a string of digits only), title, one-line reason, and category. "
    f"  Categories must be one of: {', '.join(repr(c) for c in VALID_CATEGORIES)}.\n"
    "- must_include_concepts: 5-12 short phrases (lower-case, 1-3 words each) "
    "  that a clinically relevant paper would mention.\n"
    "- penalize_rules: 3-6 rules to down-rank off-topic papers. Each has a "
    "  name, match_terms (lower-case phrases), a negative score (-5 to -20), "
    "  and a reason.\n"
    "- query_expansion_terms: 5-15 synonyms, abbreviations, MeSH-style "
    "  alternatives, and related terms PubMed should also search.\n"
    "- semantic_topic_gate: object with arrays for direct_synonyms, "
    "  component_concepts, parent_topics, parallel_topics, mechanism_terms, "
    "  and abbreviations. Use clinically equivalent terms, MeSH-like aliases, "
    "  parent disease categories, related clinical syndromes, and component "
    "  concepts. This gate decides whether a paper is direct evidence, a "
    "  parent-topic fallback, a parallel-topic fallback, or only background.\n"
    "- population_priority: a short phrase like 'adult ICU', 'pediatric', "
    "  'pregnant patients', 'general adult'.\n"
    "- expected_categories: which paper categories the user should expect to "
    "  find for this topic.\n\n"
    "DO NOT invent PMIDs. If unsure of a PMID, omit the paper rather than "
    "guess. Hallucinated PMIDs will be detected and discarded by the verifier."
)

GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "expected_papers": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "pmid": {"type": "STRING"},
                    "title": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                    "category": {"type": "STRING"},
                },
                "required": ["pmid", "title"],
            },
        },
        "must_include_concepts": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "penalize_rules": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "match_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "score": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["name", "match_terms", "score"],
            },
        },
        "query_expansion_terms": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "semantic_topic_gate": {
            "type": "OBJECT",
            "properties": {
                "direct_synonyms": {"type": "ARRAY", "items": {"type": "STRING"}},
                "component_concepts": {"type": "ARRAY", "items": {"type": "STRING"}},
                "parent_topics": {"type": "ARRAY", "items": {"type": "STRING"}},
                "parallel_topics": {"type": "ARRAY", "items": {"type": "STRING"}},
                "mechanism_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
                "abbreviations": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
        },
        "population_priority": {"type": "STRING"},
        "expected_categories": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "expected_papers",
        "must_include_concepts",
        "penalize_rules",
        "query_expansion_terms",
    ],
}


def _call_gemini(topic: str, gemini_key: str) -> dict[str, Any]:
    """POST a single request to Gemini Flash and return the parsed JSON payload."""
    url = f"{GEMINI_ENDPOINT}?key={gemini_key}"
    body = {
        "system_instruction": {"parts": [{"text": PROMPT_INSTRUCTIONS}]},
        "contents": [{"parts": [{"text": f"Topic: {topic}"}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": GEMINI_RESPONSE_SCHEMA,
            "temperature": 0.2,
        },
    }
    response = requests.post(url, json=body, timeout=PRIMER_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    raw_text = "".join(text_parts).strip()
    if not raw_text:
        raise ValueError("Gemini returned empty content")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Gemini returned non-object JSON")
    return parsed


def _normalize_text(text: str) -> set[str]:
    if not text:
        return set()
    cleaned = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return {t for t in cleaned.split() if len(t) > 2}


def _title_overlap(claimed: str, actual: str) -> float:
    """Token-overlap ratio between a claimed and an actual title (0.0-1.0)."""
    claimed_tokens = _normalize_text(claimed)
    actual_tokens = _normalize_text(actual)
    if not claimed_tokens or not actual_tokens:
        return 0.0
    intersection = claimed_tokens & actual_tokens
    return len(intersection) / max(1, min(len(claimed_tokens), len(actual_tokens)))


def _verify_pmids(
    candidates: list[dict[str, Any]],
    email: str,
    api_key: str,
) -> list[ExpectedPaper]:
    """Confirm each LLM-claimed PMID via NCBI esummary and drop hallucinations."""
    if not candidates:
        return []

    pmid_pairs: list[tuple[str, dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        pmid = re.sub(r"[^0-9]", "", str(candidate.get("pmid") or ""))
        if pmid:
            pmid_pairs.append((pmid, candidate))

    if not pmid_pairs:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmid for pmid, _ in pmid_pairs),
        "retmode": "json",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params=params,
            timeout=NCBI_VERIFY_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("PMID verification failed: %s", exc)
        return []

    result = payload.get("result", {})
    verified: list[ExpectedPaper] = []
    for pmid, candidate in pmid_pairs:
        record = result.get(pmid)
        if not isinstance(record, dict):
            continue
        actual_title = str(record.get("title") or "").strip()
        if not actual_title:
            continue
        claimed_title = str(candidate.get("title") or "")
        if claimed_title and _title_overlap(claimed_title, actual_title) < TITLE_OVERLAP_THRESHOLD:
            logger.debug("PMID %s rejected: title mismatch", pmid)
            continue
        category = str(candidate.get("category") or "Recent high-quality update")
        if category not in VALID_CATEGORIES:
            category = "Recent high-quality update"
        verified.append(
            ExpectedPaper(
                pmid=pmid,
                title=actual_title,
                reason=str(candidate.get("reason") or "Suggested by topic primer"),
                category=category,
            )
        )
        if len(verified) >= MAX_EXPECTED_PAPERS:
            break
    return verified


def _coerce_str_list(values: Any, max_items: int) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return tuple(out)


def _coerce_penalty_rules(values: Any) -> tuple[PenaltyRule, ...]:
    if not isinstance(values, list):
        return ()
    rules: list[PenaltyRule] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        name = str(value.get("name") or "").strip()
        terms = _coerce_str_list(value.get("match_terms"), max_items=12)
        raw_score = value.get("score")
        if not isinstance(raw_score, int):
            continue
        if not name or not terms:
            continue
        clamped = max(-25, min(0, raw_score))
        if clamped == 0:
            continue
        rules.append(
            PenaltyRule(
                name=name,
                match_terms=terms,
                score=clamped,
                reason=str(value.get("reason") or name),
            )
        )
        if len(rules) >= MAX_PENALTY_RULES:
            break
    return tuple(rules)


def _build_primer_from_payload(
    topic: str,
    payload: dict[str, Any],
    email: str,
    api_key: str,
    status: str,
) -> TopicPrimer:
    expected_candidates = payload.get("expected_papers")
    if not isinstance(expected_candidates, list):
        expected_candidates = []
    verified_papers = tuple(
        _verify_pmids(expected_candidates, email=email, api_key=api_key)
    )
    semantic_gate = payload.get("semantic_topic_gate")
    if not isinstance(semantic_gate, dict):
        semantic_gate = {}

    notes: list[str] = []
    suggested = len(expected_candidates)
    if suggested and len(verified_papers) < suggested:
        notes.append(
            f"{suggested - len(verified_papers)} suggested PMIDs failed NCBI verification"
        )

    return TopicPrimer(
        topic=topic.strip(),
        expected_papers=verified_papers,
        must_include_concepts=_coerce_str_list(
            payload.get("must_include_concepts"), max_items=MAX_MUST_INCLUDE
        ),
        penalize_rules=_coerce_penalty_rules(payload.get("penalize_rules")),
        query_expansion_terms=_coerce_str_list(
            payload.get("query_expansion_terms"), max_items=MAX_EXPANSION_TERMS
        ),
        population_priority=str(payload.get("population_priority") or "").strip(),
        expected_categories=_coerce_str_list(
            payload.get("expected_categories"), max_items=10
        ),
        status=status,
        notes=tuple(notes),
        direct_synonyms=_coerce_str_list(
            semantic_gate.get("direct_synonyms"), max_items=MAX_SEMANTIC_TERMS
        ),
        component_concepts=_coerce_str_list(
            semantic_gate.get("component_concepts"), max_items=MAX_SEMANTIC_TERMS
        ),
        parent_topics=_coerce_str_list(
            semantic_gate.get("parent_topics"), max_items=MAX_SEMANTIC_TERMS
        ),
        parallel_topics=_coerce_str_list(
            semantic_gate.get("parallel_topics"), max_items=MAX_SEMANTIC_TERMS
        ),
        mechanism_terms=_coerce_str_list(
            semantic_gate.get("mechanism_terms"), max_items=MAX_SEMANTIC_TERMS
        ),
        abbreviations=_coerce_str_list(
            semantic_gate.get("abbreviations"), max_items=MAX_SEMANTIC_TERMS
        ),
    )


_PRIMER_CACHE: dict[str, TopicPrimer | None] = {}


def prime_topic(
    topic: str,
    gemini_key: str,
    email: str = "",
    api_key: str = "",
) -> TopicPrimer | None:
    """Return a TopicPrimer for the given topic, or None on failure.

    Cached in-process by normalized topic. Repeat calls within the same
    process return a copy with status='cached'.
    """
    if not topic or not gemini_key:
        return None
    cache_key = _normalize_topic_key(topic)
    if cache_key in _PRIMER_CACHE:
        cached = _PRIMER_CACHE[cache_key]
        if cached is None:
            return None
        if cached.status == "generated":
            return _replace_status(cached, "cached")
        return cached

    try:
        payload = _call_gemini(topic, gemini_key)
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Topic primer (Gemini call) failed for '%s': %s", topic, exc)
        _PRIMER_CACHE[cache_key] = None
        return None

    try:
        primer = _build_primer_from_payload(
            topic=topic,
            payload=payload,
            email=email,
            api_key=api_key,
            status="generated",
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Topic primer (parse) failed for '%s': %s", topic, exc)
        _PRIMER_CACHE[cache_key] = None
        return None

    _PRIMER_CACHE[cache_key] = primer
    return primer


def _replace_status(primer: TopicPrimer, status: str) -> TopicPrimer:
    return TopicPrimer(
        topic=primer.topic,
        expected_papers=primer.expected_papers,
        must_include_concepts=primer.must_include_concepts,
        penalize_rules=primer.penalize_rules,
        query_expansion_terms=primer.query_expansion_terms,
        population_priority=primer.population_priority,
        expected_categories=primer.expected_categories,
        status=status,
        notes=primer.notes,
        direct_synonyms=primer.direct_synonyms,
        component_concepts=primer.component_concepts,
        parent_topics=primer.parent_topics,
        parallel_topics=primer.parallel_topics,
        mechanism_terms=primer.mechanism_terms,
        abbreviations=primer.abbreviations,
    )


def clear_primer_cache() -> None:
    """Clear the in-process primer cache. Used by tests and manual reset."""
    _PRIMER_CACHE.clear()
