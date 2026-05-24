"""Research workflow and evidence synthesis helpers for the app.

This module adapts reusable workflow ideas from the open-source Feynman
research agent repository without importing its CLI/runtime stack.

Feynman attribution:
- Repository reviewed: https://github.com/companion-inc/feynman
- Commit reviewed locally: ac56c99
- License notice for adapted workflow/prompt ideas: MIT License,
  Copyright (c) 2026 Companion, Inc.

No Feynman runtime code, Pi packages, alphaXiv CLI calls, Modal, RunPod, or
Docker dependencies are used here. The implementation below is native to this
Streamlit/PubMed application and operates on already retrieved paper metadata.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from typing import Any

import requests


FEYNMAN_MIT_NOTICE = (
    "Adapted workflow ideas from Feynman "
    "(MIT License, Copyright (c) 2026 Companion, Inc.; "
    "https://github.com/companion-inc/feynman). No Feynman runtime code is bundled."
)

WORKFLOW_STAGES = [
    {
        "stage": "Plan",
        "app_role": "Frame topic/PICO, expected evidence types, and search layers.",
    },
    {
        "stage": "Gather",
        "app_role": "Run broad, review/guideline, landmark, recent, focused, API-supervised, and expected-paper retrieval.",
    },
    {
        "stage": "Researcher triage",
        "app_role": "Gate by topic match, classify design, and keep source identifiers attached.",
    },
    {
        "stage": "Verifier",
        "app_role": "Admit only records with PMID/DOI/source IDs and label missing citation/quartile enrichment.",
    },
    {
        "stage": "Reviewer",
        "app_role": "Downgrade indirect papers, flag missing expected papers, and expose uncertainty.",
    },
    {
        "stage": "Writer",
        "app_role": "Produce a structured review artifact with hierarchy, comparison matrix, gaps, and citations.",
    },
]

BIOMEDICAL_PROMPT_STRUCTURE = [
    "Frame the question as research synthesis, preferably with PICO/PICOS when relevant.",
    "Define inclusion focus before scoring: population, intervention/exposure, comparator, outcomes, and designs.",
    "Separate guidelines, systematic reviews, RCTs, cohorts, case reports, preprints, and mechanistic evidence.",
    "Prefer source-backed clinical outcomes over surrogate outcomes when both are present.",
    "Keep disagreements, single-study claims, indirect evidence, and missing full-text checks visible.",
    "Attach PMID/DOI/URL identifiers to every cited paper and mark blocked verification rather than guessing.",
]

EVIDENCE_HIERARCHY = [
    "Guideline / consensus / society statement",
    "Systematic review / meta-analysis",
    "Landmark randomized trial",
    "Randomized controlled trial",
    "Diagnostic accuracy study",
    "Large multicentre prospective cohort",
    "Large retrospective / database study",
    "Single-centre observational study",
    "Narrative review",
    "Case series / case report",
    "Experimental / animal / basic science",
    "Molecular / mechanistic study",
    "Editorial / correspondence / commentary",
    "Unclear",
]

HIGH_VALUE_DESIGNS = {
    "Guideline / consensus / society statement",
    "Systematic review / meta-analysis",
    "Landmark randomized trial",
    "Randomized controlled trial",
}
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Fallback model used only if live model discovery fails. Discovery is primary
# because Google retires/renames Gemini models over time (a stale name 404s).
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
GAP_SYNTHESIS_TIMEOUT_S = 35
EVIDENCE_SYNTHESIS_TIMEOUT_S = 50
MODEL_DISCOVERY_TIMEOUT_S = 15

# Cache of a resolved generateContent-capable model name (per process).
_RESOLVED_GEMINI_MODEL: str | None = None


def _sanitize_gemini_error(exc: Exception) -> str:
    """Strip the API key from any error text before it reaches the UI/logs."""
    message = re.sub(r"key=[A-Za-z0-9_\-]+", "key=REDACTED", str(exc))
    return message[:160]


def _rank_gemini_model(name: str) -> tuple[int, str]:
    lowered = name.lower()
    score = 0
    if "flash" in lowered:
        score -= 3  # prefer fast, cheap flash models
    if "latest" in lowered:
        score -= 1
    if any(tag in lowered for tag in ("exp", "preview", "thinking", "vision", "tts", "image")):
        score += 5  # avoid experimental / non-text-chat variants
    return (score, name)


def resolve_gemini_model(gemini_key: str, preferred: str = "") -> str:
    """Return a model name that supports generateContent for this key.

    Queries the Generative Language API once and caches a stable flash model,
    so the feature keeps working even when a hardcoded model name is retired.
    Falls back to GEMINI_MODEL if discovery fails.
    """
    global _RESOLVED_GEMINI_MODEL
    if preferred:
        return preferred
    if _RESOLVED_GEMINI_MODEL:
        return _RESOLVED_GEMINI_MODEL
    try:
        response = requests.get(
            f"{GEMINI_API_BASE}/models",
            params={"key": gemini_key},
            timeout=MODEL_DISCOVERY_TIMEOUT_S,
        )
        response.raise_for_status()
        models = response.json().get("models", []) or []
        candidates = [
            str(model.get("name", "")).split("/")[-1]
            for model in models
            if "generateContent" in (model.get("supportedGenerationMethods") or [])
        ]
        candidates = [name for name in candidates if name]
        if candidates:
            _RESOLVED_GEMINI_MODEL = sorted(candidates, key=_rank_gemini_model)[0]
            return _RESOLVED_GEMINI_MODEL
    except Exception:
        pass
    return GEMINI_MODEL


def _gemini_generate_endpoint(gemini_key: str) -> str:
    return f"{GEMINI_API_BASE}/models/{resolve_gemini_model(gemini_key)}:generateContent"


def build_evidence_review(
    result: dict[str, Any],
    max_sources: int = 80,
    gemini_key: str = "",
    generate_ai_gaps: bool = False,
    generate_ai_synthesis: bool = False,
) -> dict[str, Any]:
    """Build a structured biomedical review artifact from a search result."""
    papers = [paper for paper in result.get("papers", []) if _is_review_eligible(paper)]
    source_records = [_source_record(index + 1, paper) for index, paper in enumerate(papers[:max_sources])]

    hierarchy = _evidence_hierarchy(source_records)
    top_relevant = _select_top(source_records, limit=25)
    guidelines = _select_by_design(source_records, ["Guideline / consensus / society statement"], limit=12)
    systematic_reviews = _select_by_design(source_records, ["Systematic review / meta-analysis"], limit=12)
    randomized_trials = _select_by_design(
        source_records,
        ["Landmark randomized trial", "Randomized controlled trial"],
        limit=12,
    )
    source_comparison = _source_comparison(source_records)
    verification = _verification_summary(source_records, result)
    gaps = _gap_statements(source_records, result)
    limitations = _limitations(source_records, result)

    report = {
        "title": f"Evidence review: {_topic_label(result)}",
        "date": result.get("search_date") or date.today().isoformat(),
        "question": _question_context(result),
        "workflow": _workflow_trace(result),
        "prompt_structure": BIOMEDICAL_PROMPT_STRUCTURE,
        "top_relevant_papers": top_relevant,
        "major_guidelines": guidelines,
        "major_systematic_reviews": systematic_reviews,
        "major_randomized_trials": randomized_trials,
        "evidence_hierarchy": hierarchy,
        "source_comparison": source_comparison,
        "verification": verification,
        "gaps": gaps,
        "limitations": limitations,
        "ai_gap_synthesis": {"status": "not_requested", "items": []},
        "ai_synthesis": {
            "status": "not_requested",
            "executive_summary": "",
            "themes": [],
            "agreements": [],
            "conflicts": [],
            "uncertainties": [],
        },
        "sources": source_records,
        "license_notice": FEYNMAN_MIT_NOTICE,
    }
    if generate_ai_gaps:
        report["ai_gap_synthesis"] = generate_ai_gap_synthesis(report, gemini_key)
    if generate_ai_synthesis:
        report["ai_synthesis"] = generate_ai_evidence_synthesis(report, gemini_key)
    report["markdown"] = evidence_review_to_markdown(report)
    return report


def evidence_review_to_markdown(review: dict[str, Any]) -> str:
    """Render the evidence review as a portable Markdown artifact."""
    lines: list[str] = [
        f"# {review.get('title', 'Evidence review')}",
        "",
        f"Date: {review.get('date', '')}",
        "",
        "This is research synthesis, not medical advice.",
        "",
        "## Question and Scope",
    ]

    question = review.get("question", {}) or {}
    for label, key in [
        ("Topic", "topic"),
        ("Search purpose", "search_purpose"),
        ("Question type", "question_type"),
        ("Population", "population"),
        ("Intervention / exposure", "intervention"),
        ("Comparator", "comparator"),
        ("Outcome", "outcome"),
    ]:
        value = str(question.get(key, "") or "").strip()
        if value:
            lines.append(f"- **{label}:** {value}")

    lines.extend(["", "## Workflow Trace"])
    for item in review.get("workflow", []) or []:
        lines.append(f"- **{item.get('stage', '')}:** {item.get('status', '')}")

    _append_source_section(lines, "Top Relevant Papers", review.get("top_relevant_papers", []))
    _append_source_section(lines, "Key Guidelines", review.get("major_guidelines", []))
    _append_source_section(lines, "Major Systematic Reviews", review.get("major_systematic_reviews", []))
    _append_source_section(lines, "Major RCTs", review.get("major_randomized_trials", []))

    lines.extend(["", "## Evidence Hierarchy"])
    for row in review.get("evidence_hierarchy", []) or []:
        examples = ", ".join(row.get("example_sources", []))
        suffix = f" Example: {examples}" if examples else ""
        lines.append(
            f"- **{row.get('evidence_type', '')}:** {row.get('count', 0)} paper(s).{suffix}"
        )

    lines.extend(["", "## Source Comparison Matrix"])
    lines.append("| Source | Evidence type | Key role | Confidence | Caveats |")
    lines.append("|---|---|---|---|---|")
    for row in review.get("source_comparison", []) or []:
        lines.append(
            "| {source_id} | {evidence_type} | {key_role} | {confidence} | {caveats} |".format(
                source_id=_md(row.get("source_id", "")),
                evidence_type=_md(row.get("evidence_type", "")),
                key_role=_md(row.get("key_role", "")),
                confidence=_md(row.get("confidence", "")),
                caveats=_md(row.get("caveats", "")),
            )
        )

    verification = review.get("verification", {}) or {}
    lines.extend(["", "## Citation Verification"])
    for key in [
        "records_reviewed",
        "pmid_verified",
        "doi_present",
        "citation_counts_available",
        "quartiles_available",
        "blocked_checks",
    ]:
        if key in verification:
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {verification[key]}")

    lines.extend(["", "## Gaps in the Literature"])
    for gap in review.get("gaps", []) or []:
        lines.append(f"- {gap}")

    ai_gap = review.get("ai_gap_synthesis", {}) or {}
    ai_items = ai_gap.get("items", []) or []
    if ai_items or ai_gap.get("status") not in {"", None, "not_requested"}:
        lines.extend(["", "## AI-Assisted Gap Hypotheses"])
        status = ai_gap.get("status", "")
        if status and status != "generated":
            lines.append(f"Status: {status}")
        if ai_gap.get("note"):
            lines.append(str(ai_gap["note"]))
        for item in ai_items:
            sources = ", ".join(item.get("source_ids", []) or [])
            lines.append(
                "- **{gap}** ({confidence}). Sources: {sources}. Suggested design: {design}. {rationale}".format(
                    gap=item.get("gap", ""),
                    confidence=item.get("confidence", "uncertain"),
                    sources=sources or "source IDs unavailable",
                    design=item.get("suggested_design", "not specified"),
                    rationale=item.get("rationale", ""),
                )
            )

    ai_synthesis = review.get("ai_synthesis", {}) or {}
    synthesis_has_content = bool(
        ai_synthesis.get("executive_summary") or ai_synthesis.get("themes")
    )
    if synthesis_has_content or ai_synthesis.get("status") not in {"", None, "not_requested"}:
        lines.extend(["", "## AI-Assisted Evidence Synthesis"])
        status = ai_synthesis.get("status", "")
        if status and status != "generated":
            lines.append(f"Status: {status}")
        if ai_synthesis.get("note"):
            lines.append(str(ai_synthesis["note"]))
        if ai_synthesis.get("executive_summary"):
            lines.extend(["", "### Executive Summary", str(ai_synthesis["executive_summary"])])
        themes = ai_synthesis.get("themes", []) or []
        if themes:
            lines.extend(["", "### Themes"])
            for theme in themes:
                sources = ", ".join(theme.get("source_ids", []) or [])
                inferred = " _(inference)_" if theme.get("is_inference") else ""
                heading = theme.get("theme") or "Theme"
                lines.append(
                    f"- **{heading}** (evidence: {theme.get('strength_of_evidence', 'low')})"
                    f"{inferred}. Sources: {sources or 'n/a'}. {theme.get('summary', '')}"
                )
        agreements = ai_synthesis.get("agreements", []) or []
        if agreements:
            lines.extend(["", "### Areas of Agreement"])
            for item in agreements:
                sources = ", ".join(item.get("source_ids", []) or [])
                lines.append(f"- {item.get('statement', '')} (Sources: {sources or 'n/a'})")
        conflicts = ai_synthesis.get("conflicts", []) or []
        if conflicts:
            lines.extend(["", "### Conflicting or Divergent Evidence"])
            for item in conflicts:
                sources = ", ".join(item.get("source_ids", []) or [])
                lines.append(f"- {item.get('statement', '')} (Sources: {sources or 'n/a'})")
        uncertainties = ai_synthesis.get("uncertainties", []) or []
        if uncertainties:
            lines.extend(["", "### Open Uncertainties"])
            for item in uncertainties:
                lines.append(f"- {item}")

    lines.extend(["", "## Limitations and Uncertainty"])
    for item in review.get("limitations", []) or []:
        lines.append(f"- {item}")

    lines.extend(["", "## References"])
    for source in review.get("sources", []) or []:
        citation = _citation_line(source)
        lines.append(f"- [{source.get('source_id')}] {citation}")

    lines.extend(["", "## Adapted Workflow Notice", review.get("license_notice", FEYNMAN_MIT_NOTICE)])
    return "\n".join(lines).strip() + "\n"


def generate_ai_gap_synthesis(review: dict[str, Any], gemini_key: str) -> dict[str, Any]:
    """Generate bounded, source-ID-based research-gap hypotheses.

    The LLM is allowed to synthesize only from source metadata already admitted
    by the deterministic pipeline. It cannot add papers or citations.
    """
    if not gemini_key:
        return {
            "status": "not_configured",
            "items": [],
            "note": "Add a Gemini API key to enable source-grounded AI gap hypotheses.",
        }

    sources = [
        {
            "source_id": source.get("source_id"),
            "title": source.get("title"),
            "year": source.get("year"),
            "evidence_type": source.get("evidence_type"),
            "tier": source.get("tier"),
            "why_matters": source.get("why_matters"),
            "caveats": source.get("caveats"),
        }
        for source in (review.get("sources", []) or [])[:35]
    ]
    if not sources:
        return {
            "status": "blocked",
            "items": [],
            "note": "No verified sources were available for AI gap synthesis.",
        }

    schema = {
        "type": "OBJECT",
        "properties": {
            "gap_hypotheses": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "gap": {"type": "STRING"},
                        "rationale": {"type": "STRING"},
                        "source_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "suggested_design": {"type": "STRING"},
                        "confidence": {"type": "STRING"},
                        "limitations": {"type": "STRING"},
                    },
                    "required": ["gap", "rationale", "source_ids", "suggested_design", "confidence"],
                },
            }
        },
        "required": ["gap_hypotheses"],
    }
    system = (
        "You are a biomedical research methodologist. Generate source-grounded "
        "research gap hypotheses only from the provided source IDs and metadata. "
        "Do not invent papers, effect sizes, recommendations, or patient-specific "
        "medical advice. Each gap must cite at least one provided source_id. "
        "Use confidence values: high, moderate, low, speculative."
    )
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Review question and source metadata:\n"
                            + _jsonish(
                                {
                                    "question": review.get("question", {}),
                                    "evidence_hierarchy": review.get("evidence_hierarchy", []),
                                    "deterministic_gaps": review.get("gaps", []),
                                    "sources": sources,
                                }
                            )
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0.15,
        },
    }

    try:
        response = requests.post(
            f"{_gemini_generate_endpoint(gemini_key)}?key={gemini_key}",
            json=body,
            timeout=GAP_SYNTHESIS_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") if candidates else []
        raw_text = "".join(part.get("text", "") for part in parts or [] if isinstance(part, dict))
        parsed = json.loads(raw_text)
    except Exception as exc:
        return {
            "status": "blocked",
            "items": [],
            "note": f"AI gap synthesis could not be completed: {_sanitize_gemini_error(exc)}",
        }

    valid_source_ids = {source["source_id"] for source in sources if source.get("source_id")}
    items: list[dict[str, Any]] = []
    for raw in parsed.get("gap_hypotheses", []) if isinstance(parsed, dict) else []:
        if not isinstance(raw, dict):
            continue
        source_ids = [
            str(source_id)
            for source_id in raw.get("source_ids", [])
            if str(source_id) in valid_source_ids
        ]
        if not source_ids:
            continue
        confidence = str(raw.get("confidence", "low") or "low").lower()
        if confidence not in {"high", "moderate", "low", "speculative"}:
            confidence = "low"
        items.append(
            {
                "gap": str(raw.get("gap", "") or "").strip(),
                "rationale": str(raw.get("rationale", "") or "").strip(),
                "source_ids": source_ids[:6],
                "suggested_design": str(raw.get("suggested_design", "") or "").strip(),
                "confidence": confidence,
                "limitations": str(raw.get("limitations", "") or "").strip(),
            }
        )
        if len(items) >= 8:
            break
    return {
        "status": "generated" if items else "blocked",
        "items": items,
        "note": "AI gap hypotheses cite source IDs from the verified result set only.",
    }


def generate_ai_evidence_synthesis(review: dict[str, Any], gemini_key: str) -> dict[str, Any]:
    """Generate a grounded narrative synthesis of the retrieved literature.

    Adapts Feynman's writer + verifier agent integrity model: the LLM may only
    synthesize from source metadata already admitted by the deterministic
    pipeline. It cannot add papers, citations, statistics, or recommendations.
    Every claim must cite at least one verified source_id, disagreement is
    preserved, and inferences are labelled rather than stated as fact.
    """
    if not gemini_key:
        return {
            "status": "not_configured",
            "executive_summary": "",
            "themes": [],
            "agreements": [],
            "conflicts": [],
            "uncertainties": [],
            "note": "Add a Gemini API key to enable source-grounded AI evidence synthesis.",
        }

    sources = [
        {
            "source_id": source.get("source_id"),
            "title": source.get("title"),
            "journal": source.get("journal"),
            "year": source.get("year"),
            "study_design": source.get("study_design"),
            "evidence_type": source.get("evidence_type"),
            "tier": source.get("tier"),
            "why_matters": source.get("why_matters"),
            "key_role": source.get("key_role"),
            "confidence": source.get("confidence"),
            "caveats": source.get("caveats"),
            "recall_only": bool(source.get("expansion_recall_only")),
            "abstract": (str(source.get("abstract") or "").strip()[:800]) or None,
        }
        for source in (review.get("sources", []) or [])[:40]
    ]
    if not sources:
        return {
            "status": "blocked",
            "executive_summary": "",
            "themes": [],
            "agreements": [],
            "conflicts": [],
            "uncertainties": [],
            "note": "No verified sources were available for AI evidence synthesis.",
        }

    citing_item = {
        "type": "OBJECT",
        "properties": {
            "statement": {"type": "STRING"},
            "source_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["statement", "source_ids"],
    }
    schema = {
        "type": "OBJECT",
        "properties": {
            "executive_summary": {"type": "STRING"},
            "themes": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "theme": {"type": "STRING"},
                        "summary": {"type": "STRING"},
                        "source_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "strength_of_evidence": {"type": "STRING"},
                        "is_inference": {"type": "BOOLEAN"},
                    },
                    "required": ["theme", "summary", "source_ids", "strength_of_evidence"],
                },
            },
            "agreements": {"type": "ARRAY", "items": citing_item},
            "conflicts": {"type": "ARRAY", "items": citing_item},
            "uncertainties": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["executive_summary", "themes"],
    }
    question = review.get("question", {}) or {}
    question_type = str(question.get("question_type", "") or "")
    has_pico = any(
        str(question.get(key, "") or "").strip()
        for key in ("population", "intervention", "comparator", "outcome")
    )
    focused = has_pico or question_type in {
        "Intervention or treatment",
        "Diagnosis",
        "Prognosis or prediction",
        "Implementation or cost",
    }
    if focused:
        focus_instruction = (
            " This is a FOCUSED clinical question. Populate 'agreements' with points "
            "where the studies concur on the question and 'conflicts' where they "
            "diverge — each citing source_ids. Organise themes around the question."
        )
    else:
        focus_instruction = (
            " This is a BROAD topic overview, NOT a focused clinical question, so "
            "there is no single statement to agree or disagree on. Leave 'agreements' "
            "and 'conflicts' EMPTY. Instead map the thematic landscape: cover the main "
            "subtopics the literature addresses, and use strength_of_evidence on each "
            "theme to convey whether it is well-established, emerging, or uncertain."
        )
    system = (
        "You are a biomedical evidence synthesist. Write a faithful narrative "
        "synthesis of the literature using ONLY the provided sources (their "
        "metadata and, when present, their abstracts). Integrity rules: "
        "(1) Never introduce a paper, statistic, effect size, p-value, or claim "
        "that is not present in the supplied sources. When a source includes an "
        "abstract, you may summarise the findings it reports, but do not add "
        "numbers or conclusions the abstract does not state. When a source has "
        "no abstract, restrict yourself to metadata-level statements about it "
        "(study type, role) and do not invent its findings. (2) Every theme, "
        "agreement, and conflict must cite at least one provided source_id; do "
        "not cite IDs that were not provided. (3) Preserve uncertainty and "
        "disagreement between studies — never smooth it away. (4) Set "
        "is_inference=true for any statement that is your own synthesis across "
        "sources rather than directly supported by a single source. (5) Do NOT "
        "give patient-specific medical advice, dosing, or treatment "
        "recommendations; describe what the evidence shows, not what a clinician "
        "should do. (6) strength_of_evidence must be one of: high, moderate, "
        "low, very low — based on the evidence type and tier of the citing "
        "sources. (7) Treat sources flagged recall_only=true with extra caution: "
        "they were retrieved by a broad terminology-expansion net and may be "
        "tangential — do not let them dominate the synthesis. (8) When the "
        "supplied information is insufficient to support a claim, omit it rather "
        "than guessing. This is research synthesis, not medical advice."
    ) + focus_instruction
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Review question and verified source metadata:\n"
                            + _jsonish(
                                {
                                    "question": review.get("question", {}),
                                    "evidence_hierarchy": review.get("evidence_hierarchy", []),
                                    "deterministic_gaps": review.get("gaps", []),
                                    "limitations": review.get("limitations", []),
                                    "sources": sources,
                                }
                            )
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(
            f"{_gemini_generate_endpoint(gemini_key)}?key={gemini_key}",
            json=body,
            timeout=EVIDENCE_SYNTHESIS_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") if candidates else []
        raw_text = "".join(part.get("text", "") for part in parts or [] if isinstance(part, dict))
        parsed = json.loads(raw_text)
    except Exception as exc:
        return {
            "status": "blocked",
            "executive_summary": "",
            "themes": [],
            "agreements": [],
            "conflicts": [],
            "uncertainties": [],
            "note": f"AI evidence synthesis could not be completed: {_sanitize_gemini_error(exc)}",
        }

    if not isinstance(parsed, dict):
        parsed = {}
    valid_source_ids = {source["source_id"] for source in sources if source.get("source_id")}

    def _clean_ids(values: Any) -> list[str]:
        return [str(value) for value in (values or []) if str(value) in valid_source_ids]

    def _citing_list(values: Any) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for raw in values or []:
            if not isinstance(raw, dict):
                continue
            source_ids = _clean_ids(raw.get("source_ids"))
            statement = str(raw.get("statement", "") or "").strip()
            if not statement or not source_ids:
                continue
            cleaned.append({"statement": statement, "source_ids": source_ids[:6]})
            if len(cleaned) >= 12:
                break
        return cleaned

    themes: list[dict[str, Any]] = []
    for raw in parsed.get("themes", []) if isinstance(parsed.get("themes"), list) else []:
        if not isinstance(raw, dict):
            continue
        source_ids = _clean_ids(raw.get("source_ids"))
        summary = str(raw.get("summary", "") or "").strip()
        theme = str(raw.get("theme", "") or "").strip()
        if not source_ids or not (summary or theme):
            continue
        strength = str(raw.get("strength_of_evidence", "low") or "low").lower()
        if strength not in {"high", "moderate", "low", "very low"}:
            strength = "low"
        themes.append(
            {
                "theme": theme,
                "summary": summary,
                "source_ids": source_ids[:8],
                "strength_of_evidence": strength,
                "is_inference": bool(raw.get("is_inference", False)),
            }
        )
        if len(themes) >= 10:
            break

    executive_summary = str(parsed.get("executive_summary", "") or "").strip()
    uncertainties = [
        str(item).strip()
        for item in (parsed.get("uncertainties", []) or [])
        if str(item).strip()
    ][:10]

    has_content = bool(executive_summary or themes)
    # Agreement/conflict is a focused-question construct; for a broad topic
    # overview there is no single statement to agree or disagree on, so drop them.
    agreements = _citing_list(parsed.get("agreements")) if focused else []
    conflicts = _citing_list(parsed.get("conflicts")) if focused else []
    return {
        "status": "generated" if has_content else "blocked",
        "query_focus": "focused" if focused else "broad",
        "executive_summary": executive_summary,
        "themes": themes,
        "agreements": agreements,
        "conflicts": conflicts,
        "uncertainties": uncertainties,
        "note": (
            "AI synthesis cites source IDs from the verified result set only. "
            "Research synthesis, not medical advice."
            + (
                " Broad topic overview — agreement/conflict omitted (only meaningful "
                "for a focused question)." if not focused else ""
            )
        ),
    }


def _source_record(source_number: int, paper: dict[str, Any]) -> dict[str, Any]:
    evidence_type = evidence_type_for_paper(paper)
    caveats = _paper_caveats(paper)
    confidence = _confidence_for_paper(paper, evidence_type, caveats)
    return {
        "source_id": f"S{source_number}",
        "search_mode": str(paper.get("search_mode", paper.get("search_purpose", "")) or "").strip(),
        "title": str(paper.get("title", "") or "").strip(),
        "journal": str(paper.get("journal", "") or "").strip(),
        "year": paper.get("year") or "",
        "study_design": str(paper.get("study_design", "") or "").strip(),
        "evidence_type": evidence_type,
        "relation_type": str(paper.get("relation_type", paper.get("topic_match_gate", "")) or "").strip(),
        "tier": str(paper.get("tier", "") or "").strip(),
        "topic_match_gate": str(paper.get("topic_match_gate", "") or "").strip(),
        "score": paper.get("total_score"),
        "pmid": str(paper.get("pmid", "") or "").strip(),
        "doi": str(paper.get("doi", "") or "").strip(),
        "url": str(paper.get("url", "") or "").strip(),
        "citation_count": paper.get("citation_count"),
        "citation_source": str(paper.get("citation_source", "") or "").strip(),
        "relevance_score": paper.get("relevance_score"),
        "final_score": paper.get("final_score", paper.get("total_score")),
        "quartile": str(paper.get("quartile", "") or "").strip(),
        "verification": str(paper.get("verification", "") or "").strip(),
        "why_matters": str(paper.get("why_related", "") or "").strip() or _why_matters(paper, evidence_type),
        "key_role": _key_role(paper, evidence_type),
        "confidence": confidence,
        "caveats": "; ".join(caveats) if caveats else "No major metadata caveat flagged.",
        "source_records": str(paper.get("source_records", "") or "").strip(),
        "expansion_recall_only": bool(paper.get("expansion_recall_only")),
        "abstract": str(paper.get("abstract", "") or "").strip()[:1500],
    }


def evidence_type_for_paper(paper: dict[str, Any]) -> str:
    design = str(paper.get("study_design", "") or "").strip()
    if design in EVIDENCE_HIERARCHY:
        return design

    text = " ".join(
        str(paper.get(key, "") or "").lower()
        for key in ["title", "abstract", "publication_type", "publication_types"]
    )
    if any(term in text for term in ["guideline", "consensus", "scientific statement"]):
        return "Guideline / consensus / society statement"
    if any(term in text for term in ["systematic review", "meta-analysis", "meta analysis"]):
        return "Systematic review / meta-analysis"
    if "randomized" in text or "randomised" in text:
        return "Randomized controlled trial"
    if "cohort" in text or "registry" in text or "database" in text:
        return "Large retrospective / database study"
    if "case report" in text or "case series" in text:
        return "Case series / case report"
    if "review" in text:
        return "Narrative review"
    return "Unclear"


def _is_review_eligible(paper: dict[str, Any]) -> bool:
    tier = str(paper.get("tier", "") or "")
    gate = str(paper.get("topic_match_gate", "") or "")
    if "Noise" in tier or "Noise" in gate:
        return False
    return bool(str(paper.get("reading_section", "") or "").strip())


def _select_top(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return records[:limit]


def _select_by_design(
    records: list[dict[str, Any]], evidence_types: list[str], limit: int
) -> list[dict[str, Any]]:
    allowed = set(evidence_types)
    return [record for record in records if record.get("evidence_type") in allowed][:limit]


def _evidence_hierarchy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(record.get("evidence_type", "Unclear") for record in records)
    rows: list[dict[str, Any]] = []
    for evidence_type in EVIDENCE_HIERARCHY:
        count = counts.get(evidence_type, 0)
        if not count:
            continue
        examples = [
            record["source_id"]
            for record in records
            if record.get("evidence_type") == evidence_type
        ][:3]
        rows.append(
            {
                "evidence_type": evidence_type,
                "count": count,
                "example_sources": examples,
                "hierarchy_rank": EVIDENCE_HIERARCHY.index(evidence_type) + 1,
            }
        )
    return rows


def _source_comparison(records: list[dict[str, Any]], limit: int = 25) -> list[dict[str, str]]:
    comparison: list[dict[str, str]] = []
    for record in records[:limit]:
        comparison.append(
            {
                "source_id": record.get("source_id", ""),
                "title": record.get("title", ""),
                "evidence_type": record.get("evidence_type", ""),
                "key_role": record.get("key_role", ""),
                "confidence": record.get("confidence", ""),
                "caveats": record.get("caveats", ""),
            }
        )
    return comparison


def _verification_summary(records: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    blocked = []
    citation_missing = sum(1 for record in records if record.get("citation_count") in (None, ""))
    quartile_missing = sum(
        1
        for record in records
        if record.get("quartile") in ("", "quartile not verified", None)
    )
    if citation_missing:
        blocked.append(f"{citation_missing} citation-count enrichment checks unavailable")
    if quartile_missing:
        blocked.append(f"{quartile_missing} journal quartile checks unavailable")
    if result.get("missing_expected"):
        blocked.append(f"{len(result.get('missing_expected', []))} expected-paper check(s) still missing")
    for error in result.get("errors", []) or []:
        blocked.append(str(error))

    return {
        "records_reviewed": len(records),
        "pmid_verified": sum(1 for record in records if record.get("pmid")),
        "doi_present": sum(1 for record in records if record.get("doi")),
        "citation_counts_available": sum(
            1 for record in records if record.get("citation_count") not in (None, "")
        ),
        "quartiles_available": sum(
            1
            for record in records
            if record.get("quartile") not in ("", "quartile not verified", None)
        ),
        "blocked_checks": "; ".join(blocked) if blocked else "None flagged by metadata pipeline.",
    }


def _gap_statements(records: list[dict[str, Any]], result: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    types = {record.get("evidence_type") for record in records}
    if "Guideline / consensus / society statement" not in types:
        gaps.append("No guideline or consensus statement was retrieved; check society guidance manually.")
    if "Systematic review / meta-analysis" not in types:
        gaps.append("No systematic review or meta-analysis was retrieved; evidence synthesis may be incomplete.")
    if not {"Landmark randomized trial", "Randomized controlled trial"} & types:
        gaps.append("No randomized trial was retrieved; causal claims should be treated cautiously.")
    if result.get("missing_expected"):
        titles = [
            str(item.get("title", "") or "").strip()
            for item in result.get("missing_expected", [])[:5]
        ]
        gaps.append("Expected papers still missing: " + "; ".join(t for t in titles if t))

    for gap in result.get("gap_map", []) or []:
        statement = str(gap.get("Gap statement", "") or "").strip()
        if statement:
            gaps.append(statement)

    if not gaps:
        gaps.append("No major metadata-derived gap was flagged, but full-text review is still needed.")
    return _dedupe(gaps)


def _limitations(records: list[dict[str, Any]], result: dict[str, Any]) -> list[str]:
    limitations = [
        "Ranking is metadata- and abstract-driven; it is not a substitute for full-text critical appraisal.",
        "Citation counts and journal quartiles can be unavailable or stale and should not dominate clinical relevance.",
        "Indirect, single-source, or observational findings should not be merged into guideline-level conclusions.",
        "The app verifies identifiers and source availability, not every claim inside every full text.",
    ]
    if any(record.get("confidence") == "low" for record in records):
        limitations.append("Some retained sources have low ranking confidence because identifiers, abstracts, citations, or topic gates are incomplete.")
    if result.get("manual_google_scholar_notes"):
        limitations.append("Manual Google Scholar notes were treated as cross-check signals only, not as verified source records.")
    api_warnings = (result.get("api_discovery", {}) or {}).get("warnings", []) or []
    if api_warnings:
        limitations.append("One or more API-supervisor expansion checks were degraded: " + "; ".join(api_warnings))
    return _dedupe(limitations)


def _expansion_trace_note(result: dict[str, Any]) -> str:
    """Describe whether LLM query expansion fired, for the workflow trace."""
    primer_status = str(result.get("topic_primer_status", "") or "")
    recall_terms = 0
    for layer in result.get("layers", []) or []:
        name = layer.get("name", "") if isinstance(layer, dict) else getattr(layer, "name", "")
        if name == "LLM-expanded recall":
            query = layer.get("query", "") if isinstance(layer, dict) else getattr(layer, "query", "")
            recall_terms = str(query).count("[Title/Abstract]")
            break
    if recall_terms >= 2:
        basis = {
            "generated": "LLM primer",
            "cached": "LLM primer",
            "profile": "curated profile",
        }.get(primer_status, primer_status or "topic profile")
        return f" LLM query expansion: {recall_terms} recall terms from {basis}."
    if primer_status in {"generated", "cached"}:
        return " LLM primer active (no extra recall layer in this mode)."
    if primer_status == "profile":
        return " Expansion from curated topic profile."
    if primer_status in {"error", "unavailable"}:
        return f" LLM query expansion off (primer: {primer_status})."
    return ""


def _workflow_trace(result: dict[str, Any]) -> list[dict[str, str]]:
    layer_count = len(result.get("layers", []) or [])
    api_pmids = len((result.get("api_discovery", {}) or {}).get("pmids", []) or [])
    missing_expected = len(result.get("missing_expected", []) or [])
    retrieved = result.get("retrieved_count", 0)
    accepted = len(result.get("papers", []) or [])
    purpose = result.get("search_purpose", "research goal")
    values = {
        "Plan": f"Topic/PICO framed for {purpose}; {layer_count} search layers configured.{_expansion_trace_note(result)}",
        "Gather": f"{retrieved} candidate records gathered; API supervisor contributed {api_pmids} PMID(s).",
        "Researcher triage": f"{accepted} verified records admitted after dedupe, topic gate, and design classification.",
        "Verifier": "PMID/DOI/source identifiers checked before admission; enrichment caveats retained.",
        "Reviewer": f"{missing_expected} expected-paper gap(s) remain after sanity checks.",
        "Writer": "Structured evidence review, hierarchy, comparison matrix, gaps, and limitations generated.",
    }
    return [
        {"stage": item["stage"], "status": values.get(item["stage"], item["app_role"])}
        for item in WORKFLOW_STAGES
    ]


def _question_context(result: dict[str, Any]) -> dict[str, str]:
    context = result.get("question_context", {}) or {}
    topic = context.get("topic") or result.get("topic_used") or result.get("topic_original") or ""
    return {
        "topic": str(topic),
        "search_purpose": str(context.get("search_purpose", result.get("search_purpose", "")) or ""),
        "question_type": str(context.get("question_type", "General evidence map") or ""),
        "population": str(context.get("population", "") or ""),
        "intervention": str(context.get("intervention", "") or ""),
        "comparator": str(context.get("comparator", "") or ""),
        "outcome": str(context.get("outcome", "") or ""),
    }


def _topic_label(result: dict[str, Any]) -> str:
    return _question_context(result).get("topic") or "medical topic"


def _why_matters(paper: dict[str, Any], evidence_type: str) -> str:
    existing = str(paper.get("why_included", "") or "").strip()
    if existing:
        return existing
    if paper.get("expected_paper_reason"):
        return "Expected landmark/review candidate for this topic."
    if paper.get("mandatory_review_reason"):
        return "Protected as a major review, guideline, or landmark candidate."
    if evidence_type == "Guideline / consensus / society statement":
        return "Guideline-level source useful for practice framing and definitions."
    if evidence_type == "Systematic review / meta-analysis":
        return "Synthesizes multiple studies and helps locate agreement or disagreement."
    if evidence_type in {"Landmark randomized trial", "Randomized controlled trial"}:
        return "Trial evidence can anchor intervention effectiveness or safety."
    if "cohort" in evidence_type.lower() or "database" in evidence_type.lower():
        return "Observational evidence can inform prognosis, epidemiology, and external validity."
    return "Relevant source retained for background, mechanism, or gap mapping."


def _key_role(paper: dict[str, Any], evidence_type: str) -> str:
    roles = str(paper.get("knowledge_roles", "") or "").strip()
    if roles:
        return roles
    if paper.get("expected_paper_reason"):
        return "landmark / sanity-check seed"
    if evidence_type == "Guideline / consensus / society statement":
        return "guideline anchor"
    if evidence_type == "Systematic review / meta-analysis":
        return "evidence synthesis"
    if evidence_type in {"Landmark randomized trial", "Randomized controlled trial"}:
        return "intervention evidence"
    if "cohort" in evidence_type.lower():
        return "prognosis or epidemiology evidence"
    if evidence_type == "Narrative review":
        return "clinical background"
    return "supporting context"


def _paper_caveats(paper: dict[str, Any]) -> list[str]:
    caveats: list[str] = []
    if not str(paper.get("abstract", "") or "").strip():
        caveats.append("abstract unavailable")
    if paper.get("citation_count") is None:
        caveats.append("citation count unavailable")
    if str(paper.get("quartile", "") or "") in {"", "quartile not verified"}:
        caveats.append("journal quartile not verified")
    gate = str(paper.get("topic_match_gate", "") or "")
    if gate and gate not in {"Direct topic match", "Abstract-only topic match"}:
        caveats.append(gate.lower())
    if "Narrative review" == str(paper.get("study_design", "") or ""):
        caveats.append("narrative review, not systematic synthesis")
    if "observational" in str(paper.get("study_design", "") or "").lower():
        caveats.append("observational design")
    if paper.get("tier_cap_reason"):
        caveats.append(str(paper.get("tier_cap_reason")))
    if paper.get("expansion_recall_only"):
        caveats.append("entered via expanded-recall net only")
    if paper.get("semantic_outlier"):
        caveats.append(str(paper.get("semantic_outlier_reason") or "low semantic similarity to query"))
    return _dedupe(caveats)


def _confidence_for_paper(
    paper: dict[str, Any], evidence_type: str, caveats: list[str]
) -> str:
    score = 0
    tier = str(paper.get("tier", "") or "")
    gate = str(paper.get("topic_match_gate", "") or "")
    if "Tier 1" in tier:
        score += 3
    elif "Tier 2" in tier:
        score += 2
    elif "Tier 3" in tier:
        score += 1
    if gate == "Direct topic match":
        score += 2
    elif gate == "Abstract-only topic match":
        score += 1
    if evidence_type in HIGH_VALUE_DESIGNS:
        score += 2
    if paper.get("pmid"):
        score += 1
    if paper.get("doi"):
        score += 1
    if paper.get("expected_paper_reason") or paper.get("mandatory_review_reason"):
        score += 1
    score -= min(3, len(caveats))
    if score >= 6:
        return "high"
    if score >= 3:
        return "moderate"
    return "low"


def _append_source_section(lines: list[str], title: str, records: list[dict[str, Any]]) -> None:
    lines.extend(["", f"## {title}"])
    if not records:
        lines.append("No source in this category was retrieved.")
        return
    for record in records:
        lines.append(
                "- [{sid}] **{title}** ({journal}, {year}) - {why} PMID: {pmid}; DOI: {doi}".format(
                sid=record.get("source_id", ""),
                title=record.get("title", ""),
                journal=record.get("journal", "journal unavailable"),
                year=record.get("year", "year unavailable"),
                why=record.get("why_matters", ""),
                pmid=record.get("pmid", "") or "unavailable",
                doi=record.get("doi", "") or "unavailable",
            )
        )


def _citation_line(source: dict[str, Any]) -> str:
    parts = [
        str(source.get("title", "") or "").strip(),
        str(source.get("journal", "") or "").strip(),
        str(source.get("year", "") or "").strip(),
    ]
    ids = []
    if source.get("pmid"):
        ids.append(f"PMID {source['pmid']}")
    if source.get("doi"):
        ids.append(f"DOI {source['doi']}")
    if source.get("url"):
        ids.append(str(source["url"]))
    return ". ".join(part for part in parts if part) + (" - " + "; ".join(ids) if ids else "")


def _md(value: Any) -> str:
    text = str(value or "")
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _jsonish(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out
