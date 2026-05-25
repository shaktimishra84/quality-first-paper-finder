"""Intent-aware ranking: rank results by how well they match the user's
actual clinical intent, and keep Tier 1 tight.

Two signals, both optional and fail-soft (run only with a Gemini key):
1. Embedding similarity between the LLM-written intent statement (from the
   topic primer) and each top candidate's title+abstract — scores many papers
   cheaply.
2. An LLM re-rank of the very top slice: the model reads each paper and rates
   how well it matches the intent (0-100), catching cases where two papers share
   generic ICU vocabulary but are about different diseases.

The resulting `intent_fit` (0-1) is attached to papers, used to re-gate Tier 1
(a must-read must actually match the intent) and to order results. Never raises;
no key or any API error leaves rankings unchanged.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from evidence_engine import GEMINI_API_BASE, _sanitize_gemini_error, resolve_gemini_model
from semantic_relevance import _embed, _resolve_embed_model, cosine

EMBED_TOP_K = 80          # embed the top-N lexical candidates for an intent-similarity score
RERANK_TOP_K = 40         # LLM-rerank the very top slice for precision
RERANK_TIMEOUT_S = 60


def _paper_text(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "") or "").strip()
    abstract = str(paper.get("abstract", "") or "").strip()
    return f"{title}. {abstract}".strip()


def _llm_rerank(intent_text: str, subset: list[dict[str, Any]], gemini_key: str) -> dict[int, float]:
    """Ask the model to rate each paper's fit to the intent (0-100). Returns
    {index: fit_0_to_1}. Empty dict on any failure."""
    items = [
        {
            "id": index,
            "title": str(paper.get("title", "") or "")[:300],
            "abstract": str(paper.get("abstract", "") or "")[:600],
        }
        for index, paper in enumerate(subset)
    ]
    if not items:
        return {}
    schema = {
        "type": "OBJECT",
        "properties": {
            "ratings": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"id": {"type": "INTEGER"}, "fit": {"type": "INTEGER"}},
                    "required": ["id", "fit"],
                },
            }
        },
        "required": ["ratings"],
    }
    system = (
        "You rate how well each paper matches the user's clinical intent, on a "
        "0-100 scale. 100 = the paper directly addresses the SAME clinical entity "
        "and aspect as the intent; 0 = unrelated. Judge by the actual subject, not "
        "shared generic vocabulary: a paper about a DIFFERENT disease that merely "
        "shares words like 'critical care', 'ICU', 'syndrome', or a coincidental "
        "acronym is NOT a match and should score low. Rate every id provided."
    )
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [
            {"parts": [{"text": f"Clinical intent:\n{intent_text}\n\nPapers (JSON):\n{json.dumps(items)}"}]}
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0.0,
        },
    }
    try:
        model = resolve_gemini_model(gemini_key)
        response = requests.post(
            f"{GEMINI_API_BASE}/models/{model}:generateContent?key={gemini_key}",
            json=body,
            timeout=RERANK_TIMEOUT_S,
        )
        response.raise_for_status()
        candidates = response.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts") if candidates else []
        raw = "".join(part.get("text", "") for part in parts or [] if isinstance(part, dict))
        parsed = json.loads(raw)
    except Exception:
        return {}

    fits: dict[int, float] = {}
    for rating in (parsed.get("ratings") or []) if isinstance(parsed, dict) else []:
        if not isinstance(rating, dict):
            continue
        try:
            idx = int(rating.get("id"))
            fit = float(rating.get("fit"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(subset):
            fits[idx] = max(0.0, min(1.0, fit / 100.0))
    return fits


def rank_by_intent(intent_text: str, query: str, papers: list[dict[str, Any]], gemini_key: str) -> str:
    """Attach `intent_fit` (0-1) to top candidates. Returns "scored", "skipped",
    or "error". Never raises. Papers that were LLM-reranked also get
    `intent_reranked=True` so the caller can trust those scores for tier gating."""
    target = (intent_text or "").strip() or (query or "").strip()
    if not gemini_key or not target or not papers:
        return "skipped"
    try:
        ranked = sorted(papers, key=lambda p: int(p.get("total_score", 0) or 0), reverse=True)
        embed_subset = ranked[:EMBED_TOP_K]
        model = _resolve_embed_model(gemini_key)

        query_vec = _embed([target], gemini_key, model, "RETRIEVAL_QUERY")
        if not query_vec:
            return "error"
        doc_vecs = _embed([_paper_text(p) for p in embed_subset], gemini_key, model, "RETRIEVAL_DOCUMENT")
        if not doc_vecs or len(doc_vecs) != len(embed_subset):
            return "error"
        for paper, vec in zip(embed_subset, doc_vecs):
            sim = round(cosine(query_vec[0], vec), 4)
            paper["intent_similarity"] = sim
            paper["intent_fit"] = sim

        # LLM re-rank the very top slice for a trustworthy intent score.
        rerank_subset = ranked[:RERANK_TOP_K]
        fits = _llm_rerank(target, rerank_subset, gemini_key)
        for idx, paper in enumerate(rerank_subset):
            if idx in fits:
                paper["intent_fit"] = fits[idx]
                paper["intent_reranked"] = True
        return "scored"
    except Exception:
        return "error"
