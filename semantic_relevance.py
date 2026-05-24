"""Optional semantic relevance signal via Gemini text embeddings.

Adds a meaning-based check on top of the lexical/rule-based scoring: embed the
query and each top candidate's title+abstract, measure cosine similarity, and
flag papers that rank high lexically but are semantically far from the query
(e.g. an acronym collision like "HPS" matching a "GreenLight HPS laser" paper).

Design choices:
- Guard, not a rewrite. We only FLAG clear outliers and attach a score; we do
  not overwrite tiers or the rule-based score. The flag is used as a sort
  demotion so egregious false positives cannot sit at the top.
- Fail-soft. No key, no abstract, or any API error -> no flags, current
  behaviour unchanged. Logging never blocks search.
- Conservative. An outlier must be high-tier, lack a protective flag, have an
  abstract, and be BOTH absolutely low and relatively far below the set's best.
  Tune the constants below against the search log before making it stricter.
"""

from __future__ import annotations

import math
from typing import Any

import requests

from evidence_engine import GEMINI_API_BASE

EMBED_TIMEOUT_S = 30
EMBED_TOP_K = 60                 # only embed the top-N candidates by lexical score
EMBED_BATCH = 90                 # Gemini batchEmbedContents request cap
_DEFAULT_EMBED_MODEL = "text-embedding-004"

# Outlier thresholds (deliberately conservative; tune via the search log).
OUTLIER_TIERS = {"Tier 1: Must-read", "Tier 2: Useful supporting"}
OUTLIER_ABS_MAX = 0.50           # must be below this absolute cosine similarity
OUTLIER_REL_GAP = 0.18           # AND at least this far below the set's best score

_RESOLVED_EMBED_MODEL: str | None = None


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _resolve_embed_model(gemini_key: str) -> str:
    """Find an embedContent-capable model (cached), like the chat-model resolver."""
    global _RESOLVED_EMBED_MODEL
    if _RESOLVED_EMBED_MODEL:
        return _RESOLVED_EMBED_MODEL
    try:
        response = requests.get(
            f"{GEMINI_API_BASE}/models", params={"key": gemini_key}, timeout=EMBED_TIMEOUT_S
        )
        response.raise_for_status()
        for model in response.json().get("models", []) or []:
            methods = model.get("supportedGenerationMethods") or []
            name = str(model.get("name", "")).split("/")[-1]
            if "embedContent" in methods and ("embedding" in name or "embed" in name):
                _RESOLVED_EMBED_MODEL = name
                return name
    except Exception:
        pass
    return _DEFAULT_EMBED_MODEL


def _embed(texts: list[str], gemini_key: str, model: str, task_type: str) -> list[list[float]] | None:
    """Embed a list of texts via batchEmbedContents. Returns None on any failure."""
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), EMBED_BATCH):
            chunk = texts[start : start + EMBED_BATCH]
            body = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": text[:8000]}]},
                        "taskType": task_type,
                    }
                    for text in chunk
                ]
            }
            response = requests.post(
                f"{GEMINI_API_BASE}/models/{model}:batchEmbedContents?key={gemini_key}",
                json=body,
                timeout=EMBED_TIMEOUT_S,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings") or []
            if len(embeddings) != len(chunk):
                return None
            vectors.extend(emb.get("values") or [] for emb in embeddings)
        return vectors
    except Exception:
        return None


def _paper_text(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "") or "").strip()
    abstract = str(paper.get("abstract", "") or "").strip()
    return f"{title}. {abstract}".strip()


def score_semantic_relevance(query: str, papers: list[dict[str, Any]], gemini_key: str) -> str:
    """Attach `semantic_score` to the top candidates and flag clear outliers.

    Mutates `papers` in place. Returns a short status: "scored", "skipped"
    (no key / nothing to do), or "error". Never raises.
    """
    if not gemini_key or not query.strip() or not papers:
        return "skipped"
    try:
        # Embed only the strongest lexical candidates — the ones that could rank high.
        ranked = sorted(papers, key=lambda p: int(p.get("total_score", 0) or 0), reverse=True)
        subset = ranked[:EMBED_TOP_K]
        model = _resolve_embed_model(gemini_key)

        query_vec = _embed([query.strip()], gemini_key, model, "RETRIEVAL_QUERY")
        if not query_vec:
            return "error"
        doc_vecs = _embed([_paper_text(p) for p in subset], gemini_key, model, "RETRIEVAL_DOCUMENT")
        if not doc_vecs or len(doc_vecs) != len(subset):
            return "error"

        scores: list[float] = []
        for paper, vec in zip(subset, doc_vecs):
            score = round(cosine(query_vec[0], vec), 4)
            paper["semantic_score"] = score
            scores.append(score)

        best = max(scores) if scores else 0.0
        for paper in subset:
            score = float(paper.get("semantic_score", 0.0) or 0.0)
            protected = bool(paper.get("expected_paper_reason") or paper.get("mandatory_review_candidate"))
            has_abstract = bool(str(paper.get("abstract", "") or "").strip())
            tier = str(paper.get("tier", "") or "")
            is_outlier = (
                tier in OUTLIER_TIERS
                and not protected
                and has_abstract
                and score < OUTLIER_ABS_MAX
                and (best - score) >= OUTLIER_REL_GAP
            )
            paper["semantic_outlier"] = is_outlier
            if is_outlier:
                paper["semantic_outlier_reason"] = (
                    f"low semantic similarity to query ({score:.2f} vs best {best:.2f})"
                )
        return "scored"
    except Exception:
        return "error"
