from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from evidence_engine import build_evidence_review
from paper_finder import (
    SEARCH_PURPOSE_DEEP,
    SEARCH_PURPOSE_DEFAULT,
    SEARCH_PURPOSE_KNOWLEDGE,
    SEARCH_PURPOSE_RARE,
    SEARCH_PURPOSE_RESEARCH,
    SearchContext,
    parse_quartile_overrides,
    run_quality_first_search,
    search_purpose_config,
    topic_profile,
)


APP_NAME = "CorePapers"
APP_TAGLINE = "Verified Medical Literature Search"
APP_PREVIEW_TITLE = f"{APP_NAME} | {APP_TAGLINE}"
APP_PREVIEW_DESCRIPTION = (
    "Purpose-aware PubMed evidence discovery for clinicians and researchers. "
    "Find landmark reviews, trials, guidelines, research gaps, and rare case "
    "literature with transparent evidence tiers."
)
APP_PREVIEW_IMAGE_FILENAME = "corepapers-whatsapp-preview.png"


st.set_page_config(
    page_title=APP_PREVIEW_TITLE,
    layout="wide",
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
STYLE_PATH = ASSETS_DIR / "styles.css"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

	    :root {
	        --qf-blue:    #2563EB;
	        --qf-cyan:    #0891B2;
	        --qf-red:     #DC2626;
	        --qf-amber:   #B45309;
	        --qf-green:   #047857;
	        --qf-violet:  #7C3AED;
	        --qf-tier-1:  #A16207;
	        --qf-tier-2:  #2563EB;
	        --qf-tier-3:  #475569;
	        --qf-tier-4:  #64748B;
	        --qf-noise:   #B91C1C;
	        --qf-muted:   #64748B;
	        --qf-bg: #FFFFFF;
	        --qf-bg-soft: #F8FAFC;
	        --qf-bg-muted: #F1F5F9;
	        --qf-text: #111827;
	        --qf-text-soft: #334155;
	        --qf-surface-border: #DDE5EE;
	        --qf-soft-border: #CBD5E1;
	        --qf-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
	    }

	    html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stTextInput, .stTextArea {
	        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
	        font-feature-settings: "tnum" 1;
	        color: var(--qf-text);
	    }
	    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
	        background: var(--qf-bg) !important;
	    }
	    [data-testid="stHeader"], [data-testid="stToolbar"] {
	        background: rgba(255, 255, 255, 0.94) !important;
	    }
	    [data-testid="stSidebar"] {
	        background: var(--qf-bg-soft) !important;
	    }
	    textarea, input, [data-baseweb="select"] > div {
	        background-color: var(--qf-bg) !important;
	        color: var(--qf-text) !important;
	        border-color: var(--qf-surface-border) !important;
	    }
	    label, [data-testid="stWidgetLabel"] {
	        color: var(--qf-text) !important;
	        font-weight: 600;
	    }
	    p, li, span, div {
	        letter-spacing: 0;
	    }
	    [data-testid="stDataFrame"] {
	        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
	        font-feature-settings: "tnum" 1;
	    }
	    [data-testid="stMetricValue"], code {
	        font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
	        font-variant-numeric: tabular-nums;
	    }

	    .main .block-container {
	        padding-top: 1.25rem;
	        max-width: 1600px;
	    }
	    h1 { font-size: 2.1rem; font-weight: 700; letter-spacing: 0; }
	    h2 { font-size: 1.35rem; font-weight: 600; letter-spacing: 0; }
	    h3 { font-size: 1.1rem; font-weight: 600; }
	    h4 { font-size: 1rem; font-weight: 600; }

	    [data-testid="stMetric"] {
	        background: var(--qf-bg);
	        border: 1px solid var(--qf-surface-border);
	        border-radius: 8px;
	        padding: 0.65rem 0.85rem;
	        color: var(--qf-text);
	        box-shadow: var(--qf-shadow);
	        transition: border-color 180ms ease;
	    }
	    [data-testid="stMetric"]:hover { border-color: var(--qf-soft-border); }
	    [data-testid="stMetricLabel"] { font-size: 0.74rem; opacity: 0.72; letter-spacing: 0; }
	    [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 500; }
	    [data-testid="stMetricDelta"] svg { display: none; }

	    .qf-app-header {
	        border-bottom: 1px solid var(--qf-surface-border);
	        padding: 0.35rem 0 1rem 0;
	        margin-bottom: 0.7rem;
	    }
	    .qf-app-kicker {
	        color: var(--qf-cyan);
	        font-size: 0.78rem;
	        font-weight: 600;
	        letter-spacing: 0;
	        margin-bottom: 0.15rem;
	    }
	    .qf-app-title {
	        font-size: 2.05rem;
	        font-weight: 700;
	        line-height: 1.16;
	        letter-spacing: 0;
	        margin: 0;
	        color: var(--qf-text);
	    }
	    .qf-app-subtitle {
	        max-width: 860px;
	        color: var(--qf-muted);
	        font-size: 0.96rem;
	        line-height: 1.55;
	        margin-top: 0.35rem;
	    }
	    .qf-mode-caption {
	        color: var(--qf-text-soft);
	        font-size: 0.84rem;
	        line-height: 1.42;
	        margin-top: 0.35rem;
	    }
	    .qf-results-header {
	        border-top: 1px solid var(--qf-surface-border);
	        border-bottom: 1px solid var(--qf-surface-border);
	        padding: 0.95rem 0;
	        margin: 0.8rem 0 0.75rem 0;
	    }
	    .qf-results-title {
	        font-size: 1.22rem;
	        font-weight: 700;
	        line-height: 1.35;
	        margin-bottom: 0.25rem;
	        color: var(--qf-text);
	    }
	    .qf-results-meta {
	        color: var(--qf-muted);
	        font-size: 0.9rem;
	        line-height: 1.45;
	    }

	    .qf-rule {
	        border-left: 4px solid var(--qf-blue);
        padding: 0.35rem 0 0.35rem 0.75rem;
        color: var(--qf-text);
        font-size: 0.92rem;
    }
    .qf-error {
        border-left: 4px solid var(--qf-red);
        padding: 0.35rem 0 0.35rem 0.75rem;
    }
	    .qf-chip {
	        display: inline-block;
	        padding: 0.18rem 0.52rem;
	        border-radius: 999px;
	        font-size: 0.74rem;
	        font-weight: 600;
	        margin-right: 0.4rem;
	        margin-bottom: 0.25rem;
	        border: 1px solid currentColor;
	        background: #FFFFFF;
	        line-height: 1.4;
	        transition: background-color 150ms ease;
	    }
	    .qf-chip-blue   { color: var(--qf-blue); }
	    .qf-chip-amber  { color: var(--qf-amber); }
	    .qf-chip-green  { color: var(--qf-green); }
	    .qf-chip-violet { color: var(--qf-violet); }
	    .qf-chip-muted  { color: var(--qf-muted); }
    .qf-chip-tier-1 { color: var(--qf-tier-1); background: #FEF9C3; }
    .qf-chip-tier-2 { color: var(--qf-tier-2); background: #EFF6FF; }
    .qf-chip-tier-3 { color: var(--qf-tier-3); }
    .qf-chip-tier-4 { color: var(--qf-tier-4); }
    .qf-chip-noise  { color: var(--qf-noise); background: rgba(248, 113, 113, 0.08); }

	    .qf-section-caption {
	        font-size: 0.72rem;
	        text-transform: uppercase;
	        letter-spacing: 0;
	        opacity: 0.6;
	        margin: 0.35rem 0 0.4rem 0;
	        font-weight: 500;
    }
    .qf-detail {
        background: var(--qf-bg);
        border: 1px solid var(--qf-surface-border);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        box-shadow: var(--qf-shadow);
    }
	    .qf-detail h4 { margin-top: 0; margin-bottom: 0.6rem; line-height: 1.35; }

	    .qf-section-grid {
	        display: grid;
	        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
	        gap: 0.55rem;
	        margin: 0.4rem 0 1rem 0;
	    }
	    .qf-section-tile {
	        border: 1px solid var(--qf-surface-border);
	        border-radius: 8px;
	        padding: 0.72rem 0.8rem;
	        background: var(--qf-bg);
	        box-shadow: var(--qf-shadow);
	    }
	    .qf-section-tile-title {
	        font-size: 0.86rem;
	        font-weight: 600;
	        line-height: 1.25;
	        margin-bottom: 0.35rem;
	    }
	    .qf-section-tile-meta {
	        color: var(--qf-muted);
	        font-size: 0.78rem;
	    }
	    .qf-paper-card {
	        border: 1px solid var(--qf-surface-border);
	        border-radius: 8px;
	        padding: 0.85rem 0.95rem;
	        margin-bottom: 0.65rem;
	        background: var(--qf-bg);
	        box-shadow: var(--qf-shadow);
	    }
	    .qf-paper-card:hover {
	        border-color: var(--qf-soft-border);
	    }
	    .qf-card-title {
	        font-size: 0.98rem;
	        font-weight: 700;
	        line-height: 1.35;
	        margin-bottom: 0.25rem;
	        color: var(--qf-text);
	    }
	    .qf-paper-meta {
	        color: var(--qf-muted);
	        font-size: 0.82rem;
	        line-height: 1.4;
	        margin-bottom: 0.45rem;
	    }
	    .qf-paper-why {
	        color: var(--qf-text-soft);
	        font-size: 0.86rem;
	        line-height: 1.45;
	        margin-top: 0.35rem;
	    }
	    .qf-paper-rank {
	        color: var(--qf-cyan);
	        font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
	        font-size: 0.78rem;
	        font-weight: 600;
	        margin-bottom: 0.25rem;
	    }
	    .qf-empty-state {
	        border: 1px solid var(--qf-surface-border);
	        border-radius: 8px;
	        padding: 1rem 1.1rem;
	        background: var(--qf-bg);
	        box-shadow: var(--qf-shadow);
	    }
	    .qf-empty-title {
	        font-size: 1rem;
	        font-weight: 650;
	        margin-bottom: 0.35rem;
	    }
	    .qf-empty-body {
	        color: var(--qf-muted);
	        font-size: 0.92rem;
	        line-height: 1.5;
	    }
	    [data-testid="stDataFrame"] {
	        border: 1px solid var(--qf-surface-border);
	        border-radius: 8px;
	        overflow: hidden;
	        box-shadow: var(--qf-shadow);
	    }
	    [data-testid="stExpander"] {
	        border-color: var(--qf-surface-border);
	        border-radius: 8px;
	    }

	    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--qf-surface-border); }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.9rem;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def inject_global_styles() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600;6..72,700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )
    try:
        css = STYLE_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_social_metadata() -> None:
    metadata = {
        "title": APP_PREVIEW_TITLE,
        "description": APP_PREVIEW_DESCRIPTION,
        "imagePath": f"app/static/{APP_PREVIEW_IMAGE_FILENAME}",
    }
    markup = f"""
        <script>
        (() => {{
          const metadata = {json.dumps(metadata)};
          const parentDocument = window.parent?.document || document;
          const parentLocation = window.parent?.location || window.location;
          const imageUrl = new URL(metadata.imagePath, parentLocation.href).href;

          parentDocument.title = metadata.title;

          function ensureMeta(attributeName, attributeValue, content) {{
            let tag = parentDocument.head.querySelector(
              `meta[${{attributeName}}="${{attributeValue}}"]`
            );
            if (!tag) {{
              tag = parentDocument.createElement("meta");
              tag.setAttribute(attributeName, attributeValue);
              parentDocument.head.appendChild(tag);
            }}
            tag.setAttribute("content", content);
          }}

          ensureMeta("name", "description", metadata.description);
          ensureMeta("property", "og:title", metadata.title);
          ensureMeta("property", "og:site_name", "CorePapers");
          ensureMeta("property", "og:description", metadata.description);
          ensureMeta("property", "og:type", "website");
          ensureMeta("property", "og:image", imageUrl);
          ensureMeta("property", "og:image:width", "1200");
          ensureMeta("property", "og:image:height", "630");
          ensureMeta("name", "twitter:card", "summary_large_image");
          ensureMeta("name", "twitter:title", metadata.title);
          ensureMeta("name", "twitter:description", metadata.description);
          ensureMeta("name", "twitter:image", imageUrl);
        }})();
        </script>
        """
    try:
        st.html(markup, unsafe_allow_javascript=True)
    except TypeError:
        st.markdown(markup, unsafe_allow_html=True)


inject_global_styles()
inject_social_metadata()


DISPLAY_COLUMNS = [
    "search_mode",
    "pmid",
    "reading_section",
    "tier",
    "relation_type",
    "topic_match_gate",
    "relevance_score",
    "design_strength_score",
    "journal_quality_score",
    "citation_score",
    "recency_score",
    "final_score",
    "total_score",
    "title",
    "journal",
    "year",
    "study_design",
    "publication_type",
    "citation_count",
    "purpose_fit_reason",
    "mandatory_review_reason",
    "expected_paper_reason",
    "api_discovery_reason",
    "verification",
    "why_related",
    "why_included",
    "topic_match_reason",
    "tier_cap_reason",
    "gap_suggested",
    "url",
]

VISIBLE_COLUMN_ORDER = [
    "reading_section",
    "tier",
    "relation_type",
    "title",
    "journal",
    "year",
    "study_design",
    "citation_count",
    "final_score",
    "url",
]

FULL_COLUMNS = [
    "search_mode",
    "title",
    "normalized_title",
    "authors",
    "year",
    "journal",
    "quartile",
    "quartile_source",
    "study_design",
    "publication_type",
    "relation_type",
    "pmid",
    "doi",
    "url",
    "openalex_id",
    "semantic_scholar_url",
    "citation_count",
    "citation_count_missing",
    "citation_source",
    "relevance_score",
    "clinical_relevance_score",
    "design_strength_score",
    "journal_quality_score",
    "citation_score",
    "recency_score",
    "purpose_fit_score",
    "purpose_fit_reason",
    "penalty_score",
    "final_score",
    "total_score",
    "tier",
    "reason_for_tier",
    "ranking_confidence",
    "landmark_seed_match",
    "score_only_tier",
    "tier_cap_reason",
    "reading_section",
    "mandatory_review_candidate",
    "mandatory_review_reason",
    "expected_paper_reason",
    "api_discovery_reason",
    "topic_match_gate",
    "topic_match_level",
    "topic_match_reason",
    "topic_match_max_tier",
    "raw_relevance_score",
    "relevance_cap",
    "verification",
    "evidence_group",
    "evidence_family",
    "evidence_family_rank",
    "knowledge_roles",
    "tags",
    "why_included",
    "why_related",
    "gap_suggested",
    "relevance_reason",
    "recent_high_quality_note",
    "search_layers",
    "publication_types",
    "abstract",
]

SEARCH_MODE_UI_COPY = {
    "Learning mode": {
        "internal": SEARCH_PURPOSE_KNOWLEDGE,
        "description": "Best for topic understanding. Prioritises guidelines, narrative reviews, systematic reviews, and landmark trials.",
    },
    "Research mode": {
        "internal": SEARCH_PURPOSE_RESEARCH,
        "description": "Best for research planning. Prioritises RCTs, meta-analyses, cohorts, diagnostic studies, and evidence gaps.",
    },
    "Deep search mode": {
        "internal": SEARCH_PURPOSE_DEEP,
        "description": "Best for exhaustive review. Includes broader related papers and lower-tier evidence, but ranks them transparently.",
    },
    "Rare / case mode": {
        "internal": SEARCH_PURPOSE_RARE,
        "description": "Best for rare diseases, case reports, correspondence, editorials, and uncommon presentations.",
    },
}
SEARCH_PURPOSE_UI_OPTIONS = list(SEARCH_MODE_UI_COPY)
SEARCH_PURPOSE_UI_TO_INTERNAL = {
    label: str(copy["internal"])
    for label, copy in SEARCH_MODE_UI_COPY.items()
}
SEARCH_PURPOSE_INTERNAL_TO_UI = {
    internal: label
    for label, internal in SEARCH_PURPOSE_UI_TO_INTERNAL.items()
}
SEARCH_PURPOSE_UI_DEFAULT = SEARCH_PURPOSE_INTERNAL_TO_UI[SEARCH_PURPOSE_DEFAULT]


def e(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def short_text(value: object, limit: int = 220) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "0"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return e(value)


def has_text(value: object) -> bool:
    return bool(short_text(value, 2))


def first_text(*values: object) -> str:
    for value in values:
        text = short_text(value)
        if text:
            return text
    return ""


def tier_chip_class(tier: object) -> str:
    tier_str = str(tier or "")
    if "Tier 1" in tier_str:
        return "qf-chip-tier-1"
    if "Tier 2" in tier_str:
        return "qf-chip-tier-2"
    if "Tier 3" in tier_str:
        return "qf-chip-tier-3"
    if "Tier 4" in tier_str:
        return "qf-chip-tier-4"
    if "Noise" in tier_str:
        return "qf-chip-noise"
    return "qf-chip-muted"


def chip(label: object, cls: str = "qf-chip-muted") -> str:
    text = short_text(label, 96)
    if not text:
        return ""
    return f'<span class="qf-chip {cls}">{e(text)}</span>'


def compact_html(markup: str) -> str:
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def render_html(markup: str) -> None:
    st.html(compact_html(markup))


TIER_META = {
    1: ("t1", "Must-read"),
    2: ("t2", "Useful"),
    3: ("t3", "Background"),
    4: ("noise", "Manual review"),
}


def tier_number(tier: object) -> int:
    tier_str = str(tier or "")
    if "Tier 1" in tier_str:
        return 1
    if "Tier 2" in tier_str:
        return 2
    if "Tier 3" in tier_str:
        return 3
    return 4


def tier_badge(tier: object) -> str:
    cls, label = TIER_META.get(tier_number(tier), TIER_META[4])
    return f'<span class="badge {cls}"><span class="b-dot"></span>{label}</span>'


def paper_url(row: pd.Series) -> str:
    url = short_text(row.get("url"), 500)
    pmid = row.get("pmid", "")
    if not url and has_text(pmid):
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return url


def score_value(row: pd.Series) -> int:
    value = row.get("final_score", row.get("total_score", 0))
    if value is None or pd.isna(value):
        return 0
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def score_mini(score: int) -> str:
    return (
        f'<div class="score-mini"><div class="score-text">{score}</div>'
        f'<div class="progress"><i style="width: {score}%"></i></div></div>'
    )


def paper_meta(row: pd.Series) -> str:
    bits = [
        row.get("journal", ""),
        row.get("year", ""),
        row.get("study_design", ""),
        row.get("publication_type", ""),
    ]
    return " ".join(
        f"<span>{e(short_text(bit, 72))}</span>"
        for bit in bits
        if has_text(bit)
    )


def pipeline_confidence(result: dict, df: pd.DataFrame) -> int:
    accepted = len(df)
    deduped = int(result.get("deduped_count", accepted) or 0)
    missing_expected = len(result.get("missing_expected", []) or [])
    errors = len(result.get("errors", []) or [])
    rejected = len(result.get("rejected_unverified", []) or [])
    if not accepted:
        return 0
    confidence = 92
    if deduped and rejected > accepted:
        confidence -= 8
    confidence -= min(25, missing_expected * 8)
    confidence -= min(25, errors * 10)
    return max(20, min(98, confidence))


def context_chip(label: object, kind: str = "muted") -> str:
    text = short_text(label, 110)
    if not text:
        return ""
    return f'<span class="chip {kind}">{e(text)}</span>'


def result_source_count(result: dict) -> int:
    layer_count = len(result.get("layers", []) or [])
    discovery = result.get("api_discovery", {}) or {}
    api_sources = {
        short_text(source.get("source"), 80)
        for source in discovery.get("sources", []) or []
        if isinstance(source, dict) and has_text(source.get("source"))
    }
    return layer_count + len(api_sources)


def render_mini_stats(items: list[tuple[str, object, str]]) -> None:
    cards = []
    for label, value, delta in items:
        cards.append(
            f"""
            <div class="mini-stat">
              <div class="label">{e(label)}</div>
              <div class="num">{e(fmt_int(value))}</div>
              <div class="delta">{e(delta)}</div>
            </div>
            """
        )
    render_html(f'<div class="mini-stats">{"".join(cards)}</div>')


def render_app_header() -> None:
    render_html(
        f"""
        <div class="page-head">
          <div class="eyebrow">{e(APP_TAGLINE)}</div>
          <h1 class="title">{e(APP_NAME)}</h1>
          <p class="lede">
            {e(APP_PREVIEW_DESCRIPTION)}
          </p>
        </div>
        """
    )


def normalize_search_purpose_label(search_purpose_label: str) -> str:
    if search_purpose_label in SEARCH_MODE_UI_COPY:
        return search_purpose_label
    return SEARCH_PURPOSE_INTERNAL_TO_UI.get(search_purpose_label, SEARCH_PURPOSE_UI_DEFAULT)


def internal_search_purpose(search_purpose_label: str) -> str:
    label = normalize_search_purpose_label(search_purpose_label)
    return SEARCH_PURPOSE_UI_TO_INTERNAL[label]


def display_search_purpose(search_purpose: str) -> str:
    return normalize_search_purpose_label(search_purpose)


def migrate_legacy_search_purpose_state() -> None:
    legacy_value = st.session_state.get("search_purpose")
    if legacy_value and "search_purpose_label" not in st.session_state:
        st.session_state["search_purpose_label"] = normalize_search_purpose_label(str(legacy_value))
    if "search_purpose" in st.session_state:
        del st.session_state["search_purpose"]


def render_mode_guide(search_purpose_label: str) -> None:
    selected_label = normalize_search_purpose_label(search_purpose_label)
    cards: list[str] = []
    for label, copy in SEARCH_MODE_UI_COPY.items():
        state_class = " selected" if label == selected_label else ""
        state = "Selected" if label == selected_label else "Mode"
        cards.append(
            f"""
            <div class="mode-card{state_class}">
              <div class="mode-card-state">{e(state)}</div>
              <div class="mode-card-title">{e(label)}</div>
              <div class="mode-card-copy">{e(copy["description"])}</div>
            </div>
            """
        )
    render_html(
        f'<div class="mode-guide">{"".join(cards)}</div>',
    )


def render_sidebar_intro() -> None:
    render_html(
        """
        <div class="rail-card">
          <div class="rail-kicker">Clinical context</div>
          <div class="rail-title">Optional refinements</div>
          <div class="rail-copy">Add PICO details only when they sharpen the question. CorePapers handles source discovery automatically.</div>
        </div>
        """
    )


def app_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def render_search_form() -> tuple[str, str, dict, bool]:
    migrate_legacy_search_purpose_state()
    render_html(
        """
        <div class="search-shell">
          <div class="section-kicker">New search</div>
          <div class="search-title">Choose a purpose, then enter the clinical question.</div>
          <div class="search-copy">CorePapers will choose retrieval depth, sections, and ranking behavior from the purpose you select.</div>
        </div>
        """
    )
    search_purpose_label = st.segmented_control(
        "Search purpose",
        options=SEARCH_PURPOSE_UI_OPTIONS,
        default=SEARCH_PURPOSE_UI_DEFAULT,
        required=True,
        help="Choose why you are searching. The app selects retrieval depth and ranking emphasis.",
        width="stretch",
        key="search_purpose_label",
    ) or SEARCH_PURPOSE_UI_DEFAULT
    search_purpose_label = normalize_search_purpose_label(search_purpose_label)
    search_purpose = internal_search_purpose(search_purpose_label)
    purpose_config = search_purpose_config(search_purpose)
    render_mode_guide(search_purpose_label)
    with st.form("corepapers_search", clear_on_submit=False):
        topic = st.text_area(
            "Topic or PICO question",
            placeholder="cerebral venous thrombosis in adults; anticoagulation and recurrence",
            height=128,
            help="Use a disease, clinical question, exposure, complication, or rare presentation.",
            key="topic_query",
        )
        submitted = st.form_submit_button("Run evidence search", type="primary", use_container_width=True)
    return topic, search_purpose, purpose_config, submitted


def render_advanced_sidebar() -> tuple[str, str, str, str, str, str, str, str, str, object]:
    google_notes = ""
    email = app_secret("ncbi_email") or app_secret("contact_email") or app_secret("email")
    ncbi_api_key = app_secret("ncbi_api_key")
    gemini_api_key = app_secret("gemini_api_key")
    quartile_file = None
    with st.sidebar:
        render_sidebar_intro()
        with st.expander("PICO details", expanded=False):
            question_type = st.selectbox(
                "Question type",
                [
                    "General evidence map",
                    "Intervention or treatment",
                    "Diagnosis",
                    "Prognosis or prediction",
                    "Implementation or cost",
                ],
            )
            population = st.text_input("Population", placeholder="Adults, ICU, pregnancy")
            intervention = st.text_input("Intervention or exposure", placeholder="Hydrocortisone")
            comparator = st.text_input("Comparator", placeholder="Placebo or usual care")
            outcome = st.text_input("Outcome", placeholder="Mortality, recurrence")
    return (
        question_type,
        population,
        intervention,
        comparator,
        outcome,
        google_notes,
        email,
        ncbi_api_key,
        gemini_api_key,
        quartile_file,
    )


def main() -> None:
    render_app_header()

    topic, search_purpose, purpose_config, submitted = render_search_form()
    (
        question_type,
        population,
        intervention,
        comparator,
        outcome,
        google_notes,
        email,
        ncbi_api_key,
        gemini_api_key,
        quartile_file,
    ) = render_advanced_sidebar()

    if submitted:
        if not topic.strip():
            st.warning("Enter a research topic or question.")
            return
        quartile_overrides = load_quartile_file(quartile_file)
        context_kwargs = {
            "topic": topic.strip(),
            "population": population.strip(),
            "intervention": intervention.strip(),
            "comparator": comparator.strip(),
            "outcome": outcome.strip(),
            "question_type": question_type,
            "search_purpose": search_purpose,
        }
        # Streamlit Cloud can briefly hot-reload app.py while retaining an older
        # imported paper_finder module during deploy.
        context_fields = getattr(SearchContext, "__dataclass_fields__", {})
        if "search_purpose" not in context_fields:
            context_kwargs.pop("search_purpose", None)
        if "gemini_api_key" in context_fields:
            context_kwargs["gemini_api_key"] = gemini_api_key
        context = SearchContext(**context_kwargs)
        with st.status("Preparing verified source search...", expanded=True) as status:
            status.write("- Building search layers and topic gates")
            def report_progress(message: str, completed: int, total: int) -> None:
                if total:
                    status.update(label=f"Searching sources ({completed}/{total} layers) - {message}")
                else:
                    status.update(label=message)
                status.write(f"- {message}")

            result = run_quality_first_search(
                context=context,
                max_results_per_layer=int(purpose_config["candidate_depth"]),
                email=email.strip(),
                use_openalex=True,
                use_semantic_scholar=bool(purpose_config["semantic_scholar"]),
                enrichment_limit=int(purpose_config["enrichment_limit"]),
                quartile_overrides=quartile_overrides,
                manual_google_scholar_notes=google_notes,
                progress_callback=report_progress,
                ncbi_api_key=ncbi_api_key,
            )
            status.update(
                label=f"Done — {len(result['papers'])} papers admitted",
                state="complete",
                expanded=False,
            )
        st.session_state["result"] = result
        st.session_state["last_topic"] = context.topic
        st.rerun()

    result = st.session_state.get("result")
    if not result:
        render_start_state()
        return

    papers = result["papers"]
    df = pd.DataFrame(papers)
    display_df = safe_columns(df, DISPLAY_COLUMNS)
    full_df = safe_columns(df, FULL_COLUMNS)

    topic = st.session_state.get("last_topic", "")
    render_results_header(result, df, topic)
    render_empty_source_state(result, df)
    render_metrics(result, df, topic)
    render_missing_landmarks(result)
    render_errors(result)
    render_api_discovery(result)

    with st.expander("Search layers", expanded=False):
        for layer in result["layers"]:
            st.markdown(f"**{layer.name}** - {layer.purpose} Target: {layer.retmax} candidates.")
            st.code(layer.query, language="text")

    tabs = st.tabs(
        [
            "Papers",
            "Evidence review",
            "Expected papers",
            "Knowledge summary",
            "Gap map",
            "Exports",
        ]
    )

    with tabs[0]:
        render_mode_sections(result, df, full_df)

    with tabs[1]:
        render_evidence_review(result)

    with tabs[2]:
        render_expected_papers(result)

    with tabs[3]:
        render_knowledge_summary(result["summary"])

    with tabs[4]:
        render_gap_map(result.get("gap_map", []), result.get("subtopic_coverage", []))

    with tabs[5]:
        render_exports(full_df, display_df)


def render_start_state() -> None:
    render_html(
        """
        <div class="empty-panel">
          <div class="empty-title">Start with a clinical topic or PICO question.</div>
          <div class="empty-body">
            Choose the search purpose first. The app will adjust retrieval, ranking, sections,
            and tier logic for learning, research-gap work, exhaustive screening, or rare case finding.
          </div>
        </div>
        """
    )


def render_results_header(result: dict, df: pd.DataFrame, topic: str) -> None:
    search_mode = result.get("search_purpose") or result.get("search_mode") or "Search"
    search_mode_label = display_search_purpose(search_mode)
    effective_topic = result.get("topic_used", topic)
    search_date = result.get("search_date", "")
    accepted = len(df)
    tier_1 = int((df.get("tier") == "Tier 1: Must-read").sum()) if accepted else 0
    sections = int(df.get("reading_section", pd.Series(dtype=str)).nunique()) if accepted else 0
    meta = [f"{fmt_int(accepted)} admitted papers", f"{fmt_int(tier_1)} must-read", f"{fmt_int(sections)} sections"]
    if search_date:
        meta.append(f"searched {search_date}")

    chips: list[str] = [
        context_chip(f"Mode: {search_mode_label}", "accent"),
        context_chip(f"{fmt_int(accepted)} admitted", "muted"),
        context_chip(f"{fmt_int(tier_1)} Tier 1", "warn" if tier_1 else "muted"),
        context_chip(f"{fmt_int(result_source_count(result))} source paths", "muted"),
    ]
    expanded = result.get("topic_expanded", "")
    original = result.get("topic_original", topic)
    if expanded:
        chips.append(context_chip(f'Expanded "{original}" to "{expanded}"', "warn"))

    profile = topic_profile(effective_topic) if effective_topic else None
    primer_status = result.get("topic_primer_status", "")
    if profile:
        is_primed = bool(profile.get("_primed"))
        label_prefix = "AI topic primer" if is_primed else "Topic profile"
        chips.append(
            context_chip(
                f'{label_prefix}: {profile.get("display_name", profile.get("key", ""))}',
                "warn" if is_primed else "accent",
            )
        )
        expected_count = len(profile.get("expected_papers", []))
        if expected_count:
            chips.append(context_chip(f"{expected_count} expected papers checked", "accent"))
        if is_primed and primer_status == "cached":
            chips.append(context_chip("Primer cached this session", "muted"))
    elif primer_status == "unavailable":
        chips.append(context_chip("Primer unavailable: add Gemini key in Advanced", "muted"))

    mesh_records = result.get("mesh_discovered", []) or []
    descriptor_names = [
        (record.get("descriptor") or "").strip()
        for record in mesh_records
        if (record.get("descriptor") or "").strip()
    ]
    if descriptor_names:
        synonym_total = sum(len(record.get("entry_terms", []) or []) for record in mesh_records)
        head = ", ".join(descriptor_names[:3])
        tail = "" if len(descriptor_names) <= 3 else f" +{len(descriptor_names) - 3} more"
        chips.append(context_chip(f"MeSH: {head}{tail} ({synonym_total} synonyms)", "accent"))

    discovery = result.get("api_discovery", {}) or {}
    api_pmids = discovery.get("pmids", []) or []
    if api_pmids:
        related_count = len(discovery.get("related_pmids", []) or [])
        related = f", {related_count} related" if related_count else ""
        chips.append(context_chip(f"API supervisor: {len(api_pmids)} PMIDs{related}", "accent"))

    render_html(
        f"""
        <div class="context-card">
          <div class="context-top">
            <div>
              <div class="eyebrow">Query context</div>
              <div class="query-title">{e(effective_topic or "Search results")}</div>
              <div class="context-meta">{e(" | ".join(meta))}</div>
            </div>
          </div>
          <div class="chip-row">{"".join(chips)}</div>
        </div>
        """
    )


def render_metrics(result: dict, df: pd.DataFrame, topic: str) -> None:
    accepted = len(df)
    retrieved = result.get("retrieved_count", accepted)
    deduped = result.get("deduped_count", accepted)
    top_tier = int((df.get("tier") == "Tier 1: Must-read").sum()) if accepted else 0
    missing_expected = len(result.get("missing_expected", []))
    rejected = len(result.get("rejected_unverified", []))
    confidence = pipeline_confidence(result, df)
    source_count = result_source_count(result)
    excluded_delta = f"{fmt_int(rejected)} excluded before scoring" if rejected else "topic gate and dedupe applied"
    missing_delta = "sanity check clear" if missing_expected == 0 else "needs manual add"
    if rejected:
        st.caption(f"{rejected} unverified records were excluded.")
    render_html(
        f"""
        <div class="funnel">
          <div class="funnel-flow">
            <div class="stage">
              <div class="label">Retrieved</div>
              <div class="num">{fmt_int(retrieved)}</div>
              <div class="delta">from {fmt_int(source_count)} source paths</div>
            </div>
            <div class="arrow">&rarr;</div>
            <div class="stage">
              <div class="label">Deduped</div>
              <div class="num">{fmt_int(deduped)}</div>
              <div class="delta">unique candidate records</div>
            </div>
            <div class="arrow">&rarr;</div>
            <div class="stage">
              <div class="label">Accepted</div>
              <div class="num">{fmt_int(accepted)}</div>
              <div class="delta">{e(excluded_delta)}</div>
            </div>
            <div class="arrow">&rarr;</div>
            <div class="stage">
              <div class="label">Tier 1</div>
              <div class="num">{fmt_int(top_tier)}</div>
              <div class="delta">must-read papers</div>
            </div>
            <div class="arrow">&rarr;</div>
            <div class="stage">
              <div class="label">Expected Missing</div>
              <div class="num">{fmt_int(missing_expected)}</div>
              <div class="delta">{e(missing_delta)}</div>
            </div>
          </div>
          <div class="funnel-foot">
            <span>Pipeline confidence</span>
            <div class="progress"><i style="width: {confidence}%"></i></div>
            <span>{confidence}%</span>
          </div>
        </div>
        """
    )


def render_missing_landmarks(result: dict) -> None:
    missing = result.get("missing_expected", []) or []
    expected_total = len(result.get("expected_papers", []) or [])
    if not missing or expected_total == 0:
        return
    plural = "s" if len(missing) > 1 else ""
    titles = ", ".join(item.get("title", "(untitled)")[:80] for item in missing[:3])
    suffix = "" if len(missing) <= 3 else f" + {len(missing) - 3} more"
    st.warning(
        f"**Missing expected landmark paper{plural}** "
        f"({len(missing)}/{expected_total}): {titles}{suffix}. "
        f"See the **Missing expected** tab for full list."
    )


def render_errors(result: dict) -> None:
    errors = result.get("errors", [])
    if not errors:
        return
    with st.expander("Source errors", expanded=True):
        for error in errors:
            render_html(f'<div class="qf-error">{e(error)}</div>')


def render_api_discovery(result: dict) -> None:
    discovery = result.get("api_discovery", {}) or {}
    sources = discovery.get("sources", []) or []
    pmids = discovery.get("pmids", []) or []
    if not sources and not pmids:
        return

    with st.expander("API discovery supervisor", expanded=False):
        st.caption(
            "Verified candidate PMIDs gathered before scoring from PubMed, Europe PMC, Crossref, "
            "OpenAlex, Semantic Scholar, Unpaywall, ClinicalTrials.gov, medRxiv/bioRxiv, "
            "and PubMed related-article expansion."
        )
        render_mini_stats(
            [
                ("API PMIDs", len(pmids), "direct verified candidates"),
                ("Related PMIDs", len(discovery.get("related_pmids", []) or []), "related-article expansion"),
                ("API queries", len(sources), "supervisor search paths"),
            ]
        )

        source_df = pd.DataFrame(sources)
        if not source_df.empty:
            for list_col in ["pmids", "dois", "trial_ids"]:
                if list_col not in source_df.columns:
                    continue
                source_df[list_col] = source_df[list_col].apply(
                    lambda values: ", ".join(str(value) for value in values)
                    if isinstance(values, list)
                    else str(values)
                )
            show_cols = [
                col
                for col in ["source", "count", "doi_count", "trial_count", "query", "pmids", "dois", "trial_ids"]
                if col in source_df.columns
            ]
            st.dataframe(
                source_df[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "source": st.column_config.TextColumn("Source", width="medium"),
                    "count": st.column_config.NumberColumn("PMIDs", width="small"),
                    "doi_count": st.column_config.NumberColumn("DOIs", width="small"),
                    "trial_count": st.column_config.NumberColumn("Trials", width="small"),
                    "query": st.column_config.TextColumn("Query / seed", width="large"),
                    "pmids": st.column_config.TextColumn("PMIDs", width="medium"),
                    "dois": st.column_config.TextColumn("DOIs", width="medium"),
                    "trial_ids": st.column_config.TextColumn("Trial IDs", width="medium"),
                },
            )
        api_messages = (discovery.get("warnings", []) or []) + (discovery.get("errors", []) or [])
        for message in api_messages:
            st.caption(message)


def render_empty_source_state(result: dict, df: pd.DataFrame) -> None:
    if not df.empty:
        return
    errors = result.get("errors", [])
    if errors:
        title = "No verified papers were returned."
        body = (
            "The live literature sources were not reachable from this app process. "
            "Source errors are shown below."
        )
    else:
        title = "No verified papers were returned."
        body = "Broaden the topic, try Deep search mode, or add more PICO detail."
    render_html(
        f"""
        <div class="empty-panel">
          <div class="empty-title">{e(title)}</div>
          <div class="empty-body">{e(body)}</div>
        </div>
        """
    )


def render_paper_table(
    table_df: pd.DataFrame,
    empty_message: str,
    full_df: pd.DataFrame | None = None,
    tier_filter: bool = False,
    key: str = "tbl",
) -> None:
    del full_df
    if table_df.empty:
        st.warning(empty_message)
        return

    filtered = table_df
    if tier_filter and "tier" in table_df:
        tier_options = list(dict.fromkeys(table_df["tier"].dropna().tolist()))
        filter_col, count_col = st.columns([3, 1])
        with filter_col:
            selected_tiers = st.multiselect(
                "Tier filter", tier_options, default=tier_options, key=f"{key}_tier_filter"
            )
        with count_col:
            render_html(f'<div class="qf-section-caption">{len(table_df)} records</div>')
        filtered = table_df[table_df["tier"].isin(selected_tiers)] if selected_tiers else table_df

    rows_html: list[str] = []
    for _, row in filtered.iterrows():
        title = first_text(row.get("title"), "(untitled)")
        url = paper_url(row)
        title_html = e(title)
        if url:
            title_html = f'<a class="paper-link" href="{e(url)}" target="_blank" rel="noopener noreferrer">{e(title)}</a>'
        note = first_text(
            row.get("why_related"),
            row.get("purpose_fit_reason"),
            row.get("reason_for_tier"),
            row.get("topic_match_reason"),
        )
        journal_year = " | ".join(
            e(bit)
            for bit in [
                short_text(row.get("journal"), 54),
                short_text(row.get("year"), 12),
                short_text(row.get("relation_type"), 42),
            ]
            if has_text(bit)
        )
        citation_value = row.get("citation_count")
        citation_display = "NA"
        if citation_value is not None and not pd.isna(citation_value):
            citation_display = fmt_int(citation_value)
        pmid = short_text(row.get("pmid"), 24)
        link = ""
        if url:
            link = f'<a class="paper-link" href="{e(url)}" target="_blank" rel="noopener noreferrer">Open</a>'
        rows_html.append(
            f"""
            <tr>
              <td>
                <div class="table-title">{title_html}</div>
                <div class="table-note">{journal_year}</div>
                <div class="table-note">{e(short_text(note, 180))}</div>
                <div class="pmid">{e("PMID " + pmid if pmid else "")}</div>
              </td>
              <td>{tier_badge(row.get("tier"))}</td>
              <td>{e(short_text(first_text(row.get("study_design"), row.get("publication_type"), "Unclassified"), 54))}</td>
              <td>{e(short_text(row.get("year"), 12))}</td>
              <td>{e(citation_display)}</td>
              <td>{score_mini(score_value(row))}</td>
              <td>{link}</td>
            </tr>
            """
        )

    render_html(
        f"""
        <div class="table-wrap">
          <table class="papers-table">
            <thead>
              <tr>
                <th>Paper</th>
                <th>Tier</th>
                <th>Design</th>
                <th>Year</th>
                <th>Citations</th>
                <th>Score</th>
                <th>Link</th>
              </tr>
            </thead>
            <tbody>{"".join(rows_html)}</tbody>
          </table>
        </div>
        """
    )


def render_mode_sections(result: dict, df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No papers were admitted.")
        return
    search_mode = result.get("search_purpose") or result.get("search_mode") or ""
    search_mode_label = display_search_purpose(search_mode)
    sections = section_order_for_mode(search_mode)
    discovered = [section for section in df.get("reading_section", pd.Series(dtype=str)).dropna().unique()]
    ordered_sections = [section for section in sections if section in discovered]
    ordered_sections.extend(section for section in discovered if section not in ordered_sections)

    st.caption(f"{len(df)} papers grouped for **{search_mode_label or 'selected search mode'}**. Ranking and tiers change with this purpose.")
    render_top_paper_cards(full_df, limit=3)
    render_section_overview(df, ordered_sections)

    for index, section in enumerate(ordered_sections):
        section_df = section_rows(df, section, DISPLAY_COLUMNS, limit=500)
        if section_df.empty:
            continue
        tier_1 = int((section_df.get("tier") == "Tier 1: Must-read").sum()) if "tier" in section_df else 0
        label = f"{section} ({len(section_df)} papers"
        if tier_1:
            label += f", {tier_1} Tier 1"
        label += ")"
        with st.expander(label, expanded=index < 2):
            render_paper_table(
                section_df,
                f"No papers in {section}.",
                full_df=full_df,
                tier_filter=True,
                key=f"tbl_mode_{index}",
            )


def render_top_paper_cards(full_df: pd.DataFrame, limit: int = 3) -> None:
    if full_df.empty:
        return
    render_html('<div class="section-kicker">Top ranked papers</div>')
    cards: list[str] = []
    for rank, (_, row) in enumerate(full_df.head(limit).iterrows(), start=1):
        title = short_text(row.get("title", "(untitled)"), 190) or "(untitled)"
        url = paper_url(row)
        title_html = e(title)
        if url:
            title_html = f'<a href="{e(url)}" target="_blank" rel="noopener noreferrer">{e(title)}</a>'
        badges = [tier_badge(row.get("tier"))]
        relation = row.get("relation_type")
        if has_text(relation):
            badges.append(context_chip(relation, "accent"))
        section = row.get("reading_section")
        if has_text(section):
            badges.append(context_chip(section, "muted"))
        publication_type = row.get("publication_type")
        if has_text(publication_type):
            badges.append(context_chip(publication_type, "muted"))
        score = score_value(row)
        why = first_text(row.get("why_related"), row.get("reason_for_tier"), row.get("topic_match_reason"))
        cards.append(
            f"""
            <div class="paper">
              <div class="rank">No. {rank}</div>
              <div>
                <div class="paper-title">{title_html}</div>
                <div class="paper-meta">{paper_meta(row)}</div>
                <div class="chip-row">{"".join(badges)}</div>
                <div class="paper-why">{e(short_text(why, 260))}</div>
              </div>
              <div class="score-col">
                <div class="score">{score}</div>
                <div class="score-label">Composite</div>
                {score_mini(score)}
              </div>
            </div>
            """
        )
    render_html(f'<div class="top-papers">{"".join(cards)}</div>')


def render_section_overview(df: pd.DataFrame, ordered_sections: list[str]) -> None:
    if df.empty or not ordered_sections:
        return
    tiles = []
    for section in ordered_sections:
        rows = df[df["reading_section"] == section] if "reading_section" in df else pd.DataFrame()
        if rows.empty:
            continue
        tier_1 = int((rows.get("tier") == "Tier 1: Must-read").sum()) if "tier" in rows else 0
        designs = rows.get("study_design", pd.Series(dtype=str)).dropna()
        top_design = designs.mode().iloc[0] if not designs.empty else ""
        meta_bits = [f"{len(rows)} papers"]
        if tier_1:
            meta_bits.append(f"{tier_1} Tier 1")
        if top_design:
            meta_bits.append(short_text(top_design, 42))
        tiles.append(
            f"""
            <div class="qf-section-tile">
              <div class="qf-section-tile-title">{e(section)}</div>
              <div class="qf-section-tile-meta">{e(' | '.join(meta_bits))}</div>
            </div>
            """
        )
    if tiles:
        render_html('<div class="qf-section-caption">Section overview</div>')
        render_html(f'<div class="qf-section-grid">{"".join(tiles)}</div>')


def section_order_for_mode(search_mode: str) -> list[str]:
    normalized_mode = internal_search_purpose(search_mode)
    return {
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
            "Editorials/correspondence",
            "Low-priority/background papers",
        ],
        SEARCH_PURPOSE_RARE: [
            "Closest matching case reports",
            "Case series",
            "Rare complications",
            "Rare associations",
            "Unusual diagnostic findings",
            "Editorials/correspondence",
            "Background references",
            "Tier 4 / weak but related papers",
        ],
    }.get(normalized_mode, [])


def render_gap_map(gaps: list[dict], coverage: list[dict] | None = None) -> None:
    if coverage:
        covered = sum(1 for c in coverage if c.get("covered"))
        total = len(coverage)
        render_html('<div class="qf-section-caption">Subtopic coverage</div>')
        st.caption(f"{covered}/{total} subtopics from the topic profile have at least one matching paper.")
        chips: list[str] = []
        for item in coverage:
            cls = "qf-chip-green" if item.get("covered") else "qf-chip-amber"
            mark = "covered" if item.get("covered") else "missing"
            chips.append(
                chip(f'{mark}: {item.get("name", "")}', cls)
            )
        render_html("".join(chips))
        st.markdown("---")

    if not gaps:
        st.info("No gap signals were generated.")
        return

    priority_order = {"High": 0, "Medium": 1, "Low": 2, "": 3}
    sorted_gaps = sorted(gaps, key=lambda g: priority_order.get(g.get("Priority", ""), 3))

    render_html('<div class="qf-section-caption">Gap signals</div>')
    st.caption(f"{len(sorted_gaps)} gap signal(s). Subtopic gaps appear when the profile defines them and no matching paper was found.")

    for gap in sorted_gaps:
        gap_type = gap.get("Gap type", "Gap")
        statement = gap.get("Gap statement", "")
        why = gap.get("Why it matters", "")
        design = gap.get("Best study design", "")
        feasibility = gap.get("Feasibility in ICU/network", "")
        priority = gap.get("Priority", "")
        priority_class = {
            "High": "qf-chip-blue",
            "Medium": "qf-chip-amber",
            "Low": "qf-chip-muted",
        }.get(priority, "qf-chip-muted")
        chips = [
            chip(gap_type, "qf-chip-muted"),
        ]
        if priority:
            chips.append(chip(f"Priority: {priority}", priority_class))
        if feasibility:
            chips.append(chip(f"Feasibility: {feasibility}", "qf-chip-muted"))
        render_html(
            f"""
            <div class="qf-detail">
              <div style="margin-bottom: 0.45rem;">{''.join(chips)}</div>
              <div style="font-size: 1.02rem; font-weight: 500; margin-bottom: 0.4rem;">{e(statement)}</div>
              <div style="opacity: 0.85; margin-bottom: 0.4rem;">{e(why)}</div>
              <div style="opacity: 0.7; font-size: 0.88rem;"><strong>Best study design:</strong> {e(design)}</div>
            </div>
            """
        )


def render_paper_detail(row: pd.Series) -> None:
    title = first_text(row.get("title"), "(untitled)")
    journal = row.get("journal", "")
    year = row.get("year", "")
    authors = row.get("authors", "")
    design = row.get("study_design", "")
    citations = row.get("citation_count")
    quartile = row.get("quartile", "")
    pmid = row.get("pmid", "")
    doi = row.get("doi", "")
    abstract = row.get("abstract", "")
    publication_type = row.get("publication_type", "")

    header_bits = [
        journal if has_text(journal) else "",
        f"{year}" if has_text(year) else "",
        design if has_text(design) else "",
    ]
    chips: list[str] = []
    tier = row.get("tier")
    if has_text(tier):
        chips.append(chip(tier, tier_chip_class(tier)))
    gate = row.get("topic_match_gate")
    if has_text(gate):
        chips.append(chip(gate, "qf-chip-muted"))
    relation = row.get("relation_type")
    if has_text(relation):
        chips.append(chip(relation, "qf-chip-green"))
    if has_text(quartile) and quartile not in ("quartile not verified", ""):
        chips.append(chip(quartile, "qf-chip-amber"))
    if has_text(publication_type):
        chips.append(chip(publication_type, "qf-chip-violet"))
    if citations is not None and not pd.isna(citations):
        chips.append(chip(f"{int(citations)} citations", "qf-chip-muted"))

    render_html(
        f"""
        <div class="qf-detail">
          <h4>{e(title)}</h4>
          <div style="opacity: 0.8; margin-bottom: 0.4rem;">{e(' | '.join(str(b) for b in header_bits if has_text(b)))}</div>
          <div style="margin-bottom: 0.6rem;">{''.join(chips)}</div>
          <div style="opacity: 0.85; font-size: 0.9rem; margin-bottom: 0.5rem;">{e(authors)}</div>
        </div>
        """
    )
    if has_text(abstract):
        st.markdown("**Abstract**")
        st.write(abstract)
    else:
        st.caption("No abstract available for this record.")

    diagnostic_bits: list[tuple[str, str]] = []
    for label, key in [
        ("Reason for tier", "reason_for_tier"),
        ("Goal fit", "purpose_fit_reason"),
        ("Ranking confidence", "ranking_confidence"),
        ("Why included", "why_included"),
        ("Topic gate reason", "topic_match_reason"),
        ("Landmark/review protection", "mandatory_review_reason"),
        ("Expected-paper reason", "expected_paper_reason"),
        ("API discovery", "api_discovery_reason"),
        ("Tier cap reason", "tier_cap_reason"),
        ("Gap suggested", "gap_suggested"),
        ("Verified by", "verification"),
    ]:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            diagnostic_bits.append((label, str(value)))
    if diagnostic_bits:
        with st.expander("Scoring diagnostics", expanded=False):
            for label, value in diagnostic_bits:
                st.markdown(f"**{label}** — {value}")

    link_bits: list[str] = []
    if has_text(pmid):
        link_bits.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
    if has_text(doi):
        link_bits.append(f"[DOI](https://doi.org/{doi})")
    if link_bits:
        st.markdown(" · ".join(link_bits))


def render_expected_papers(result: dict) -> None:
    recovered = pd.DataFrame(result.get("recovered_expected", []))
    missing_automatic = pd.DataFrame(result.get("missing_from_automatic", []))
    missing = pd.DataFrame(result.get("missing_expected", []))
    expected = pd.DataFrame(result.get("expected_papers", []))

    if expected.empty:
        st.info("No topic-specific expected-paper checklist is available for this topic yet.")
        return

    if missing.empty:
        st.success("Expected landmark/review papers were present after the sanity-check layer.")
    else:
        st.warning("These expected papers were still not retrieved automatically. Add them manually.")
        st.dataframe(missing, use_container_width=True, hide_index=True)

    if not recovered.empty:
        st.subheader("Recovered by sanity seed")
        st.dataframe(recovered, use_container_width=True, hide_index=True)

    if not missing_automatic.empty:
        st.subheader("Would have been missed by automatic layers")
        st.dataframe(missing_automatic, use_container_width=True, hide_index=True)


def render_knowledge_summary(summary: dict) -> None:
    sections = [
        ("What We Know", "what_we_know"),
        ("What Remains Uncertain", "what_remains_uncertain"),
        ("What Is Changing", "what_is_changing"),
        ("Clinical Usefulness", "clinical_usefulness"),
    ]
    for title, key in sections:
        st.subheader(title)
        for line in summary.get(key, []):
            st.markdown(f"- {line}")


def render_evidence_review(result: dict) -> None:
    review = result.get("evidence_review") or build_evidence_review(result)
    verification = review.get("verification", {}) or {}

    st.subheader("Medical Evidence Review")
    st.caption("Structured synthesis with source IDs, evidence hierarchy, verification caveats, and gaps.")

    render_mini_stats(
        [
            ("Sources reviewed", verification.get("records_reviewed", 0), "records in synthesis"),
            ("PMID verified", verification.get("pmid_verified", 0), "source identifiers present"),
            ("DOI present", verification.get("doi_present", 0), "citation links available"),
            ("Citation counts", verification.get("citation_counts_available", 0), "enriched records"),
        ]
    )

    blocked = verification.get("blocked_checks")
    if blocked and blocked != "None flagged by metadata pipeline.":
        st.warning(blocked)

    top_records = pd.DataFrame(review.get("top_relevant_papers", []))
    if not top_records.empty:
        top_display = top_records[
            [
                "source_id",
                "title",
                "journal",
                "year",
                "evidence_type",
                "tier",
                "confidence",
                "pmid",
                "doi",
            ]
        ].copy()
        top_display["year"] = pd.to_numeric(top_display["year"], errors="coerce").astype("Int64")
        render_html('<div class="qf-section-caption">Top relevant papers</div>')
        st.dataframe(
            top_display,
            use_container_width=True,
            hide_index=True,
            height=min(520, 56 + 34 * len(top_records)),
            column_config={
                "source_id": st.column_config.TextColumn("ID", width="small"),
                "title": st.column_config.TextColumn("Title", width="large"),
                "journal": st.column_config.TextColumn("Journal", width="medium"),
                "year": st.column_config.NumberColumn("Year", format="%d", width="small"),
                "evidence_type": st.column_config.TextColumn("Evidence type", width="medium"),
                "tier": st.column_config.TextColumn("Tier", width="small"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "pmid": st.column_config.TextColumn("PMID", width="small"),
                "doi": st.column_config.TextColumn("DOI", width="medium"),
            },
        )
    else:
        st.info("No review-eligible sources were admitted.")

    render_html('<div class="qf-section-caption">Major evidence buckets</div>')
    bucket_cols = st.columns(3)
    buckets = [
        ("Guidelines", review.get("major_guidelines", [])),
        ("Systematic reviews", review.get("major_systematic_reviews", [])),
        ("RCTs", review.get("major_randomized_trials", [])),
    ]
    for col, (title, records) in zip(bucket_cols, buckets):
        with col:
            st.markdown(f"**{title}**")
            if not records:
                st.caption("None retrieved.")
            for record in records[:5]:
                st.caption(f"{record.get('source_id')}: {record.get('title')}")

    hierarchy_df = pd.DataFrame(review.get("evidence_hierarchy", []))
    if not hierarchy_df.empty:
        hierarchy_df["example_sources"] = hierarchy_df["example_sources"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else str(values)
        )
        render_html('<div class="qf-section-caption">Evidence hierarchy</div>')
        st.dataframe(
            hierarchy_df[["hierarchy_rank", "evidence_type", "count", "example_sources"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "hierarchy_rank": st.column_config.NumberColumn("Rank", width="small"),
                "evidence_type": st.column_config.TextColumn("Evidence type", width="large"),
                "count": st.column_config.NumberColumn("Papers", width="small"),
                "example_sources": st.column_config.TextColumn("Examples", width="medium"),
            },
        )

    comparison_df = pd.DataFrame(review.get("source_comparison", []))
    if not comparison_df.empty:
        with st.expander("Source comparison matrix", expanded=False):
            st.dataframe(
                comparison_df[["source_id", "evidence_type", "key_role", "confidence", "caveats"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "source_id": st.column_config.TextColumn("ID", width="small"),
                    "evidence_type": st.column_config.TextColumn("Evidence type", width="medium"),
                    "key_role": st.column_config.TextColumn("Role", width="medium"),
                    "confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "caveats": st.column_config.TextColumn("Caveats", width="large"),
                },
            )

    gap_col, limit_col = st.columns(2)
    with gap_col:
        st.markdown("**Gaps**")
        for gap in review.get("gaps", [])[:10]:
            st.markdown(f"- {gap}")
    with limit_col:
        st.markdown("**Limitations / uncertainty**")
        for item in review.get("limitations", [])[:10]:
            st.markdown(f"- {item}")

    ai_gap = review.get("ai_gap_synthesis", {}) or {}
    ai_items = ai_gap.get("items", []) or []
    if ai_gap.get("status") not in {None, "", "not_requested"}:
        with st.expander("AI-assisted research gap hypotheses", expanded=bool(ai_items)):
            status = ai_gap.get("status", "")
            if status == "generated":
                st.caption(ai_gap.get("note", "Source-grounded AI gap hypotheses."))
            else:
                st.caption(ai_gap.get("note", status))
            if ai_items:
                ai_df = pd.DataFrame(ai_items)
                ai_df["source_ids"] = ai_df["source_ids"].apply(
                    lambda values: ", ".join(values) if isinstance(values, list) else str(values)
                )
                st.dataframe(
                    ai_df[["gap", "confidence", "suggested_design", "source_ids", "rationale", "limitations"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "gap": st.column_config.TextColumn("Gap hypothesis", width="large"),
                        "confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "suggested_design": st.column_config.TextColumn("Suggested design", width="medium"),
                        "source_ids": st.column_config.TextColumn("Sources", width="small"),
                        "rationale": st.column_config.TextColumn("Rationale", width="large"),
                        "limitations": st.column_config.TextColumn("Limitations", width="medium"),
                    },
                )

    with st.expander("Workflow and prompt pattern adapted from Feynman", expanded=False):
        for item in review.get("workflow", []):
            st.markdown(f"- **{item.get('stage')}** - {item.get('status')}")
        st.markdown("**Biomedical prompt structure**")
        for item in review.get("prompt_structure", []):
            st.markdown(f"- {item}")
        st.caption(review.get("license_notice", ""))

    markdown = review.get("markdown", "")
    if markdown:
        st.download_button(
            "Download evidence review Markdown",
            data=markdown.encode("utf-8"),
            file_name="quality_first_evidence_review.md",
            mime="text/markdown",
        )


def render_exports(full_df: pd.DataFrame, display_df: pd.DataFrame) -> None:
    if full_df.empty:
        st.warning("No exportable records.")
        return

    top_results = display_df.head(25)
    pmids = "\n".join(full_df["pmid"].dropna().astype(str).loc[lambda s: s.str.len() > 0].tolist())

    st.subheader("Full database")
    st.dataframe(
        full_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("PubMed"),
            "openalex_id": st.column_config.LinkColumn("OpenAlex"),
            "semantic_scholar_url": st.column_config.LinkColumn("Semantic Scholar"),
            "abstract": st.column_config.TextColumn("Abstract", width="large"),
        },
    )

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download full CSV",
        data=full_df.to_csv(index=False).encode("utf-8"),
        file_name="quality_first_full_database.csv",
        mime="text/csv",
    )
    col2.download_button(
        "Download top results CSV",
        data=top_results.to_csv(index=False).encode("utf-8"),
        file_name="quality_first_top_results.csv",
        mime="text/csv",
    )
    col3.download_button(
        "Download PMID list",
        data=pmids.encode("utf-8"),
        file_name="quality_first_pmids.txt",
        mime="text/plain",
    )


def safe_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in df.columns]
    return df[available].copy()


def section_rows(
    df: pd.DataFrame,
    section: str,
    columns: list[str],
    limit: int | None = None,
) -> pd.DataFrame:
    if df.empty or "reading_section" not in df.columns:
        return pd.DataFrame(columns=columns)
    rows = df[df["reading_section"] == section]
    if limit:
        rows = rows.head(limit)
    return safe_columns(rows, columns)


def relevant_rows(df: pd.DataFrame, columns: list[str], limit: int = 400) -> pd.DataFrame:
    if df.empty or "reading_section" not in df.columns:
        return pd.DataFrame(columns=columns)
    rows = df[df["reading_section"] == "Extended evidence base"].head(limit)
    return safe_columns(rows, columns)


def load_quartile_file(uploaded_file) -> dict[str, dict[str, str]]:
    if uploaded_file is None:
        return {}
    try:
        text = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.warning("Could not read quartile CSV as UTF-8. Quartile overrides were skipped.")
        return {}
    overrides = parse_quartile_overrides(text)
    if not overrides:
        st.warning("Quartile CSV had no usable journal/quartile rows.")
    return overrides


if __name__ == "__main__":
    main()
