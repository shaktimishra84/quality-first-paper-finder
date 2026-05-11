from __future__ import annotations

import pandas as pd
import streamlit as st

from paper_finder import (
    SearchContext,
    parse_quartile_overrides,
    run_quality_first_search,
    topic_profile,
)


st.set_page_config(
    page_title="Quality-First Paper Finder",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');

    :root {
        --qf-blue:    #60A5FA;
        --qf-red:     #F87171;
        --qf-amber:   #FBBF24;
        --qf-green:   #34D399;
        --qf-tier-1:  #FBBF24;
        --qf-tier-2:  #60A5FA;
        --qf-tier-3:  #94A3B8;
        --qf-tier-4:  #64748B;
        --qf-noise:   #F87171;
        --qf-muted:   rgba(160, 168, 180, 0.85);
        --qf-surface-border: rgba(148, 163, 184, 0.18);
    }

    html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stTextInput, .stTextArea {
        font-family: 'Fira Sans', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        font-feature-settings: "tnum" 1, "ss01" 1;
    }
    [data-testid="stDataFrame"] {
        font-family: 'Fira Sans', system-ui, sans-serif;
        font-feature-settings: "tnum" 1;
    }
    [data-testid="stMetricValue"], code {
        font-family: 'Fira Code', 'SFMono-Regular', Consolas, monospace;
        font-variant-numeric: tabular-nums;
    }

    .main .block-container {
        padding-top: 1.25rem;
        max-width: 1600px;
    }
    h1 { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.015em; }
    h2 { font-size: 1.35rem; font-weight: 600; letter-spacing: -0.005em; }
    h3 { font-size: 1.1rem; font-weight: 600; }
    h4 { font-size: 1rem; font-weight: 600; }

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--qf-surface-border);
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        color: var(--text-color);
        transition: border-color 180ms ease;
    }
    [data-testid="stMetric"]:hover { border-color: rgba(148, 163, 184, 0.35); }
    [data-testid="stMetricLabel"] { font-size: 0.74rem; opacity: 0.7; letter-spacing: 0.02em; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 500; }
    [data-testid="stMetricDelta"] svg { display: none; }

    .qf-rule {
        border-left: 4px solid var(--qf-blue);
        padding: 0.35rem 0 0.35rem 0.75rem;
        color: var(--text-color);
        font-size: 0.92rem;
    }
    .qf-error {
        border-left: 4px solid var(--qf-red);
        padding: 0.35rem 0 0.35rem 0.75rem;
    }
    .qf-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-bottom: 0.25rem;
        border: 1px solid currentColor;
        line-height: 1.4;
        transition: background-color 150ms ease;
    }
    .qf-chip-blue   { color: var(--qf-blue); }
    .qf-chip-amber  { color: var(--qf-amber); }
    .qf-chip-green  { color: var(--qf-green); }
    .qf-chip-muted  { color: var(--qf-muted); }
    .qf-chip-tier-1 { color: var(--qf-tier-1); background: rgba(251, 191, 36, 0.08); }
    .qf-chip-tier-2 { color: var(--qf-tier-2); background: rgba(96, 165, 250, 0.08); }
    .qf-chip-tier-3 { color: var(--qf-tier-3); }
    .qf-chip-tier-4 { color: var(--qf-tier-4); }
    .qf-chip-noise  { color: var(--qf-noise); background: rgba(248, 113, 113, 0.08); }

    .qf-section-caption {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.6;
        margin: 0.35rem 0 0.4rem 0;
        font-weight: 500;
    }
    .qf-detail {
        background: var(--secondary-background-color);
        border: 1px solid var(--qf-surface-border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }
    .qf-detail h4 { margin-top: 0; margin-bottom: 0.6rem; line-height: 1.35; }

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


DISPLAY_COLUMNS = [
    "pmid",
    "reading_section",
    "tier",
    "topic_match_gate",
    "total_score",
    "title",
    "journal",
    "year",
    "study_design",
    "citation_count",
    "mandatory_review_reason",
    "expected_paper_reason",
    "api_discovery_reason",
    "verification",
    "why_included",
    "topic_match_reason",
    "tier_cap_reason",
    "gap_suggested",
    "url",
]

VISIBLE_COLUMN_ORDER = [
    "reading_section",
    "tier",
    "title",
    "journal",
    "year",
    "study_design",
    "citation_count",
    "url",
]

FULL_COLUMNS = [
    "title",
    "normalized_title",
    "authors",
    "year",
    "journal",
    "quartile",
    "quartile_source",
    "study_design",
    "publication_type",
    "pmid",
    "doi",
    "url",
    "openalex_id",
    "semantic_scholar_url",
    "citation_count",
    "citation_count_missing",
    "citation_source",
    "clinical_relevance_score",
    "design_strength_score",
    "journal_quality_score",
    "citation_score",
    "recency_score",
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
    "gap_suggested",
    "relevance_reason",
    "recent_high_quality_note",
    "search_layers",
    "publication_types",
    "abstract",
]


def main() -> None:
    st.title("Quality-First Paper Finder")
    st.caption("Knowledge-base builder with landmark/review discovery and strict topic gates")

    search_expanded = st.session_state.get("result") is None
    with st.expander("Search setup", expanded=search_expanded):
        with st.form("search_form"):
            topic = st.text_area(
                "Research topic or question",
                placeholder="Example: cerebral venous thrombosis · vili · sepsis",
                height=74,
            )
            max_results = st.slider(
                "Candidate depth",
                min_value=25,
                max_value=200,
                value=100,
                step=5,
                help="Maximum candidates pulled per search layer (6 layers run in parallel).",
            )

            with st.expander("Advanced — PICO, infrastructure, enrichment", expanded=False):
                pico_col_1, pico_col_2, pico_col_3 = st.columns(3)
                with pico_col_1:
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
                with pico_col_2:
                    intervention = st.text_input("Intervention or exposure", placeholder="Hydrocortisone")
                    comparator = st.text_input("Comparator", placeholder="Placebo or usual care")
                with pico_col_3:
                    outcome = st.text_input("Outcome", placeholder="Mortality, recurrence")

                st.markdown("---")

                infra_col_1, infra_col_2, infra_col_3, infra_col_4 = st.columns(4)
                with infra_col_1:
                    email = st.text_input("NCBI email", placeholder="Optional")
                with infra_col_2:
                    secret_api_key = ""
                    try:
                        secret_api_key = str(st.secrets.get("ncbi_api_key", "") or "")
                    except Exception:
                        secret_api_key = ""
                    api_key_help = (
                        "Loaded from app secrets (st.secrets['ncbi_api_key'])."
                        if secret_api_key
                        else "Optional. Free at ncbi.nlm.nih.gov/account; raises rate limit 3→10 req/s."
                    )
                    api_key_field = st.text_input(
                        "NCBI API key",
                        placeholder="Loaded from app secrets" if secret_api_key else "Optional",
                        type="password",
                        help=api_key_help,
                    )
                    ncbi_api_key = (api_key_field or secret_api_key or "").strip()
                with infra_col_3:
                    secret_gemini_key = ""
                    try:
                        secret_gemini_key = str(st.secrets.get("gemini_api_key", "") or "")
                    except Exception:
                        secret_gemini_key = ""
                    gemini_help = (
                        "Loaded from app secrets (st.secrets['gemini_api_key'])."
                        if secret_gemini_key
                        else "Optional. Free at aistudio.google.com/apikey. Generates a topic primer for un-profiled topics."
                    )
                    gemini_field = st.text_input(
                        "Gemini API key",
                        placeholder="Loaded from app secrets" if secret_gemini_key else "Optional",
                        type="password",
                        help=gemini_help,
                    )
                    gemini_api_key = (gemini_field or secret_gemini_key or "").strip()
                with infra_col_4:
                    enrichment_limit = st.slider("Citation enrichment limit", 0, 150, 100, step=10)

                enrich_col_1, enrich_col_2, enrich_col_3 = st.columns(3)
                with enrich_col_1:
                    use_openalex = st.checkbox("OpenAlex citations", value=True)
                with enrich_col_2:
                    use_semantic = st.checkbox("Semantic Scholar check", value=False)
                with enrich_col_3:
                    quartile_file = st.file_uploader(
                        "Journal quartile CSV",
                        type=["csv"],
                        help="Optional columns: journal, quartile, quartile_source.",
                    )

                google_notes = st.text_area(
                    "Manual Google Scholar notes",
                    placeholder="Use only for cross-checking landmark or cited-by observations.",
                    height=70,
                )

            submitted = st.form_submit_button("Run search", type="primary", use_container_width=True)

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
        }
        # Streamlit Cloud can briefly hot-reload app.py while retaining an older
        # imported paper_finder module during deploy.
        if "gemini_api_key" in getattr(SearchContext, "__dataclass_fields__", {}):
            context_kwargs["gemini_api_key"] = gemini_api_key
        context = SearchContext(**context_kwargs)
        with st.status("Searching PubMed in parallel...", expanded=True) as status:
            def report_progress(message: str, completed: int, total: int) -> None:
                if total:
                    status.update(label=f"Searching PubMed ({completed}/{total} layers) — {message}")
                else:
                    status.update(label=message)
                status.write(f"- {message}")

            result = run_quality_first_search(
                context=context,
                max_results_per_layer=max_results,
                email=email.strip(),
                use_openalex=use_openalex,
                use_semantic_scholar=use_semantic,
                enrichment_limit=enrichment_limit,
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
        st.info("Enter a topic above and run a verified search.")
        return

    papers = result["papers"]
    df = pd.DataFrame(papers)
    display_df = safe_columns(df, DISPLAY_COLUMNS)
    full_df = safe_columns(df, FULL_COLUMNS)

    topic = st.session_state.get("last_topic", "")
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
            "Core reading pack",
            "Extended evidence base",
            "Low-priority / indirect",
            "Missing expected",
            "Knowledge summary",
            "Research gap map",
            "Exports",
        ]
    )

    with tabs[0]:
        core_df = section_rows(df, "Core reading pack", DISPLAY_COLUMNS, limit=300)
        st.caption(f"All on-topic papers Tier 1–3 (incl. landmark / expected-paper protected). Filter by tier below. {len(core_df)} papers.")
        render_paper_table(core_df, "No core reading-pack papers were admitted.", full_df=full_df, tier_filter=True, key="tbl_core")

    with tabs[1]:
        evidence_df = relevant_rows(df, DISPLAY_COLUMNS, limit=400)
        st.caption(f"On-topic Tier 4 plus partial-match papers. Use these to widen scope. {len(evidence_df)} papers.")
        render_paper_table(evidence_df, "No extended evidence-base papers were admitted.", full_df=full_df, tier_filter=True, key="tbl_evidence")

    with tabs[2]:
        low_df = section_rows(df, "Low-priority / indirect papers", DISPLAY_COLUMNS, limit=200)
        render_paper_table(low_df, "No low-priority or indirect papers were kept.", full_df=full_df, key="tbl_low")

    with tabs[3]:
        render_expected_papers(result)

    with tabs[4]:
        render_knowledge_summary(result["summary"])

    with tabs[5]:
        render_gap_map(result.get("gap_map", []), result.get("subtopic_coverage", []))

    with tabs[6]:
        render_exports(full_df, display_df)


def render_metrics(result: dict, df: pd.DataFrame, topic: str) -> None:
    accepted = len(df)
    retrieved = result.get("retrieved_count", accepted)
    deduped = result.get("deduped_count", accepted)
    core_candidates = int((df.get("reading_section") == "Core reading pack").sum()) if accepted else 0
    core_shown = min(core_candidates, 25)
    missing_expected = len(result.get("missing_expected", []))
    rejected = len(result.get("rejected_unverified", []))

    expanded = result.get("topic_expanded", "")
    original = result.get("topic_original", topic)
    effective_topic = result.get("topic_used", topic)
    profile = topic_profile(effective_topic) if effective_topic else None
    chips: list[str] = []
    if expanded:
        chips.append(
            f'<span class="qf-chip qf-chip-amber">Expanded "{original}" → "{expanded}"</span>'
        )
    primer_status = result.get("topic_primer_status", "")
    if profile:
        is_primed = bool(profile.get("_primed"))
        label_prefix = "Topic primer (AI)" if is_primed else "Topic profile"
        chip_class = "qf-chip-amber" if is_primed else "qf-chip-blue"
        chips.append(
            f'<span class="qf-chip {chip_class}">{label_prefix}: '
            f'{profile.get("display_name", profile.get("key", ""))}</span>'
        )
        expected_count = len(profile.get("expected_papers", []))
        if expected_count:
            chips.append(
                f'<span class="qf-chip qf-chip-green">{expected_count} expected papers checked</span>'
            )
        subtopic_count = len(profile.get("gap_subtopics", []))
        if subtopic_count:
            chips.append(
                f'<span class="qf-chip qf-chip-amber">{subtopic_count} subtopic gap probes</span>'
            )
        if is_primed and primer_status == "cached":
            chips.append('<span class="qf-chip qf-chip-muted">Primer cached this session</span>')
    else:
        if primer_status == "unavailable":
            chips.append(
                '<span class="qf-chip qf-chip-muted">No profile · primer unavailable '
                '(add Gemini key in Advanced)</span>'
            )
        else:
            chips.append('<span class="qf-chip qf-chip-muted">Generic topic — no profile loaded</span>')

    mesh_records = result.get("mesh_discovered", []) or []
    if mesh_records:
        descriptor_names = [
            (record.get("descriptor") or "").strip()
            for record in mesh_records
            if (record.get("descriptor") or "").strip()
        ]
        synonym_total = sum(len(record.get("entry_terms", []) or []) for record in mesh_records)
        if descriptor_names:
            head = ", ".join(descriptor_names[:3])
            tail = "" if len(descriptor_names) <= 3 else f" +{len(descriptor_names) - 3} more"
            chips.append(
                f'<span class="qf-chip qf-chip-green">MeSH: {head}{tail} ({synonym_total} synonyms)</span>'
            )
    api_discovery = result.get("api_discovery", {}) or {}
    api_pmids = api_discovery.get("pmids", []) or []
    if api_pmids:
        related_count = len(api_discovery.get("related_pmids", []) or [])
        chips.append(
            f'<span class="qf-chip qf-chip-green">API supervisor: {len(api_pmids)} PMIDs'
            f'{f" · {related_count} related" if related_count else ""}</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)

    funnel_col, reading_col = st.columns([3, 2])
    with funnel_col:
        st.markdown('<div class="qf-section-caption">Search funnel</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieved", retrieved)
        c2.metric(
            "Deduped",
            deduped,
            delta=(deduped - retrieved) if retrieved else None,
            delta_color="off",
        )
        c3.metric(
            "Accepted",
            accepted,
            delta=(accepted - deduped) if deduped else None,
            delta_color="off",
        )
    with reading_col:
        st.markdown('<div class="qf-section-caption">Reading set</div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        c4.metric("Core shown", core_shown)
        c5.metric(
            "Expected missing",
            missing_expected,
            delta=None if missing_expected == 0 else "needs manual add",
            delta_color="inverse",
        )
    if rejected:
        st.caption(f"{rejected} unverified records were excluded.")


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
            st.markdown(f'<div class="qf-error">{error}</div>', unsafe_allow_html=True)


def render_api_discovery(result: dict) -> None:
    discovery = result.get("api_discovery", {}) or {}
    sources = discovery.get("sources", []) or []
    pmids = discovery.get("pmids", []) or []
    if not sources and not pmids:
        return

    with st.expander("API discovery supervisor", expanded=False):
        st.caption(
            "Verified candidate PMIDs gathered before scoring from PubMed exact searches, "
            "Europe PMC, OpenAlex, and PubMed related-article expansion."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("API PMIDs", len(pmids))
        c2.metric("Related PMIDs", len(discovery.get("related_pmids", []) or []))
        c3.metric("API queries", len(sources))

        source_df = pd.DataFrame(sources)
        if not source_df.empty:
            if "pmids" in source_df.columns:
                source_df["pmids"] = source_df["pmids"].apply(
                    lambda values: ", ".join(str(value) for value in values)
                    if isinstance(values, list)
                    else str(values)
                )
            show_cols = [col for col in ["source", "count", "query", "pmids"] if col in source_df.columns]
            st.dataframe(
                source_df[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "source": st.column_config.TextColumn("Source", width="medium"),
                    "count": st.column_config.NumberColumn("PMIDs", width="small"),
                    "query": st.column_config.TextColumn("Query / seed", width="large"),
                    "pmids": st.column_config.TextColumn("PMIDs", width="medium"),
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
        st.error(
            "No verified papers were returned because the live literature sources were not reachable "
            "from this app process. The source errors are shown below."
        )
    else:
        st.warning(
            "No verified papers were returned for this search. Broaden the topic or increase candidate depth."
        )


def render_paper_table(
    table_df: pd.DataFrame,
    empty_message: str,
    full_df: pd.DataFrame | None = None,
    tier_filter: bool = False,
    key: str = "tbl",
) -> None:
    if table_df.empty:
        st.warning(empty_message)
        return

    filtered = table_df
    if tier_filter and "tier" in table_df:
        tier_options = list(dict.fromkeys(table_df["tier"].dropna().tolist()))
        selected_tiers = st.multiselect(
            "Tier filter", tier_options, default=tier_options, key=f"{key}_tier_filter"
        )
        filtered = table_df[table_df["tier"].isin(selected_tiers)] if selected_tiers else table_df

    visible_columns = [col for col in VISIBLE_COLUMN_ORDER if col in filtered.columns]
    event = st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=min(560, 56 + 36 * len(filtered)),
        on_select="rerun",
        selection_mode="single-row",
        column_order=visible_columns,
        key=key,
        column_config={
            "reading_section": st.column_config.TextColumn("Section", width="small"),
            "tier": st.column_config.TextColumn("Tier", width="small"),
            "topic_match_gate": st.column_config.TextColumn("Topic gate", width="small"),
            "url": st.column_config.LinkColumn("PubMed", width="small", display_text="open"),
            "total_score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "year": st.column_config.NumberColumn("Year", format="%d", width="small"),
            "journal": st.column_config.TextColumn("Journal", width="medium"),
            "study_design": st.column_config.TextColumn("Design", width="small"),
            "citation_count": st.column_config.NumberColumn("Citations", format="%d", width="small"),
            "mandatory_review_reason": st.column_config.TextColumn("Landmark/review protection", width="medium"),
            "expected_paper_reason": st.column_config.TextColumn("Expected-paper reason", width="medium"),
            "api_discovery_reason": st.column_config.TextColumn("API discovery reason", width="medium"),
            "verification": st.column_config.TextColumn("Verified by", width="small"),
            "why_included": st.column_config.TextColumn("Why included", width="medium"),
            "topic_match_reason": st.column_config.TextColumn("Topic gate reason", width="medium"),
            "tier_cap_reason": st.column_config.TextColumn("Tier cap reason", width="medium"),
            "gap_suggested": st.column_config.TextColumn("Gap suggested", width="medium"),
            "title": st.column_config.TextColumn("Title", width="large"),
        },
    )

    selected_rows = getattr(event.selection, "rows", []) if event and getattr(event, "selection", None) else []
    if selected_rows and full_df is not None and not full_df.empty:
        selected_idx = selected_rows[0]
        if 0 <= selected_idx < len(filtered):
            selected_pmid = str(filtered.iloc[selected_idx].get("pmid", ""))
            if selected_pmid:
                match = full_df[full_df["pmid"].astype(str) == selected_pmid]
                if not match.empty:
                    render_paper_detail(match.iloc[0])


def render_gap_map(gaps: list[dict], coverage: list[dict] | None = None) -> None:
    if coverage:
        covered = sum(1 for c in coverage if c.get("covered"))
        total = len(coverage)
        st.markdown('<div class="qf-section-caption">Subtopic coverage</div>', unsafe_allow_html=True)
        st.caption(f"{covered}/{total} subtopics from the topic profile have at least one matching paper.")
        chips: list[str] = []
        for item in coverage:
            cls = "qf-chip-green" if item.get("covered") else "qf-chip-amber"
            mark = "✓" if item.get("covered") else "—"
            chips.append(
                f'<span class="qf-chip {cls}">{mark} {item.get("name", "")}</span>'
            )
        st.markdown("".join(chips), unsafe_allow_html=True)
        st.markdown("---")

    if not gaps:
        st.info("No gap signals were generated.")
        return

    priority_order = {"High": 0, "Medium": 1, "Low": 2, "": 3}
    sorted_gaps = sorted(gaps, key=lambda g: priority_order.get(g.get("Priority", ""), 3))

    st.markdown('<div class="qf-section-caption">Gap signals</div>', unsafe_allow_html=True)
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
            f'<span class="qf-chip qf-chip-muted">{gap_type}</span>',
        ]
        if priority:
            chips.append(f'<span class="qf-chip {priority_class}">Priority: {priority}</span>')
        if feasibility:
            chips.append(f'<span class="qf-chip qf-chip-muted">Feasibility: {feasibility}</span>')
        st.markdown(
            f"""
            <div class="qf-detail">
              <div style="margin-bottom: 0.45rem;">{''.join(chips)}</div>
              <div style="font-size: 1.02rem; font-weight: 500; margin-bottom: 0.4rem;">{statement}</div>
              <div style="opacity: 0.85; margin-bottom: 0.4rem;">{why}</div>
              <div style="opacity: 0.7; font-size: 0.88rem;"><strong>Best study design:</strong> {design}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_paper_detail(row: pd.Series) -> None:
    title = row.get("title", "(untitled)")
    journal = row.get("journal", "")
    year = row.get("year", "")
    authors = row.get("authors", "")
    design = row.get("study_design", "")
    citations = row.get("citation_count")
    quartile = row.get("quartile", "")
    pmid = row.get("pmid", "")
    doi = row.get("doi", "")
    abstract = row.get("abstract", "")

    header_bits = [
        f"<strong>{journal}</strong>" if journal else "",
        f"{year}" if year else "",
        design or "",
    ]
    chips: list[str] = []
    tier = row.get("tier")
    if tier:
        tier_str = str(tier)
        if "Tier 1" in tier_str:
            cls = "qf-chip-tier-1"
        elif "Tier 2" in tier_str:
            cls = "qf-chip-tier-2"
        elif "Tier 3" in tier_str:
            cls = "qf-chip-tier-3"
        elif "Tier 4" in tier_str:
            cls = "qf-chip-tier-4"
        else:
            cls = "qf-chip-noise"
        chips.append(f'<span class="qf-chip {cls}">{tier}</span>')
    gate = row.get("topic_match_gate")
    if gate:
        chips.append(f'<span class="qf-chip qf-chip-muted">{gate}</span>')
    if quartile and quartile not in ("quartile not verified", ""):
        chips.append(f'<span class="qf-chip qf-chip-amber">{quartile}</span>')
    if citations is not None and not pd.isna(citations):
        chips.append(f'<span class="qf-chip qf-chip-muted">{int(citations)} citations</span>')

    st.markdown(
        f"""
        <div class="qf-detail">
          <h4>{title}</h4>
          <div style="opacity: 0.8; margin-bottom: 0.4rem;">{' · '.join(b for b in header_bits if b)}</div>
          <div style="margin-bottom: 0.6rem;">{''.join(chips)}</div>
          <div style="opacity: 0.85; font-size: 0.9rem; margin-bottom: 0.5rem;">{authors}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if abstract:
        st.markdown("**Abstract**")
        st.write(abstract)
    else:
        st.caption("No abstract available for this record.")

    diagnostic_bits: list[tuple[str, str]] = []
    for label, key in [
        ("Reason for tier", "reason_for_tier"),
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
    if pmid:
        link_bits.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
    if doi:
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

    notes = result.get("manual_google_scholar_notes", "")
    if notes:
        st.subheader("Manual Google Scholar notes")
        st.write(notes)


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


def render_exports(full_df: pd.DataFrame, display_df: pd.DataFrame) -> None:
    if full_df.empty:
        st.warning("No exportable records.")
        return

    core_pack = display_df[display_df["reading_section"] == "Core reading pack"].head(25)
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
        "Download core pack CSV",
        data=core_pack.to_csv(index=False).encode("utf-8"),
        file_name="quality_first_core_reading_pack.csv",
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
