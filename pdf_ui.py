from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from pdf_download import generate_download_zip
from pdf_finder import find_legal_pdf, get_pdf_status_label, PDFSearchResult
from pdf_storage import PDFMetadata, PDFStorage

# Maximum number of papers a user may select for download at once.
MAX_SELECTIONS = 50


def _resolve_secret(*names: str) -> str:
    """Resolve the first non-empty value from Streamlit secrets, then env."""
    for name in names:
        try:
            value = str(st.secrets.get(name, "") or "").strip()
        except Exception:
            value = ""
        if not value:
            value = (os.environ.get(name) or os.environ.get(name.upper()) or "").strip()
        if value:
            return value
    return ""


def _resolve_secret_email() -> str:
    """Resolve a contact email from secrets/env, matching the search path."""
    return _resolve_secret("ncbi_email", "contact_email", "email")


def get_s2_api_key() -> str:
    """Resolve the Semantic Scholar API key from secrets/env."""
    return _resolve_secret("semantic_scholar_api_key", "s2_api_key")


def get_ncbi_email() -> str:
    """Get the contact email used for Unpaywall.

    Prefers a value already configured in Streamlit secrets / env; only falls
    back to a sidebar input when none is configured.
    """
    configured = _resolve_secret_email()
    if configured:
        return configured

    email = st.sidebar.text_input(
        "NCBI Email (for Unpaywall)",
        value=st.session_state.get("ncbi_email", ""),
        type="password",
        help="Required for Unpaywall PDF lookup. Use your real email address.",
    )
    st.session_state["ncbi_email"] = email
    return email


def find_pdf_for_paper(
    pmid: str = "",
    doi: str = "",
    email: str = "",
) -> PDFSearchResult:
    """Find legal PDF using all available sources."""
    return find_legal_pdf(pmid=pmid, doi=doi, email=email)


def render_pdf_status_badge(result: PDFSearchResult) -> None:
    """Render PDF status as a badge."""
    if result.has_pdf:
        st.success(f"📄 {result.best_source.source if result.best_source else 'PDF available'}")
    elif result.oa_status == "green":
        st.info("🔗 Open access (landing page only)")
    else:
        st.error("❌ No legal PDF found")


def render_pdf_actions(
    row: pd.Series,
    topic: str,
    download_folder: Path,
    email: str,
) -> None:
    """Render PDF download UI for a single paper."""
    pmid = row.get("pmid", "")
    doi = row.get("doi", "")

    if not pmid and not doi:
        st.caption("⚠️ No PMID or DOI available")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button(
            "🔍 Find PDF",
            key=f"find_pdf_{pmid or doi}",
            help="Search legal open-access sources",
        ):
            result = find_pdf_for_paper(pmid=pmid, doi=doi, email=email)

            if result.has_pdf and result.best_source:
                st.session_state[f"pdf_result_{pmid or doi}"] = result
            else:
                st.warning(result.message)

    with col2:
        result = st.session_state.get(f"pdf_result_{pmid or doi}")
        if result and result.has_pdf:
            st.success("✓ Found")


def download_pdf_for_paper(
    row: pd.Series,
    topic: str,
    download_folder: Path,
    email: str,
) -> tuple[bool, str]:
    """Download PDF and save metadata for a paper."""
    pmid = str(row.get("pmid", ""))
    doi = str(row.get("doi", ""))

    result = find_pdf_for_paper(pmid=pmid, doi=doi, email=email)

    if not result.has_pdf:
        return False, result.message

    try:
        storage = PDFStorage(download_folder)

        metadata = PDFMetadata(
            title=str(row.get("title", "(untitled)")),
            authors=str(row.get("authors", "")),
            journal=str(row.get("journal", "")),
            year=str(row.get("year", "")),
            doi=doi,
            pmid=pmid,
            pmcid=str(row.get("pmcid", "")),
            source_of_pdf=result.best_source.source if result.best_source else "unknown",
            license=result.best_source.license if result.best_source else "unknown",
            downloaded_at=pd.Timestamp.now().isoformat(),
            search_query=str(st.session_state.get("last_search_query", "")),
            relevance_score=float(row.get("composite_score", 0)) if "composite_score" in row else None,
        )

        success, message, pdf_path = storage.save_pdf_with_metadata(
            result.best_source,
            metadata,
            topic,
            output_folder=storage.get_paper_folder(topic),
        )

        return success, message

    except Exception as e:
        return False, f"Error: {str(e)}"


def init_selection_state() -> None:
    """Initialize session state for paper selection."""
    if "selected_papers" not in st.session_state:
        st.session_state.selected_papers = {}


def _resolve_selected_rows(df: pd.DataFrame, selection_keys: list[str]) -> list[dict]:
    """Map selection keys (PMID, then DOI, then title) back to full paper rows."""
    rows: list[dict] = []
    for key in selection_keys:
        matching = df[df["pmid"].astype(str) == key]
        if matching.empty and "doi" in df:
            matching = df[df["doi"].astype(str) == key]
        if matching.empty:
            matching = df[df["title"].astype(str) == key]
        if not matching.empty:
            rows.append(matching.iloc[0].to_dict())
    return rows


def render_download_button(df: pd.DataFrame, topic: str, email: str) -> None:
    """Select papers, attempt to find their free PDFs, then download the ones found."""
    if df.empty:
        return

    init_selection_state()
    s2_api_key = get_s2_api_key()
    selected_keys = list(st.session_state.selected_papers.keys())
    selected_count = len(selected_keys)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.metric(
            f"Papers selected (max {MAX_SELECTIONS})",
            selected_count,
            delta=f"out of {len(df)}",
        )
    with col2:
        if st.button("Clear selection", use_container_width=True):
            st.session_state.selected_papers = {}
            for widget_key in [k for k in st.session_state if k.startswith("cb_")]:
                del st.session_state[widget_key]
            st.session_state.pop("pdf_attempt", None)
            st.rerun()
    with col3:
        find_clicked = st.button(
            "🔍 Find free PDFs",
            key="find_pdfs_button",
            use_container_width=True,
            type="primary",
            disabled=selected_count == 0,
        )

    if selected_count == 0:
        st.caption("Tick papers in the list above (up to 10), then find their free PDFs.")
        return

    # Step 1: check each selected paper for a legal open-access PDF.
    if find_clicked:
        rows = _resolve_selected_rows(df, selected_keys)
        attempt: list[dict] = []
        progress = st.progress(0.0, text="Searching open-access sources…")
        for index, paper in enumerate(rows, start=1):
            result = find_legal_pdf(
                pmid=str(paper.get("pmid", "")),
                doi=str(paper.get("doi", "")),
                email=email,
                s2_api_key=s2_api_key,
            )
            attempt.append(
                {
                    "title": str(paper.get("title", "(untitled)")),
                    "has_pdf": result.has_pdf,
                    "source": result.best_source.source
                    if (result.has_pdf and result.best_source)
                    else "",
                    "url": result.best_source.url
                    if (result.has_pdf and result.best_source)
                    else "",
                }
            )
            progress.progress(index / max(len(rows), 1), text=f"Checked {index} of {len(rows)}…")
        progress.empty()
        st.session_state["pdf_attempt"] = {
            "rows": rows,
            "attempt": attempt,
            "keys": list(selected_keys),
        }

    attempt_state = st.session_state.get("pdf_attempt")
    if not attempt_state:
        st.caption("Click **Find free PDFs** to check open-access availability for your selection.")
        return

    if attempt_state.get("keys") != list(selected_keys):
        st.caption("Selection changed since the last search — click **Find free PDFs** again.")

    # Step 2: show what the attempt found.
    attempt = attempt_state["attempt"]
    found = [item for item in attempt if item["has_pdf"]]
    st.markdown(f"**Found {len(found)} free PDF(s)** out of {len(attempt)} selected paper(s).")
    for item in attempt:
        title = item["title"][:90]
        if item["has_pdf"]:
            if item.get("url"):
                st.markdown(f"✅ [{title}]({item['url']}) — _{item['source']}_")
            else:
                st.markdown(f"✅ {title} — _{item['source']}_")
        else:
            st.markdown(f"⚪ {title} — no free open-access PDF")

    if not found:
        st.info(
            "None of the selected papers have a free open-access PDF — they are likely "
            "paywalled. Tip: add your email in the sidebar to enable Unpaywall, which "
            "can find more open-access copies."
        )
        return

    # Step 3: download only the PDFs that were actually retrieved.
    if st.button(f"📥 Download {len(found)} PDF(s) as ZIP", key="build_zip_button", type="primary"):
        with st.spinner("Fetching PDFs and building ZIP…"):
            try:
                zip_bytes, filename, packaged = generate_download_zip(
                    attempt_state["rows"], topic, email, s2_api_key=s2_api_key
                )
            except Exception as exc:
                st.error(f"Error building ZIP: {exc}")
                return

        if packaged > 0:
            st.success(f"✅ {packaged} PDF(s) packaged — {len(zip_bytes) / 1024 / 1024:.1f} MB")
            st.caption("Contains the PDFs + metadata.csv + metadata.json")
            st.download_button(
                label=f"⬇️ Save {filename}",
                data=zip_bytes,
                file_name=filename,
                mime="application/zip",
                key="download_zip_file",
            )
            if packaged < len(found):
                st.info(
                    f"{len(found) - packaged} found PDF(s) couldn't be downloaded "
                    "automatically (the host blocks server downloads). Open them "
                    "via the ✅ links above to grab them in your browser."
                )
        else:
            st.warning(
                "The sources listed open-access PDFs, but none could be downloaded "
                "automatically (some hosts block server downloads). Click a ✅ paper's "
                "title above to open its PDF directly."
            )


def _on_paper_checkbox_change(checkbox_key: str, selection_key: str) -> None:
    """Sync a checkbox toggle into selected_papers, enforcing MAX_SELECTIONS."""
    init_selection_state()
    checked = st.session_state.get(checkbox_key, False)

    if checked:
        if selection_key in st.session_state.selected_papers:
            return
        if len(st.session_state.selected_papers) >= MAX_SELECTIONS:
            # Cap reached: undo this tick and tell the user.
            st.session_state[checkbox_key] = False
            st.toast(
                f"You can select up to {MAX_SELECTIONS} papers at a time. "
                "Uncheck one before adding another.",
                icon="⚠️",
            )
            return
        st.session_state.selected_papers[selection_key] = True
    else:
        st.session_state.selected_papers.pop(selection_key, None)


def selection_key_for_row(pmid: str = "", doi: str = "", title: str = "") -> str:
    """Stable selection key for a paper: PMID, else DOI, else title, else 'unknown'."""
    if pmid and pmid.strip():
        return pmid.strip()
    if doi and doi.strip():
        return doi.strip()
    if title and title.strip():
        return title.strip()[:100]
    return "unknown"


def _checkbox_key_for(selection_key: str) -> str:
    return f"cb_{selection_key.replace(':', '_').replace('/', '_')[:40]}"


def select_papers(rows: list[dict], max_total: int = MAX_SELECTIONS) -> tuple[int, bool]:
    """Add the given paper rows to the selection, up to the global cap.

    Returns (added_count, cap_reached). Also sets each checkbox's widget state
    so the boxes reflect the new selection after a rerun.
    """
    init_selection_state()
    added = 0
    cap_reached = False
    for row in rows:
        if len(st.session_state.selected_papers) >= max_total:
            cap_reached = True
            break
        key = selection_key_for_row(
            str(row.get("pmid", "")), str(row.get("doi", "")), str(row.get("title", ""))
        )
        if key in st.session_state.selected_papers:
            continue
        st.session_state.selected_papers[key] = True
        st.session_state[_checkbox_key_for(key)] = True
        added += 1
    return added, cap_reached


def render_paper_checkbox(pmid: str, doi: str = "", title: str = "") -> bool:
    """Render checkbox for paper selection. Uses PMID/DOI as stable key.

    Selections are capped at MAX_SELECTIONS; an over-cap tick is reverted with a toast.
    """
    init_selection_state()

    selection_key = selection_key_for_row(pmid, doi, title)
    checkbox_key = _checkbox_key_for(selection_key)

    # Seed widget state from the source of truth before the widget is created,
    # so we never set a widget value and pass `value=` at the same time.
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = selection_key in st.session_state.selected_papers

    # Give each checkbox a meaningful accessible name (read by screen readers)
    # while keeping it visually hidden, instead of a generic repeated "Select".
    label = f"Select paper: {title.strip()[:80]}" if title and title.strip() else "Select paper"

    return st.checkbox(
        label,
        key=checkbox_key,
        label_visibility="collapsed",
        on_change=_on_paper_checkbox_change,
        args=(checkbox_key, selection_key),
    )


def render_paper_selection_list(df: pd.DataFrame, max_per_view: int = 10) -> None:
    """Render paginated checkbox list for all papers in a dataframe."""
    if df.empty:
        return

    init_selection_state()

    st.caption(f"📋 Select papers to download ({len(st.session_state.selected_papers)} selected)")

    # Pagination
    total_papers = len(df)
    total_pages = (total_papers + max_per_view - 1) // max_per_view

    if "paper_selection_page" not in st.session_state:
        st.session_state.paper_selection_page = 0

    page_col, page_info = st.columns([1, 2])
    with page_col:
        st.session_state.paper_selection_page = st.number_input(
            "Page",
            min_value=0,
            max_value=total_pages - 1,
            value=st.session_state.paper_selection_page,
            key="paper_page_input"
        )
    with page_info:
        st.caption(f"Page {st.session_state.paper_selection_page + 1} of {total_pages}")

    # Get papers for current page
    start_idx = st.session_state.paper_selection_page * max_per_view
    end_idx = min(start_idx + max_per_view, total_papers)
    page_df = df.iloc[start_idx:end_idx]

    # Render checkboxes for papers on this page
    for _, row in page_df.iterrows():
        cb_col, title_col, year_col, tier_col = st.columns([0.5, 3, 0.8, 1])

        with cb_col:
            render_paper_checkbox(
                str(row.get("pmid", "")),
                str(row.get("doi", "")),
                str(row.get("title", ""))
            )
        with title_col:
            title = str(row.get("title", "(untitled)"))[:80]
            st.caption(title)
        with year_col:
            year = str(row.get("year", ""))[:12]
            st.caption(year)
        with tier_col:
            st.caption(row.get("tier", "—"))


def show_pdf_settings() -> str:
    """Resolve the contact email for Unpaywall.

    Uses the email configured in Streamlit secrets; only renders a sidebar
    input when none is configured. No download-folder field (the ZIP is
    delivered through the browser, so a server-side path is meaningless).
    """
    return get_ncbi_email()
