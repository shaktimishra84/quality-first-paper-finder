from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from pdf_download import generate_download_zip
from pdf_finder import find_legal_pdf, get_pdf_status_label, PDFSearchResult
from pdf_storage import PDFMetadata, PDFStorage

# Maximum number of papers a user may select for download at once.
MAX_SELECTIONS = 10


def get_download_folder() -> Path:
    """Get download folder from sidebar, create if needed."""
    default_path = str(Path.home() / "Documents" / "CorePaper_Downloads")
    path_str = st.sidebar.text_input(
        "PDF Download Folder",
        value=st.session_state.get("download_folder", default_path),
        help="Where to save downloaded PDFs and metadata",
    )
    st.session_state["download_folder"] = path_str
    return Path(path_str)


def get_ncbi_email() -> str:
    """Get NCBI email from sidebar or Streamlit secrets."""
    if "secrets" in dir(st):
        try:
            email = st.secrets.get("ncbi_email", "")
            if email:
                return email
        except Exception:
            pass

    email = st.sidebar.text_input(
        "NCBI Email (for Unpaywall)",
        value=st.session_state.get("ncbi_email", ""),
        type="password",
        help="Required for Unpaywall API access. Your actual email address.",
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


def render_download_button(df: pd.DataFrame, topic: str, email: str) -> None:
    """Render download selected papers as ZIP button."""
    if df.empty:
        return

    init_selection_state()

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_count = len(st.session_state.selected_papers)
        st.metric(
            f"Papers selected (max {MAX_SELECTIONS})",
            selected_count,
            delta=f"out of {len(df)}",
        )

    with col2:
        if st.button("Clear selection", use_container_width=True):
            st.session_state.selected_papers = {}
            # Also reset the individual checkbox widget states.
            for widget_key in [k for k in st.session_state if k.startswith("cb_")]:
                del st.session_state[widget_key]
            st.rerun()

    with col3:
        if selected_count > 0:
            if st.button(
                "📥 Download as ZIP",
                key="download_zip_button",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner(f"Preparing {selected_count} paper(s)..."):
                    selected_papers = []
                    for key in st.session_state.selected_papers.keys():
                        # Try matching by PMID first (most common)
                        matching_rows = df[df["pmid"].astype(str) == key]

                        # If no match by PMID, try DOI
                        if matching_rows.empty:
                            matching_rows = df[df["doi"].astype(str) == key]

                        # If no match by DOI, try title
                        if matching_rows.empty:
                            matching_rows = df[df["title"].astype(str) == key]

                        if not matching_rows.empty:
                            selected_papers.append(matching_rows.iloc[0].to_dict())

                    try:
                        zip_bytes, filename = generate_download_zip(
                            selected_papers, topic, email
                        )

                        st.download_button(
                            label="⬇️ Click to download ZIP",
                            data=zip_bytes,
                            file_name=filename,
                            mime="application/zip",
                            key="download_zip_file",
                        )

                        st.success(
                            f"✅ ZIP ready: {filename} ({len(zip_bytes) / 1024 / 1024:.1f} MB)"
                        )
                        st.caption(
                            "Contains PDFs + metadata.csv + metadata.json"
                        )

                    except Exception as e:
                        st.error(f"Error generating ZIP: {str(e)}")
        else:
            st.caption("Select papers to download")


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


def render_paper_checkbox(pmid: str, doi: str = "", title: str = "") -> bool:
    """Render checkbox for paper selection. Uses PMID/DOI as stable key.

    Selections are capped at MAX_SELECTIONS; the 11th tick is reverted with a toast.
    """
    init_selection_state()

    # Use PMID if available, fallback to DOI, fallback to title
    if pmid and pmid.strip():
        selection_key = pmid.strip()
    elif doi and doi.strip():
        selection_key = doi.strip()
    elif title and title.strip():
        selection_key = title.strip()[:100]
    else:
        selection_key = "unknown"

    checkbox_key = f"cb_{selection_key.replace(':', '_').replace('/', '_')[:40]}"

    # Seed widget state from the source of truth before the widget is created,
    # so we never set a widget value and pass `value=` at the same time.
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = selection_key in st.session_state.selected_papers

    return st.checkbox(
        "Select",
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


def render_bulk_download(
    df: pd.DataFrame,
    topic: str,
    download_folder: Path,
    email: str,
) -> None:
    """Legacy function - now handled in main results."""
    pass


def show_pdf_settings() -> tuple[Path, str]:
    """Show PDF settings in sidebar, return folder and email."""
    with st.sidebar:
        st.divider()
        st.subheader("📥 PDF Downloads")

        download_folder = get_download_folder()
        email = get_ncbi_email()

        if download_folder.exists():
            file_count = len(list(download_folder.glob("**/*.pdf")))
            st.caption(f"📁 {file_count} PDFs stored")

    return download_folder, email
