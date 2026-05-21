from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from pdf_download import generate_download_zip
from pdf_finder import find_legal_pdf, get_pdf_status_label, PDFSearchResult
from pdf_storage import PDFMetadata, PDFStorage


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
        st.metric("Papers selected", selected_count, delta=f"out of {len(df)}")

    with col2:
        if st.button("Clear selection", use_container_width=True):
            st.session_state.selected_papers = {}
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
                        matching_rows = None
                        if key.startswith("pmid:"):
                            pmid = key.replace("pmid:", "")
                            matching_rows = df[df["pmid"].astype(str) == pmid]
                        elif key.startswith("doi:"):
                            doi = key.replace("doi:", "")
                            matching_rows = df[df["doi"].astype(str) == doi]
                        elif key.startswith("title:"):
                            title = key.replace("title:", "")
                            matching_rows = df[df["title"].astype(str) == title]

                        if matching_rows is not None and not matching_rows.empty:
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


def render_paper_checkbox(pmid: str, doi: str = "", title: str = "", idx: int | None = None, section_key: str = "") -> bool:
    """Render checkbox for paper selection. Uses PMID/DOI as stable key."""
    init_selection_state()

    # Use PMID if available, fallback to DOI, fallback to title
    if pmid and pmid.strip():
        selection_key = f"pmid:{pmid.strip()}"
    elif doi and doi.strip():
        selection_key = f"doi:{doi.strip()}"
    elif title and title.strip():
        selection_key = f"title:{title.strip()[:100]}"
    else:
        selection_key = f"idx:{idx}" if idx is not None else "unknown"

    is_selected = selection_key in st.session_state.selected_papers

    # Make checkbox key unique per section to avoid duplicate key errors
    checkbox_key = f"paper_select_{section_key}_{selection_key.replace(':', '_')}" if section_key else f"paper_select_{selection_key.replace(':', '_')}"
    checked = st.checkbox(
        "Download",
        value=is_selected,
        key=checkbox_key,
        label_visibility="collapsed",
    )

    if checked and selection_key not in st.session_state.selected_papers:
        st.session_state.selected_papers[selection_key] = True
    elif not checked and selection_key in st.session_state.selected_papers:
        del st.session_state.selected_papers[selection_key]

    return checked


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
