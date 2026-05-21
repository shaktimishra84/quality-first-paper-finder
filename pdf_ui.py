from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

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


def render_bulk_download(
    df: pd.DataFrame,
    topic: str,
    download_folder: Path,
    email: str,
) -> None:
    """Render bulk download UI for all papers in results."""
    if df.empty:
        st.warning("No papers to download")
        return

    if st.button(
        "📥 Download All Legal PDFs",
        help="Download all papers with confirmed open-access PDFs",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        storage = PDFStorage(download_folder)
        results_list = []
        successful = 0

        for idx, (_, row) in enumerate(df.iterrows()):
            status_text.text(f"Processing {idx + 1}/{len(df)}...")

            success, message = download_pdf_for_paper(
                row,
                topic,
                download_folder,
                email,
            )

            results_list.append(
                {
                    "pmid": row.get("pmid", ""),
                    "title": row.get("title", ""),
                    "success": success,
                    "message": message,
                }
            )

            if success:
                successful += 1

            progress_bar.progress((idx + 1) / len(df))

        progress_bar.empty()
        status_text.empty()

        st.success(f"Downloaded {successful}/{len(df)} papers successfully")

        with st.expander("Download Summary"):
            for result in results_list:
                if result["success"]:
                    st.success(f"✓ {result['pmid']}: {result['message']}")
                else:
                    st.warning(f"✗ {result['pmid']}: {result['message']}")

        st.info(f"Saved to: {download_folder}")


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
