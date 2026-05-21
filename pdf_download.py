from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from pdf_finder import find_legal_pdf
from pdf_storage import PDFMetadata, PDFStorage


# A realistic browser User-Agent; many OA publisher hosts reject bot-style
# agents. We are only fetching open-access PDFs the user is entitled to.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
}


def _fetch_pdf_bytes(url: str) -> Optional[bytes]:
    """Fetch a URL and return its bytes only if it is a genuine PDF."""
    import requests

    try:
        response = requests.get(
            url,
            headers=BROWSER_HEADERS,
            timeout=(5, 30),
            allow_redirects=True,
        )
        response.raise_for_status()
    except Exception:
        return None

    content = response.content
    content_type = response.headers.get("Content-Type", "").lower()
    if content[:5] == b"%PDF-" or "application/pdf" in content_type:
        return content
    return None


def generate_download_zip(
    selected_papers: list[dict],
    topic: str,
    email: str = "",
    s2_api_key: str = "",
) -> tuple[bytes, str, int]:
    """
    Generate ZIP file with selected papers and metadata.
    Returns (zip_bytes, filename, pdfs_packaged) where pdfs_packaged is the
    number of genuine PDFs actually written into the archive.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        metadata_list = []
        successful_downloads = 0

        for paper in selected_papers:
            pmid = str(paper.get("pmid", ""))
            doi = str(paper.get("doi", ""))

            # Find PDF
            result = find_legal_pdf(pmid=pmid, doi=doi, email=email, s2_api_key=s2_api_key)

            if not result.has_pdf or not result.best_source:
                continue

            # Try each discovered source URL until one yields a genuine PDF.
            candidate_sources = result.sources or [result.best_source]
            content = None
            used_source = result.best_source
            for source in candidate_sources:
                fetched = _fetch_pdf_bytes(source.url)
                if fetched is not None:
                    content = fetched
                    used_source = source
                    break

            if content is None:
                continue

            pdf_filename = _safe_filename(
                pmid=pmid,
                doi=doi,
                title=paper.get("title", ""),
                year=paper.get("year", ""),
            )

            zip_file.writestr(pdf_filename, content)
            successful_downloads += 1

            metadata = PDFMetadata(
                title=str(paper.get("title", "")),
                authors=str(paper.get("authors", "")),
                journal=str(paper.get("journal", "")),
                year=str(paper.get("year", "")),
                doi=doi,
                pmid=pmid,
                pmcid=str(paper.get("pmcid", "")),
                source_of_pdf=used_source.source,
                license=used_source.license,
                downloaded_at=pd.Timestamp.now().isoformat(),
                search_query=str(paper.get("reading_section", "")),
                relevance_score=float(paper.get("composite_score", 0))
                if "composite_score" in paper
                else None,
            )
            metadata_list.append(metadata)

        # Add metadata CSV to ZIP
        if metadata_list:
            csv_buffer = io.StringIO()
            import csv

            writer = csv.writer(csv_buffer)
            writer.writerow(PDFMetadata.csv_headers())
            for metadata in metadata_list:
                writer.writerow(metadata.to_csv_row())

            zip_file.writestr("metadata.csv", csv_buffer.getvalue())

            # Add metadata JSON
            import json

            metadata_json = json.dumps(
                [m.to_dict() for m in metadata_list], indent=2
            )
            zip_file.writestr("metadata.json", metadata_json)

    zip_buffer.seek(0)
    filename = f"corepapers_{topic.replace(' ', '_')}.zip"
    return zip_buffer.getvalue(), filename, successful_downloads


def _safe_filename(
    pmid: str = "",
    doi: str = "",
    title: str = "",
    year: str = "",
) -> str:
    """Generate safe filename for PDF."""
    import re

    parts = []

    if pmid:
        parts.append(f"PMID_{pmid}")
    elif doi:
        doi_clean = doi.replace("/", "_").replace(".", "_")
        parts.append(f"DOI_{doi_clean[:20]}")

    if title:
        title_clean = re.sub(r"[^\w\s\-()]", "", title)
        title_clean = re.sub(r"\s+", "_", title_clean)
        title_clean = re.sub(r"_+", "_", title_clean)
        parts.append(title_clean[:40].strip("_"))

    if year:
        parts.append(year)

    filename = "_".join(filter(None, parts)) or "paper"
    return f"{filename}.pdf"
