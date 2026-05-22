from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd

from export_formatter import export_to_bibtex, export_to_ris
from pdf_finder import find_all_pdf_sources
from pdf_storage import PDFMetadata, PDFStorage


def _enrich_from_crossref(paper: dict) -> dict:
    """Best-effort fill of volume/issue/pages/authors/journal/year via Crossref.

    Reference managers want full bibliographic detail; search records often lack
    volume/issue/pages. Looked up by DOI; returns the paper unchanged on any error.
    """
    import requests

    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(paper.get("doi", "")).strip(), flags=re.IGNORECASE)
    if not doi:
        return paper
    if all(str(paper.get(field, "")).strip() for field in ("volume", "issue", "pages")):
        return paper

    try:
        response = requests.get(
            f"https://api.crossref.org/works/{doi}",
            params={"mailto": "corepapers@streamlit.app"},
            headers={"User-Agent": "CorePapers/1.0 (mailto:corepapers@streamlit.app)"},
            timeout=(5, 15),
        )
        response.raise_for_status()
        message = response.json().get("message", {})
    except Exception:
        return paper

    enriched = dict(paper)

    def _fill(field: str, value) -> None:
        if value and not str(enriched.get(field, "")).strip():
            enriched[field] = str(value)

    _fill("volume", message.get("volume"))
    _fill("issue", message.get("issue"))
    _fill("pages", message.get("page"))
    container = message.get("container-title") or []
    if container:
        _fill("journal", container[0])
    issued = (message.get("issued") or {}).get("date-parts") or [[]]
    if issued and issued[0]:
        _fill("year", issued[0][0])
    if not str(enriched.get("authors", "")).strip():
        names = [
            f"{a.get('family', '')}, {a.get('given', '')}".strip().strip(",")
            for a in (message.get("author") or [])
            if a.get("family")
        ]
        if names:
            enriched["authors"] = "; ".join(names)
    return enriched


# A realistic browser User-Agent; many OA publisher hosts reject bot-style
# agents. We are only fetching open-access PDFs the user is entitled to.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,text/html,*/*",
}

# OA landing pages declare their real PDF via this meta tag (used by Google
# Scholar). The two patterns cover either attribute order.
_CITATION_PDF_PATTERNS = (
    re.compile(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
        re.IGNORECASE,
    ),
)


def _looks_like_html(content: bytes, content_type: str) -> bool:
    if "html" in content_type:
        return True
    head = content[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or b"<html" in head


def _http_get(url: str, headers: dict):
    """GET with retry/backoff on transient failures (throttling, 5xx).

    OA hosts (especially Europe PMC) drop connections under rapid bulk
    fetching; a couple of retries recovers most of those.
    """
    import time

    import requests

    for attempt in range(3):
        try:
            response = requests.get(
                url, headers=headers, timeout=(5, 30), allow_redirects=True
            )
        except Exception:
            time.sleep(1.0 * (attempt + 1))
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(1.0 * (attempt + 1))
            continue
        return response
    return None


def _fetch_pdf_bytes(url: str, referer: Optional[str] = None, _depth: int = 0) -> Optional[bytes]:
    """Fetch a URL and return its bytes only if it is a genuine PDF.

    If the URL returns an OA landing page, follow its declared
    citation_pdf_url once to reach the actual PDF.
    """
    headers = dict(BROWSER_HEADERS)
    if referer:
        headers["Referer"] = referer

    response = _http_get(url, headers)
    if response is None:
        return None
    try:
        response.raise_for_status()
    except Exception:
        return None

    content = response.content
    content_type = response.headers.get("Content-Type", "").lower()
    if content[:5] == b"%PDF-" or "application/pdf" in content_type:
        return content

    # Landing page: follow the publisher-declared PDF link one level deep.
    if _depth == 0 and _looks_like_html(content, content_type):
        try:
            html = response.text
        except Exception:
            return None
        for pattern in _CITATION_PDF_PATTERNS:
            match = pattern.search(html)
            if match:
                pdf_url = urljoin(response.url, match.group(1).strip())
                if pdf_url and pdf_url != url:
                    return _fetch_pdf_bytes(pdf_url, referer=response.url, _depth=1)
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
        packaged_papers: list[dict] = []
        successful_downloads = 0

        for paper in selected_papers:
            pmid = str(paper.get("pmid", ""))
            doi = str(paper.get("doi", ""))

            # Collect every OA source for this paper, then try each URL until
            # one yields a genuine PDF (publisher links often block bots while
            # a PMC/Europe PMC copy of the same paper downloads fine).
            candidate_sources = find_all_pdf_sources(
                pmid=pmid, doi=doi, email=email, s2_api_key=s2_api_key
            )
            if not candidate_sources:
                continue

            content = None
            used_source = candidate_sources[0]
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
            packaged_papers.append(paper)

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

            # Reference-manager exports (Zotero / EndNote / Mendeley). Enrich
            # each packaged paper with full bibliographic detail via Crossref.
            enriched = [_enrich_from_crossref(paper) for paper in packaged_papers]
            references_df = pd.DataFrame(enriched)
            try:
                zip_file.writestr("references.bib", export_to_bibtex(references_df))
                zip_file.writestr("references.ris", export_to_ris(references_df))
            except Exception:
                pass

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

    # Inputs may arrive as ints/floats/NaN from the dataframe; coerce to str.
    def _clean(value: object) -> str:
        text = "" if value is None else str(value).strip()
        return "" if text.lower() in ("nan", "none") else text

    pmid = _clean(pmid)
    doi = _clean(doi)
    title = _clean(title)
    year = _clean(year)

    parts: list[str] = []

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

    filename = "_".join(part for part in parts if part) or "paper"
    return f"{filename}.pdf"
