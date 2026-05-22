from __future__ import annotations

import functools
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests


UNPAYWALL_API_URL = "https://api.unpaywall.org/v2"
EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMC_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX_API_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper"

REQUEST_TIMEOUT = (5, 12)
DEFAULT_HEADERS = {
    "User-Agent": "CorePapers/1.0; legal-open-access-pdf-finder"
}

OA_STATUS_NOT_FOUND = "not_found"
OA_STATUS_CLOSED = "closed"
OA_STATUS_GREEN = "green"
OA_STATUS_GOLD = "gold"
OA_STATUS_HYBRID = "hybrid"
OA_STATUS_BRONZE = "bronze"


@dataclass
class PDFSource:
    url: str
    source: str
    license: str
    is_best_oa: bool = False

    def __repr__(self) -> str:
        return f"PDFSource(source={self.source}, license={self.license})"


@dataclass
class PDFSearchResult:
    has_pdf: bool
    sources: list[PDFSource] = None
    best_source: Optional[PDFSource] = None
    message: str = ""
    oa_status: str = OA_STATUS_NOT_FOUND

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.sources and not self.best_source:
            self.best_source = self.sources[0]


def _normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.lower()


def _is_valid_pdf_url(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    if url_lower.endswith(".pdf"):
        return True
    if "pdf" in url_lower:
        return True
    return False


def _check_unpaywall_doi(doi: str, email: str = "") -> PDFSearchResult:
    if not doi:
        return PDFSearchResult(has_pdf=False, message="No DOI provided")

    if not email:
        return PDFSearchResult(
            has_pdf=False,
            message="Unpaywall API requires email. Provide ncbi_email in Streamlit secrets.",
        )

    doi = _normalize_doi(doi)
    try:
        url = f"{UNPAYWALL_API_URL}/{doi}"
        params = {"email": email.strip()}
        response = requests.get(
            url,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            return PDFSearchResult(has_pdf=False, message="Invalid Unpaywall response")

        oa_status = data.get("is_oa", False)
        best_oa_location = data.get("best_oa_location")

        if best_oa_location and isinstance(best_oa_location, dict):
            pdf_url = best_oa_location.get("url_for_pdf") or best_oa_location.get("url")
            if pdf_url and _is_valid_pdf_url(pdf_url):
                license_info = best_oa_location.get("license", "unspecified")
                host_type = best_oa_location.get("host_type", "unknown")
                source_name = f"Unpaywall ({host_type})"

                source = PDFSource(
                    url=pdf_url,
                    source=source_name,
                    license=license_info,
                    is_best_oa=True,
                )

                return PDFSearchResult(
                    has_pdf=True,
                    sources=[source],
                    best_source=source,
                    oa_status="gold" if oa_status else "closed",
                    message=f"OA PDF available via {host_type}",
                )

        if oa_status:
            return PDFSearchResult(
                has_pdf=False,
                oa_status="green",
                message="OA status indicated but PDF URL not available",
            )

        return PDFSearchResult(
            has_pdf=False,
            oa_status=OA_STATUS_CLOSED,
            message="No open-access version found",
        )

    except requests.exceptions.RequestException as e:
        return PDFSearchResult(has_pdf=False, message=f"Unpaywall API error: {str(e)}")
    except Exception as e:
        return PDFSearchResult(has_pdf=False, message=f"Unexpected error checking Unpaywall: {str(e)}")


@functools.lru_cache(maxsize=256)
def _check_pmc_oa_pmid(pmid: str) -> PDFSearchResult:
    if not pmid:
        return PDFSearchResult(has_pdf=False, message="No PMID provided")

    try:
        params = {
            "db": "pmc",
            "id": pmid,
            "rettype": "json",
        }
        response = requests.get(
            f"{PMC_API_URL}/esummary.fcgi",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        result = data.get("result", {})
        if pmid in result:
            record = result[pmid]
            if isinstance(record, dict):
                article_ids = record.get("articleids", [])
                for aid in article_ids:
                    if aid.get("idtype") == "pmcid":
                        pmcid = aid.get("value", "")
                        if pmcid:
                            # The id value may already include the "PMC" prefix;
                            # strip it so we don't build a "PMCPMC..." URL.
                            pmcid_num = pmcid.upper().replace("PMC", "")
                            # Europe PMC's article render endpoint serves the
                            # PDF bytes (redirecting to api/getPdf). NCBI's
                            # /pdf/ page only returns an HTML interstitial to
                            # server clients, so it is not used.
                            source = PDFSource(
                                url=f"https://europepmc.org/articles/PMC{pmcid_num}?pdf=render",
                                source="Europe PMC (OA render)",
                                license="Open access (PMC)",
                                is_best_oa=True,
                            )
                            return PDFSearchResult(
                                has_pdf=True,
                                sources=[source],
                                best_source=source,
                                oa_status="gold",
                                message="PMC OA PDF available",
                            )

        return PDFSearchResult(
            has_pdf=False,
            message="PMID not in PMC OA subset",
        )

    except requests.exceptions.RequestException as e:
        return PDFSearchResult(has_pdf=False, message=f"PMC API error: {str(e)}")
    except Exception as e:
        return PDFSearchResult(has_pdf=False, message=f"Unexpected error checking PMC OA: {str(e)}")


@functools.lru_cache(maxsize=256)
def _check_europe_pmc_pmid(pmid: str) -> PDFSearchResult:
    if not pmid:
        return PDFSearchResult(has_pdf=False, message="No PMID provided")

    try:
        params = {
            "query": f"PMID:{pmid}",
            "format": "json",
            "pageSize": "1",
        }
        response = requests.get(
            f"{EUROPE_PMC_API_URL}/search",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("resultList", {}).get("result", [])
        if results and isinstance(results[0], dict):
            result = results[0]
            is_open_access = result.get("isOpenAccess") == "Y"

            if is_open_access:
                source_name = result.get("source", "Europe PMC")
                license_info = result.get("license", "unspecified")

                pdf_urls = []
                # Prefer the OA render endpoint built from the PMCID; Europe PMC
                # often reports OA without a usable fullTextLink in search.
                pmcid = result.get("pmcid", "")
                if pmcid:
                    pmcid_num = str(pmcid).upper().replace("PMC", "")
                    pdf_urls.append(
                        f"https://europepmc.org/articles/PMC{pmcid_num}?pdf=render"
                    )
                if result.get("fullTextLink"):
                    pdf_urls.append(result["fullTextLink"])
                if result.get("fullTextLinks"):
                    for link in result.get("fullTextLinks", []):
                        if isinstance(link, dict) and _is_valid_pdf_url(link.get("url", "")):
                            pdf_urls.append(link["url"])

                if pdf_urls:
                    source = PDFSource(
                        url=pdf_urls[0],
                        source=f"Europe PMC ({source_name})",
                        license=license_info,
                    )
                    return PDFSearchResult(
                        has_pdf=True,
                        sources=[PDFSource(url=u, source=f"Europe PMC ({source_name})", license=license_info) for u in pdf_urls],
                        best_source=source,
                        oa_status="green",
                        message="OA article found on Europe PMC",
                    )

                return PDFSearchResult(
                    has_pdf=False,
                    oa_status="green",
                    message="OA article on Europe PMC but PDF URL not available",
                )

        return PDFSearchResult(has_pdf=False, message="Not found on Europe PMC")

    except requests.exceptions.RequestException as e:
        return PDFSearchResult(has_pdf=False, message=f"Europe PMC API error: {str(e)}")
    except Exception as e:
        return PDFSearchResult(has_pdf=False, message=f"Unexpected error checking Europe PMC: {str(e)}")


@functools.lru_cache(maxsize=256)
def _check_openalex_doi(doi: str) -> PDFSearchResult:
    if not doi:
        return PDFSearchResult(has_pdf=False, message="No DOI provided")

    doi = _normalize_doi(doi)
    try:
        params = {
            "filter": f"doi:{doi}",
        }
        response = requests.get(
            OPENALEX_API_URL,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if results and isinstance(results[0], dict):
            work = results[0]
            open_access = work.get("open_access", {})

            if isinstance(open_access, dict):
                oa_url = open_access.get("oa_url")
                oa_status = open_access.get("oa_status")

                if oa_url and _is_valid_pdf_url(oa_url):
                    source = PDFSource(
                        url=oa_url,
                        source=f"OpenAlex ({oa_status or 'unknown'})",
                        license="unspecified",
                    )
                    return PDFSearchResult(
                        has_pdf=True,
                        sources=[source],
                        best_source=source,
                        oa_status=oa_status or "unknown",
                        message=f"OA URL available ({oa_status})",
                    )

        return PDFSearchResult(has_pdf=False, message="Not found on OpenAlex or no OA URL")

    except requests.exceptions.RequestException as e:
        return PDFSearchResult(has_pdf=False, message=f"OpenAlex API error: {str(e)}")
    except Exception as e:
        return PDFSearchResult(has_pdf=False, message=f"Unexpected error checking OpenAlex: {str(e)}")


@functools.lru_cache(maxsize=256)
def _check_semantic_scholar(pmid: str = "", doi: str = "", api_key: str = "") -> PDFSearchResult:
    if doi:
        paper_id = f"DOI:{_normalize_doi(doi)}"
    elif pmid:
        paper_id = f"PMID:{pmid.strip()}"
    else:
        return PDFSearchResult(has_pdf=False, message="No DOI or PMID provided")

    headers = dict(DEFAULT_HEADERS)
    if api_key:
        headers["x-api-key"] = api_key.strip()

    try:
        response = requests.get(
            f"{SEMANTIC_SCHOLAR_API_URL}/{paper_id}",
            params={"fields": "openAccessPdf,isOpenAccess"},
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            oa_pdf = data.get("openAccessPdf")
            if isinstance(oa_pdf, dict):
                pdf_url = oa_pdf.get("url", "")
                if pdf_url:
                    status = str(oa_pdf.get("status") or "green").lower()
                    source = PDFSource(
                        url=pdf_url,
                        source="Semantic Scholar",
                        license=str(oa_pdf.get("license") or "unspecified"),
                        is_best_oa=True,
                    )
                    return PDFSearchResult(
                        has_pdf=True,
                        sources=[source],
                        best_source=source,
                        oa_status=status,
                        message=f"OA PDF via Semantic Scholar ({status})",
                    )

        return PDFSearchResult(has_pdf=False, message="No OA PDF on Semantic Scholar")

    except requests.exceptions.RequestException as e:
        return PDFSearchResult(has_pdf=False, message=f"Semantic Scholar API error: {str(e)}")
    except Exception as e:
        return PDFSearchResult(has_pdf=False, message=f"Unexpected error checking Semantic Scholar: {str(e)}")


def find_legal_pdf(
    pmid: str = "", doi: str = "", email: str = "", s2_api_key: str = ""
) -> PDFSearchResult:
    """
    Search for legal open-access PDFs.
    Tries sources in order of preference and returns best match.

    Args:
        pmid: PubMed ID
        doi: Digital Object Identifier
        email: Email for Unpaywall API (required for best results)

    Returns:
        PDFSearchResult with has_pdf, sources, best_source, and oa_status.
    """
    if not pmid and not doi:
        return PDFSearchResult(has_pdf=False, message="No PMID or DOI provided")

    sources_found: list[PDFSource] = []
    best_result: Optional[PDFSearchResult] = None
    oa_status = OA_STATUS_NOT_FOUND

    if doi:
        doi = _normalize_doi(doi)
        unpaywall_result = _check_unpaywall_doi(doi, email)
        if unpaywall_result.has_pdf:
            best_result = unpaywall_result
            sources_found.extend(unpaywall_result.sources or [])

    if pmid and not best_result:
        pmc_result = _check_pmc_oa_pmid(pmid)
        if pmc_result.has_pdf:
            best_result = pmc_result
            sources_found.extend(pmc_result.sources or [])

    if pmid and not best_result:
        europe_result = _check_europe_pmc_pmid(pmid)
        if europe_result.has_pdf:
            best_result = europe_result
            sources_found.extend(europe_result.sources or [])

    if (doi or pmid) and not best_result:
        s2_result = _check_semantic_scholar(pmid=pmid, doi=doi, api_key=s2_api_key)
        if s2_result.has_pdf:
            best_result = s2_result
            sources_found.extend(s2_result.sources or [])

    if doi and not best_result:
        openalex_result = _check_openalex_doi(doi)
        if openalex_result.has_pdf:
            best_result = openalex_result
            sources_found.extend(openalex_result.sources or [])

    if best_result:
        return PDFSearchResult(
            has_pdf=True,
            sources=sources_found or best_result.sources,
            best_source=best_result.best_source,
            oa_status=best_result.oa_status,
            message=best_result.message,
        )

    return PDFSearchResult(
        has_pdf=False,
        sources=[],
        oa_status=OA_STATUS_CLOSED,
        message="No legal open-access PDF found in Unpaywall, PMC OA, Europe PMC, Semantic Scholar, or OpenAlex",
    )


def _source_fetch_rank(source: PDFSource) -> int:
    """Order sources by how reliably they serve a PDF to a server fetch."""
    name = (source.source or "").lower()
    url = (source.url or "").lower()
    if "pdf=render" in url or "oa render" in name:
        return 0
    if "pubmed central" in name:
        return 1
    if "europe pmc" in name:
        return 2
    if "semantic scholar" in name:
        return 3
    if "openalex" in name:
        return 4
    if "unpaywall" in name:
        return 5
    return 6


def find_all_pdf_sources(
    pmid: str = "", doi: str = "", email: str = "", s2_api_key: str = ""
) -> list[PDFSource]:
    """Collect candidate OA PDF sources from every resolver (not just the first).

    Returns a de-duplicated list ordered by fetch reliability, so the download
    step can try each URL until one returns a genuine PDF.
    """
    collected: list[PDFSource] = []

    if doi:
        result = _check_unpaywall_doi(doi, email)
        if result.has_pdf:
            collected.extend(result.sources or [])
    if pmid:
        for result in (_check_pmc_oa_pmid(pmid), _check_europe_pmc_pmid(pmid)):
            if result.has_pdf:
                collected.extend(result.sources or [])
    if doi or pmid:
        result = _check_semantic_scholar(pmid=pmid, doi=doi, api_key=s2_api_key)
        if result.has_pdf:
            collected.extend(result.sources or [])
    if doi:
        result = _check_openalex_doi(doi)
        if result.has_pdf:
            collected.extend(result.sources or [])

    # Any PMC id surfaced by any resolver (e.g. an ncbi.nlm.nih.gov article URL
    # that blocks bots) can be fetched via Europe PMC's render endpoint, which
    # serves the PDF directly. Inject those as high-priority sources.
    pmc_render: list[PDFSource] = []
    for source in collected:
        match = re.search(r"PMC(\d+)", source.url or "", re.IGNORECASE)
        if match:
            pmc_render.append(
                PDFSource(
                    url=f"https://europepmc.org/articles/PMC{match.group(1)}?pdf=render",
                    source="Europe PMC (OA render)",
                    license="Open access (PMC)",
                    is_best_oa=True,
                )
            )
    collected = pmc_render + collected

    seen: set[str] = set()
    unique: list[PDFSource] = []
    for source in collected:
        if source.url and source.url not in seen:
            seen.add(source.url)
            unique.append(source)

    unique.sort(key=_source_fetch_rank)
    return unique


def get_pdf_status_label(result: PDFSearchResult) -> str:
    """Return a user-friendly status label for UI display."""
    if result.has_pdf:
        if result.best_source:
            return f"📄 {result.best_source.source}"
        return "📄 PDF available"
    elif result.oa_status == OA_STATUS_GREEN:
        return "🔗 OA landing page only"
    else:
        return "❌ No legal PDF found"
