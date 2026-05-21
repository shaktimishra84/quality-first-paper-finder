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
                            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_num}/pdf/"
                            source = PDFSource(
                                url=pdf_url,
                                source="PubMed Central OA",
                                license="Public domain (PMC)",
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


def find_legal_pdf(pmid: str = "", doi: str = "", email: str = "") -> PDFSearchResult:
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
        message="No legal open-access PDF found in Unpaywall, PMC OA, Europe PMC, or OpenAlex",
    )


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
