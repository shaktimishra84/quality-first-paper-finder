from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd


def clean_doi(doi: str) -> str:
    """Clean DOI to standard format."""
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi


def normalize_pmid(pmid: str) -> str:
    """Normalize PMID."""
    pmid = str(pmid).strip()
    return pmid if pmid and pmid != "nan" else ""


def format_authors_string(authors_str: str) -> list[str]:
    """Parse author string into list."""
    if not authors_str or pd.isna(authors_str):
        return []
    authors = [a.strip() for a in str(authors_str).split(";")]
    return [a for a in authors if a]


def escape_bibtex(text: str) -> str:
    """Escape special characters for BibTeX."""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    # Escape special characters
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    # Note: { } and \ need careful handling, usually left as-is in titles
    return text


def format_ris_authors(authors: list[str]) -> list[str]:
    """Format authors for RIS format."""
    ris_authors = []
    for author in authors:
        # Parse "First Last" or "Last, First"
        author = author.strip()
        if not author:
            continue
        # Try to swap to "Last, First" format if it's "First Last"
        if "," not in author and len(author.split()) >= 2:
            parts = author.split()
            author = f"{parts[-1]}, {' '.join(parts[:-1])}"
        ris_authors.append(author)
    return ris_authors


class BibTexFormatter:
    """Generate professional BibTeX exports."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.used_keys = {}

    def _citation_key(self, row: pd.Series, index: int) -> str:
        """Generate citation key from author and year."""
        author = str(row.get("authors", "")).split(";")[0].strip()
        if "," in author:
            # "Family, Given" -> family name
            author_part = author.split(",")[0].strip().split()[-1].lower()
        elif author:
            # "Given Family" -> last token
            author_part = author.split()[-1].lower()
        else:
            author_part = f"paper{index}"
        year = str(row.get("year", "")).strip()[-4:]  # Last 4 chars
        base_key = f"{author_part}{year}"

        # Handle duplicates
        count = self.used_keys.get(base_key, 0)
        self.used_keys[base_key] = count + 1
        return f"{base_key}" if count == 0 else f"{base_key}{chr(97 + count)}"

    def format_all(self) -> str:
        """Generate BibTeX for all papers."""
        entries = []

        for idx, (_, row) in enumerate(self.df.iterrows(), start=1):
            entry = self._format_entry(row, idx)
            if entry:
                entries.append(entry)

        return "\n\n".join(entries) + ("\n" if entries else "")

    def _format_entry(self, row: pd.Series, index: int) -> str:
        """Format single paper as BibTeX entry."""
        pmid = normalize_pmid(row.get("pmid", ""))
        doi = clean_doi(str(row.get("doi", "") or ""))
        title = str(row.get("title", "")).strip()
        authors = format_authors_string(row.get("authors", ""))
        journal = str(row.get("journal", "")).strip()
        year = str(row.get("year", "")).strip()
        volume = str(row.get("volume", "")).strip()
        issue = str(row.get("issue", "")).strip()
        pages = str(row.get("pages", "")).strip()
        url = str(row.get("url", "")).strip()
        abstract = str(row.get("abstract", "")).strip()

        if not title:
            return None

        citation_key = self._citation_key(row, index)

        fields = []
        fields.append(("title", escape_bibtex(title)))

        if authors:
            author_str = " and ".join(escape_bibtex(a) for a in authors)
            fields.append(("author", author_str))

        if journal:
            fields.append(("journal", escape_bibtex(journal)))

        if year:
            fields.append(("year", year))

        if volume:
            fields.append(("volume", volume))

        if issue:
            fields.append(("number", issue))

        if pages:
            fields.append(("pages", pages))

        if doi:
            fields.append(("doi", doi))

        if pmid:
            fields.append(("note", f"PMID: {pmid}"))

        if url:
            fields.append(("url", url))

        if abstract:
            fields.append(("abstract", escape_bibtex(abstract)))

        # CorePapers specific
        tier = str(row.get("tier", "")).strip()
        if tier:
            fields.append(("keywords", f"CorePapers: {tier}"))

        field_lines = [
            f"  {name} = {{{value}}}" for name, value in fields if value
        ]

        return "@article{" + citation_key + ",\n" + ",\n".join(field_lines) + "\n}"


class RISFormatter:
    """Generate professional RIS exports."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def format_all(self) -> str:
        """Generate RIS for all papers."""
        entries = []

        for _, row in self.df.iterrows():
            entry = self._format_entry(row)
            if entry:
                entries.append(entry)

        return "\n\n".join(entries) + ("\n" if entries else "")

    def _format_entry(self, row: pd.Series) -> str:
        """Format single paper as RIS entry."""
        title = str(row.get("title", "")).strip()
        if not title:
            return None

        lines = ["TY  - JOUR"]  # Journal article

        # Title
        lines.append(f"TI  - {title}")

        # Authors
        authors = format_authors_string(row.get("authors", ""))
        ris_authors = format_ris_authors(authors)
        for author in ris_authors:
            lines.append(f"AU  - {author}")

        # Journal
        journal = str(row.get("journal", "")).strip()
        if journal:
            lines.append(f"T2  - {journal}")
            lines.append(f"JO  - {journal}")

        # Year
        year = str(row.get("year", "")).strip()
        if year:
            lines.append(f"PY  - {year}")

        # Volume
        volume = str(row.get("volume", "")).strip()
        if volume:
            lines.append(f"VL  - {volume}")

        # Issue
        issue = str(row.get("issue", "")).strip()
        if issue:
            lines.append(f"IS  - {issue}")

        # Pages
        pages = str(row.get("pages", "")).strip()
        if pages:
            lines.append(f"SP  - {pages.split('-')[0] if '-' in pages else pages}")
            if "-" in pages:
                lines.append(f"EP  - {pages.split('-')[1]}")

        # DOI
        doi = clean_doi(str(row.get("doi", "") or ""))
        if doi:
            lines.append(f"DO  - {doi}")

        # PMID
        pmid = normalize_pmid(row.get("pmid", ""))
        if pmid:
            lines.append(f"AN  - PMID:{pmid}")

        # PMCID
        pmcid = str(row.get("pmcid", "")).strip()
        if pmcid:
            lines.append(f"AN  - PMCID:{pmcid}")

        # URL
        url = str(row.get("url", "")).strip()
        if url:
            lines.append(f"UR  - {url}")

        # Abstract
        abstract = str(row.get("abstract", "")).strip()
        if abstract:
            lines.append(f"AB  - {abstract}")

        # Keywords / Tags
        keywords = []

        # Add tier as keyword
        tier = str(row.get("tier", "")).strip()
        if tier:
            keywords.append(f"Tier: {tier}")

        # Add study design
        design = str(row.get("study_design", "")).strip()
        if design:
            keywords.append(f"Design: {design}")

        # Add section
        section = str(row.get("reading_section", "")).strip()
        if section:
            keywords.append(f"Section: {section}")

        if keywords:
            lines.append(f"KW  - {'; '.join(keywords)}")

        # End of record
        lines.append("ER  -")

        return "\n".join(lines)


class JSONFormatter:
    """Generate JSON export (rich metadata, Zotero-compatible)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def format_all(self) -> str:
        """Generate JSON for all papers."""
        items = []

        for _, row in self.df.iterrows():
            item = self._format_entry(row)
            if item:
                items.append(item)

        return json.dumps({"papers": items}, indent=2)

    def _format_entry(self, row: pd.Series) -> dict[str, Any]:
        """Format single paper as JSON."""
        title = str(row.get("title", "")).strip()
        if not title:
            return None

        authors = format_authors_string(row.get("authors", ""))

        entry = {
            "title": title,
            "authors": authors,
            "journal": str(row.get("journal", "")).strip(),
            "year": str(row.get("year", "")).strip(),
            "volume": str(row.get("volume", "")).strip() or None,
            "issue": str(row.get("issue", "")).strip() or None,
            "pages": str(row.get("pages", "")).strip() or None,
            "doi": clean_doi(str(row.get("doi", "") or "")),
            "pmid": normalize_pmid(row.get("pmid", "")),
            "pmcid": str(row.get("pmcid", "")).strip() or None,
            "url": str(row.get("url", "")).strip() or None,
            "abstract": str(row.get("abstract", "")).strip() or None,
        }

        # CorePapers metadata
        metadata = {}

        tier = str(row.get("tier", "")).strip()
        if tier:
            metadata["tier"] = tier

        design = str(row.get("study_design", "")).strip()
        if design:
            metadata["study_design"] = design

        section = str(row.get("reading_section", "")).strip()
        if section:
            metadata["reading_section"] = section

        score = row.get("composite_score", row.get("score"))
        if score:
            metadata["relevance_score"] = float(score)

        if metadata:
            entry["corepapers_metadata"] = metadata

        return entry


def export_to_bibtex(df: pd.DataFrame) -> str:
    """Export dataframe to BibTeX format."""
    formatter = BibTexFormatter(df)
    return formatter.format_all()


def export_to_ris(df: pd.DataFrame) -> str:
    """Export dataframe to RIS format."""
    formatter = RISFormatter(df)
    return formatter.format_all()


def export_to_json(df: pd.DataFrame) -> str:
    """Export dataframe to JSON format."""
    formatter = JSONFormatter(df)
    return formatter.format_all()
