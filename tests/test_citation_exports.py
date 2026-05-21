from __future__ import annotations

from paper_finder import papers_to_bibtex, papers_to_ris


def sample_papers() -> list[dict[str, object]]:
    return [
        {
            "title": "Implementation of CAM-ICU compliance in critical care",
            "authors": "Smith J; Rao P",
            "journal": "Critical Care",
            "year": 2024,
            "doi": "https://doi.org/10.1000/test_doi",
            "pmid": "12345678",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "abstract": "A delirium screening implementation study.",
        }
    ]


def test_papers_to_bibtex_exports_verified_metadata() -> None:
    bibtex = papers_to_bibtex(sample_papers())

    assert bibtex.startswith("@article{Smith2024implementation")
    assert "author = {Smith J and Rao P}" in bibtex
    assert "journal = {Critical Care}" in bibtex
    assert "year = {2024}" in bibtex
    assert "doi = {10.1000/test\\_doi}" in bibtex
    assert "pmid = {12345678}" in bibtex


def test_papers_to_ris_exports_verified_metadata() -> None:
    ris = papers_to_ris(sample_papers())

    assert "TY  - JOUR" in ris
    assert "TI  - Implementation of CAM-ICU compliance in critical care" in ris
    assert "AU  - Smith J" in ris
    assert "AU  - Rao P" in ris
    assert "DO  - 10.1000/test_doi" in ris
    assert "AN  - PMID:12345678" in ris
    assert ris.rstrip().endswith("ER  -")
