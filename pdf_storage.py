from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from pdf_finder import PDFSource, PDFSearchResult


@dataclass
class PDFMetadata:
    title: str
    authors: str
    journal: str
    year: str
    doi: str
    pmid: str
    pmcid: str
    source_of_pdf: str
    license: str
    downloaded_at: str
    search_query: str
    relevance_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_csv_row(self) -> list:
        return [
            self.title,
            self.authors,
            self.journal,
            self.year,
            self.doi,
            self.pmid,
            self.pmcid,
            self.source_of_pdf,
            self.license,
            self.downloaded_at,
            self.search_query,
            str(self.relevance_score) if self.relevance_score else "",
        ]

    @staticmethod
    def csv_headers() -> list[str]:
        return [
            "title",
            "authors",
            "journal",
            "year",
            "doi",
            "pmid",
            "pmcid",
            "source_of_pdf",
            "license",
            "downloaded_at",
            "search_query",
            "relevance_score",
        ]


class PDFStorage:
    def __init__(self, base_download_path: Path | str = "./CorePaper_Downloads"):
        self.base_path = Path(base_download_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_paper_folder(
        self,
        topic: str,
        date: Optional[str] = None,
    ) -> Path:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        folder = self.base_path / topic / date
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def safe_filename(self, text: str, max_length: int = 60) -> str:
        text = re.sub(r"[^\w\s\-()]", "", text)
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text[:max_length].strip("_")

    def get_pdf_filename(
        self,
        pmid: str = "",
        doi: str = "",
        title: str = "",
        year: str = "",
    ) -> str:
        parts = []

        if pmid:
            parts.append(f"PMID_{pmid}")
        elif doi:
            doi_clean = doi.replace("/", "_").replace(".", "_")
            parts.append(f"DOI_{doi_clean[:20]}")

        if title:
            parts.append(self.safe_filename(title, 40))

        if year:
            parts.append(year)

        filename = "_".join(filter(None, parts)) or "paper"
        return f"{filename}.pdf"

    def download_pdf(
        self,
        pdf_url: str,
        output_path: Path,
        timeout: tuple[int, int] = (5, 30),
    ) -> bool:
        try:
            headers = {
                "User-Agent": "CorePapers/1.0; pdf-download"
            }
            response = requests.get(
                pdf_url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()

            if response.headers.get("content-type", "").lower() not in [
                "application/pdf",
                "application/octet-stream",
            ]:
                return False

            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path.exists() and output_path.stat().st_size > 0

        except Exception:
            if output_path.exists():
                output_path.unlink()
            return False

    def save_pdf_with_metadata(
        self,
        pdf_source: PDFSource,
        metadata: PDFMetadata,
        topic: str,
        output_folder: Optional[Path] = None,
    ) -> tuple[bool, str, Path]:
        try:
            if not output_folder:
                output_folder = self.get_paper_folder(topic)

            pdf_filename = self.get_pdf_filename(
                pmid=metadata.pmid,
                doi=metadata.doi,
                title=metadata.title,
                year=metadata.year,
            )

            pdf_path = output_folder / pdf_filename
            metadata_path = output_folder / f"{pdf_path.stem}_metadata.json"

            if pdf_path.exists():
                return True, f"Already downloaded: {pdf_filename}", pdf_path

            success = self.download_pdf(pdf_source.url, pdf_path)

            if not success:
                return False, f"Failed to download PDF: {pdf_source.url}", pdf_path

            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        **metadata.to_dict(),
                        "pdf_filename": pdf_filename,
                        "pdf_url": pdf_source.url,
                        "pdf_source": pdf_source.source,
                    },
                    f,
                    indent=2,
                )

            return True, f"Downloaded: {pdf_filename}", pdf_path

        except Exception as e:
            return False, f"Error saving PDF: {str(e)}", Path()

    def write_metadata_csv(
        self,
        topic: str,
        metadata_list: list[PDFMetadata],
        date: Optional[str] = None,
    ) -> Path:
        folder = self.get_paper_folder(topic, date)
        csv_path = folder / "metadata.csv"

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(PDFMetadata.csv_headers())
                for metadata in metadata_list:
                    writer.writerow(metadata.to_csv_row())
            return csv_path

        except Exception as e:
            raise IOError(f"Failed to write metadata CSV: {str(e)}")

    def list_downloaded_pdfs(
        self,
        topic: str,
        date: Optional[str] = None,
    ) -> list[dict]:
        folder = self.get_paper_folder(topic, date)

        pdfs = []
        for json_file in folder.glob("*_metadata.json"):
            try:
                with open(json_file, "r") as f:
                    metadata = json.load(f)
                    pdfs.append(metadata)
            except Exception:
                continue

        return pdfs

    def cleanup_partial_download(self, pdf_path: Path) -> None:
        if pdf_path.exists():
            pdf_path.unlink()
        metadata_path = pdf_path.parent / f"{pdf_path.stem}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
