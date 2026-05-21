# CorePapers: Open-Access PDF Finder Feature

## Overview
Added legal open-access PDF discovery, download, and storage to CorePapers. Respects copyright and paywalls by exclusively using legal sources.

## Implemented Phases

### Phase 1: PDF Finder Module (`pdf_finder.py`)
Searches legal open-access sources via APIs:

**Sources (in preference order):**
1. **Unpaywall** - Best for finding published PDFs
   - Requires: User's email in Streamlit secrets (`ncbi_email`)
   - Returns: DOI-based OA metadata and direct PDF links
   
2. **PubMed Central OA Subset** - Safest for biomedical papers
   - No auth required
   - Returns: Direct PMC PDF URLs for OA articles
   
3. **Europe PMC** - European biomedical repository
   - No auth required
   - Returns: OA articles and full-text links

4. **OpenAlex** - Comprehensive OA status tracking
   - No auth required
   - Returns: OA URLs and status indicators

**API Safety:**
- All calls respect robots.txt and rate limits
- No Sci-Hub or paywall bypass
- All sources are legal and authorized
- Proper User-Agent headers

**Usage:**
```python
from pdf_finder import find_legal_pdf, get_pdf_status_label

result = find_legal_pdf(
    pmid="12345678",
    doi="10.1234/example",
    email="user@example.com"  # optional, required for Unpaywall
)

if result.has_pdf:
    print(f"Source: {result.best_source.source}")
    print(f"URL: {result.best_source.url}")
    print(f"License: {result.best_source.license}")
```

### Phase 2: Storage Module (`pdf_storage.py`)
Handles PDF download and metadata management:

**Folder Structure:**
```
CorePaper_Downloads/
  Topic_Name/
    2026-05-21/
      PMID_12345678_paper_title_2024.pdf
      PMID_12345678_paper_title_2024_metadata.json
      metadata.csv
```

**Metadata Stored (JSON + CSV):**
- Title, authors, journal, year
- DOI, PMID, PMCID
- PDF source and license
- Download timestamp
- Search query used
- Relevance score (from CorePapers)

**Features:**
- Automatic folder creation by topic and date
- Safe filename generation (sanitizes special characters)
- PDF URL validation before download
- Metadata JSON for machine readability
- CSV summary for spreadsheet analysis
- Duplicate detection (skips re-download)
- Partial download cleanup on failure

**Usage:**
```python
from pdf_storage import PDFStorage, PDFMetadata

storage = PDFStorage("./CorePaper_Downloads")

metadata = PDFMetadata(
    title="Study Title",
    authors="Smith J, Jones M",
    journal="Nature",
    year="2024",
    doi="10.1038/nature.2024.001",
    pmid="12345678",
    pmcid="PMC9876543",
    source_of_pdf="PubMed Central OA",
    license="CC-BY-4.0",
    downloaded_at="2026-05-21T10:30:00",
    search_query="sepsis mortality",
    relevance_score=0.95
)

success, msg, path = storage.save_pdf_with_metadata(
    pdf_source,
    metadata,
    topic="Sepsis"
)
```

### Phase 3: Streamlit UI Integration (`pdf_ui.py` + `app.py`)

**New Tab: "PDF Download"**
- Appears in results alongside Papers, Evidence Review, Exports, etc.
- Shows PDF discovery status for all papers
- Bulk download button for entire result set
- Download summary with success/failure counts

**Sidebar Settings:**
- PDF download folder selection (defaults to ~/Documents/CorePaper_Downloads)
- NCBI email input for Unpaywall API access
- Shows number of PDFs already stored

**Features:**
- Find PDFs by individual paper
- Bulk download all legal PDFs from a search result
- Progress bar during bulk download
- Download summary report
- Safe handling of failed downloads

**Safety:**
- User must provide real email for Unpaywall (API requirement)
- All sources are verified legal
- No automatic paywalls bypass
- Metadata tracks source and license

## Configuration

### Streamlit Secrets (Optional but recommended)
Add to `.streamlit/secrets.toml`:
```toml
ncbi_email = "your-real-email@example.com"
```

### Environment Variable (Alternative)
```bash
export NCBI_EMAIL="your-real-email@example.com"
```

## Important Legal Notes

1. **"Free to read" ≠ "Free to distribute"**
   - PubMed Central notes that not all PMC articles are available for text mining/reuse
   - The OA subset is the safer downloadable subset
   - License information is always tracked

2. **Personal use**
   - Downloaded PDFs are for personal research
   - Always check license before sharing or publishing derived work
   - Metadata includes license status for verification

3. **Rate limiting**
   - APIs are queried with appropriate User-Agent headers
   - Requests respect rate limits and robots.txt
   - Caching prevents redundant queries

## Testing

All modules have been tested with:
- Real API responses from Unpaywall, Europe PMC, OpenAlex
- PDF URL validation
- Metadata JSON/CSV generation
- Safe filename handling
- Storage and retrieval workflows

## Future Enhancements (Phase 4)

Planned safety controls:
- Download logging with audit trail
- License compliance checking before sharing
- Batch processing with cancellation support
- Integration with Zotero for library sync
- PDF metadata extraction and indexing

## Files Added/Modified

**New Files:**
- `pdf_finder.py` (285 lines) - Open-access source discovery
- `pdf_storage.py` (225 lines) - PDF and metadata management
- `pdf_ui.py` (180 lines) - Streamlit UI components
- `PDF_FEATURE.md` (this file)

**Modified Files:**
- `app.py` - Added PDF Download tab, PDF settings sidebar, render_pdf_downloads() function
