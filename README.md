# CorePapers

**Verified Medical Literature Search**

Purpose-aware PubMed evidence discovery for clinicians and researchers. Find landmark reviews, trials, guidelines, research gaps, and rare case literature with transparent evidence tiers.

![CorePapers WhatsApp preview](static/corepapers-whatsapp-preview.png)

Use `app/static/corepapers-share.html` as the bot-readable share URL when the deployed app shell does not expose Open Graph tags to WhatsApp crawlers.

## What it does

- Generates broad, focused, review/guideline, landmark/classic, recent-update, and gap PubMed searches from a topic or PICO question.
- Uses a mandatory expected-paper sanity layer for topics with known sentinel papers, starting with cerebral venous thrombosis.
- Runs an API discovery supervisor before scoring: PubMed, Europe PMC, Crossref, OpenAlex, Semantic Scholar, Unpaywall, ClinicalTrials.gov, medRxiv/bioRxiv, and optional PubMed related-article expansion.
- Builds a medical evidence review artifact with source IDs, evidence hierarchy, source comparison, citation verification caveats, gaps, and limitations.
- Uses researcher-facing search modes instead of technical depth knobs: Learning mode, Research mode, Deep search mode, and Rare / case mode. Deep search and Rare / case modes keep all relevant publication types visible, including editorials, correspondence, letters, comments, case reports, and case series.
- Admits only papers with a verifiable PMID, DOI, PubMed link, OpenAlex record, or Semantic Scholar record.
- Enriches accepted records with OpenAlex citation counts and optional Semantic Scholar cross-checks.
- Lets citation enrichment scale up to the retrieved candidate set while keeping it optional.
- Uses a compact main-page search flow with optional PICO details collapsed by default.
- Scores each paper using the rule weights plus a search-mode fit adjustment:
  - Relevance: 40
  - Study design: 20
  - Journal quality: 20
  - Citation strength: 10
  - Recency: 10
- Splits output into search-mode-specific sections such as best reviews, original research papers, exhaustive evidence buckets, editorials/correspondence, or closest matching case reports.
- Protects major reviews and landmark candidates from disappearing when citation counts or journal-quality metadata is temporarily unavailable.
- Marks missing data explicitly instead of guessing.
- Exports the full CSV database, core reading pack CSV, and PMID list.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py --server.headless true --server.port 8512
```

Then open:

```text
http://localhost:8512
```

The app needs outbound network access to PubMed and the discovery/enrichment APIs. If it is run inside a network-blocked sandbox, it will show source errors and will not admit papers. Unpaywall title search uses an email from app secrets when one is configured.

## Safety rules implemented

- No AI-memory references.
- No guessed PMIDs, DOIs, citation counts, journal quartiles, or conclusions.
- Manual landmark hints are used only for cross-checking; they do not create accepted papers.
- For supported topics, expected landmark papers are fetched by PMID and reported if still missing.
- API-discovered papers are admitted only after PubMed PMID verification/fetch, or after DOI-based records are resolved back through PubMed where possible.
- The evidence review workflow is adapted from reusable ideas in Feynman under its MIT license notice, without bundling the Feynman CLI/runtime.
- Search goals automatically tune retrieval depth, ranking emphasis, citation enrichment, and whether source-grounded AI gap hypotheses are attempted.
- Recent high-quality papers are protected from being unfairly penalized for immature citation counts.
- Related papers are tagged into evidence families so duplicate trial/database outputs do not silently dominate.

## Version 1 limits

This app intentionally does not summarize PDFs, extract full text, sync with Zotero, write manuscripts, or run a RAG chatbot. It maps verified metadata, evidence quality, and research gaps so the final clinical interpretation can be done from the source papers.
