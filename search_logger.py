"""Append-only search feedback logging to a private Google Sheet.

Streamlit Cloud has an ephemeral filesystem, so search logs must live in
external persistent storage to be analysable days/weeks later. This module
appends one row per search (input + result summary) to a ``searches`` tab and
all admitted papers (tagged by ``search_id``) to a ``papers`` tab.

It is intentionally fail-soft: any configuration or network problem returns a
status string and never raises, so logging can never break the search UX.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections import Counter
from typing import Any

import pandas as pd

SEARCHES_TAB = "searches"
PAPERS_TAB = "papers"

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SEARCH_HEADERS = [
    "search_id",
    "timestamp_utc",
    "topic",
    "search_purpose",
    "question_type",
    "population",
    "intervention",
    "comparator",
    "outcome",
    "primer_status",
    "candidates_retrieved",
    "papers_admitted",
    "tier_breakdown",
    "recall_only_count",
    "errors_count",
    "error_sample",
]

# Columns of the per-paper log (intersected with whatever the result provides).
PAPER_LOG_COLUMNS = [
    "pmid",
    "doi",
    "title",
    "journal",
    "year",
    "study_design",
    "evidence_group",
    "tier",
    "reading_section",
    "total_score",
    "relevance_score",
    "citation_count",
    "quartile",
    "topic_match_gate",
    "relation_type",
    "intent_fit",
    "intent_reranked",
    "search_layers",
    "url",
]

# Cache one gspread client per service-account identity for the process life.
_CLIENTS: dict[str, Any] = {}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _client(service_account_info: dict[str, Any]):
    """Return a cached, authorised gspread client. Imports lazily so the app
    does not hard-depend on gspread when logging is unconfigured."""
    import gspread  # noqa: PLC0415 - lazy import keeps failures contained
    from google.oauth2.service_account import Credentials  # noqa: PLC0415

    info = dict(service_account_info)
    key = str(info.get("client_email", "")) or "default"
    client = _CLIENTS.get(key)
    if client is None:
        creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
        client = gspread.authorize(creds)
        _CLIENTS[key] = client
    return client


def _open_spreadsheet(client, sheet_id: str, sheet_name: str):
    if sheet_id:
        return client.open_by_key(sheet_id)
    return client.open(sheet_name)


def _worksheet(spreadsheet, name: str, headers: list[str]):
    """Get a worksheet, creating it with a header row if missing."""
    import gspread  # noqa: PLC0415

    try:
        return spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=name, rows=2000, cols=max(len(headers), 10))
        worksheet.append_row(headers, value_input_option="RAW")
        return worksheet


def _cell(value: Any) -> str:
    """Coerce any value to a sheet-safe string (cap length to keep cells sane)."""
    if value is None:
        return ""
    text = str(value)
    return text[:4000]


def _tier_breakdown(papers: list[dict[str, Any]]) -> str:
    counts = Counter(str(p.get("tier", "") or "unknown") for p in papers)
    return "; ".join(f"{tier}={count}" for tier, count in sorted(counts.items()))


def log_search(
    result: dict[str, Any],
    context_meta: dict[str, Any],
    service_account_info: dict[str, Any] | None,
    sheet_id: str = "",
    sheet_name: str = "",
) -> str:
    """Append a search's input + papers to the Google Sheet.

    Returns a short status string: ``"logged"``, ``"not_configured"``, or
    ``"error: ..."``. Never raises.
    """
    if not service_account_info:
        return "not_configured: no service-account credentials found in secrets"
    if not (sheet_id or sheet_name):
        return "not_configured: feedback_sheet_id is missing from secrets"

    try:
        papers = list(result.get("papers", []) or [])
        search_id = f"{_dt.date.today().isoformat()}-{uuid.uuid4().hex[:8]}"
        timestamp = _utc_now()
        errors = list(result.get("errors", []) or [])
        recall_only = sum(1 for p in papers if p.get("expansion_recall_only"))

        search_row = [
            search_id,
            timestamp,
            _cell(context_meta.get("topic")),
            _cell(context_meta.get("search_purpose")),
            _cell(context_meta.get("question_type")),
            _cell(context_meta.get("population")),
            _cell(context_meta.get("intervention")),
            _cell(context_meta.get("comparator")),
            _cell(context_meta.get("outcome")),
            _cell(result.get("topic_primer_status")),
            _cell(result.get("retrieved_count")),
            _cell(len(papers)),
            _cell(_tier_breakdown(papers)),
            _cell(recall_only),
            _cell(len(errors)),
            _cell(errors[0] if errors else ""),
        ]

        client = _client(service_account_info)
        spreadsheet = _open_spreadsheet(client, sheet_id, sheet_name)

        searches_ws = _worksheet(spreadsheet, SEARCHES_TAB, SEARCH_HEADERS)
        searches_ws.append_row(search_row, value_input_option="RAW")

        if papers:
            frame = pd.DataFrame(papers)
            available = [c for c in PAPER_LOG_COLUMNS if c in frame.columns]
            paper_headers = ["search_id", "timestamp_utc", *available]
            papers_ws = _worksheet(spreadsheet, PAPERS_TAB, paper_headers)
            rows = [
                [search_id, timestamp, *[_cell(row.get(col)) for col in available]]
                for row in frame[available].to_dict("records")
            ]
            # Batch append in chunks to stay within request size limits.
            for start in range(0, len(rows), 500):
                papers_ws.append_rows(rows[start : start + 500], value_input_option="RAW")

        return "logged"
    except Exception as exc:  # fail-soft: logging must never break search
        return f"error: {str(exc)[:200]}"
