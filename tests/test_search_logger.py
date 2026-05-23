"""Tests for the Google Sheets search-feedback logger. gspread is mocked out so
these run without the dependency or any network."""

from __future__ import annotations

import search_logger


class _FakeWS:
    def __init__(self) -> None:
        self.rows: list[list] = []

    def append_row(self, row, value_input_option="RAW") -> None:
        self.rows.append(row)

    def append_rows(self, rows, value_input_option="RAW") -> None:
        self.rows.extend(rows)


def _patch_sheets(monkeypatch) -> dict[str, _FakeWS]:
    sheets: dict[str, _FakeWS] = {}

    def fake_ws(spreadsheet, name, headers):
        ws = sheets.get(name)
        if ws is None:
            ws = _FakeWS()
            sheets[name] = ws
        return ws

    monkeypatch.setattr(search_logger, "_client", lambda info: object())
    monkeypatch.setattr(search_logger, "_open_spreadsheet", lambda client, sid, sname: object())
    monkeypatch.setattr(search_logger, "_worksheet", fake_ws)
    return sheets


def test_not_configured_without_credentials_or_sheet() -> None:
    assert search_logger.log_search({}, {}, None, sheet_id="X") == "not_configured"
    assert search_logger.log_search({}, {}, {"client_email": "a"}, sheet_id="", sheet_name="") == "not_configured"


def test_logs_search_row_and_paper_rows(monkeypatch) -> None:
    sheets = _patch_sheets(monkeypatch)
    result = {
        "papers": [
            {"pmid": "1", "title": "A", "tier": "Tier 1: Must-read", "expansion_recall_only": False},
            {"pmid": "2", "title": "B", "tier": "Tier 3: Background", "expansion_recall_only": True},
        ],
        "retrieved_count": 50,
        "topic_primer_status": "generated",
        "errors": [],
    }
    meta = {
        "topic": "ards", "search_purpose": "Deep Search", "question_type": "General evidence map",
        "population": "", "intervention": "", "comparator": "", "outcome": "",
    }

    status = search_logger.log_search(result, meta, {"client_email": "x@y.iam"}, sheet_id="SID")

    assert status == "logged"
    assert set(sheets) == {"searches", "papers"}
    assert len(sheets["searches"].rows) == 1
    search_row = sheets["searches"].rows[0]
    assert search_row[2] == "ards"                 # topic column
    assert "Tier 1: Must-read=1" in search_row[12]  # tier_breakdown
    assert search_row[13] == "1"                   # recall_only_count

    paper_rows = sheets["papers"].rows
    assert len(paper_rows) == 2
    assert paper_rows[0][0] == search_row[0]        # paper rows tagged with the search_id
    assert paper_rows[0][0] == paper_rows[1][0]


def test_logging_is_fail_soft(monkeypatch) -> None:
    def boom(info):
        raise RuntimeError("auth failed")

    monkeypatch.setattr(search_logger, "_client", boom)
    status = search_logger.log_search(
        {"papers": []}, {"topic": "t"}, {"client_email": "x"}, sheet_id="SID"
    )
    assert status.startswith("error:")


def test_tier_breakdown_counts() -> None:
    out = search_logger._tier_breakdown(
        [{"tier": "Tier 1"}, {"tier": "Tier 1"}, {"tier": "Tier 2"}]
    )
    assert "Tier 1=2" in out
    assert "Tier 2=1" in out
