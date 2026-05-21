from __future__ import annotations

from paper_finder import (
    SearchContext,
    api_supervisor_clinicaltrials_queries,
    api_supervisor_crossref_queries,
    api_supervisor_preprint_queries,
    api_supervisor_semantic_scholar_queries,
    api_supervisor_unpaywall_queries,
    resolve_dois_to_pubmed_pmids,
    search_semantic_scholar_identifiers,
    semantic_scholar_headers,
)


def test_new_api_supervisor_query_builders_include_requested_sources() -> None:
    context = SearchContext(topic="cerebral venous thrombosis")

    layer_names = [
        *(layer for layer, *_ in api_supervisor_crossref_queries(context, 10)),
        *(layer for layer, *_ in api_supervisor_semantic_scholar_queries(context, 10)),
        *(layer for layer, *_ in api_supervisor_unpaywall_queries(context, 10)),
        *(layer for layer, *_ in api_supervisor_clinicaltrials_queries(context, 10)),
        *(layer for layer, *_ in api_supervisor_preprint_queries(context)),
    ]

    assert "API supervisor - Crossref exact" in layer_names
    assert "API supervisor - Semantic Scholar exact" in layer_names
    assert "API supervisor - Unpaywall exact" in layer_names
    assert "API supervisor - ClinicalTrials.gov" in layer_names
    assert "API supervisor - medRxiv recent preprints" in layer_names
    assert "API supervisor - bioRxiv recent preprints" in layer_names


def test_doi_resolution_uses_pubmed_article_identifier_search(monkeypatch) -> None:
    captured_queries: list[str] = []

    def fake_search_pubmed(query: str, retmax: int, email: str = "", api_key: str = "") -> list[str]:
        captured_queries.append(query)
        return ["12345678", "not-a-pmid", "12345678"]

    monkeypatch.setattr("paper_finder.search_pubmed", fake_search_pubmed)

    pmids = resolve_dois_to_pubmed_pmids(["https://doi.org/10.1000/test"], email="user@example.com")

    assert pmids == ["12345678"]
    assert captured_queries == ['"10.1000/test"[AID]']


def test_semantic_scholar_api_key_is_sent_as_header(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {
                        "externalIds": {
                            "PubMed": "12345678",
                            "DOI": "10.1000/test",
                        }
                    }
                ]
            }

    def fake_get(url: str, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return FakeResponse()

    monkeypatch.setattr("paper_finder.requests.get", fake_get)
    monkeypatch.setattr("paper_finder.throttle_semantic_scholar", lambda api_key="": None)

    search_semantic_scholar_identifiers("unique header test", limit=1, api_key="s2k-test")

    assert semantic_scholar_headers("s2k-test")["x-api-key"] == "s2k-test"
    assert captured["headers"]["x-api-key"] == "s2k-test"
