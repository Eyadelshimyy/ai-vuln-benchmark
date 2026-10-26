"""Tests for response models."""

from __future__ import annotations

from app.models.search_result import (
    AnalyticsData,
    SearchHit,
    SearchResponse,
    SuggestResponse,
)


class TestSearchHit:
    """Tests for SearchHit model."""

    def test_to_dict_basic(self):
        hit = SearchHit(
            id="doc-1",
            title="Test",
            content_snippet="snippet",
            type="document",
            owner_id="user-1",
        )
        d = hit.to_dict()
        assert d["id"] == "doc-1"
        assert d["title"] == "Test"
        assert d["type"] == "document"
        assert "created_at" not in d

    def test_to_dict_with_optional_fields(self):
        hit = SearchHit(
            id="file-1",
            title="report.pdf",
            content_snippet="",
            type="file",
            owner_id="user-1",
            mime_type="application/pdf",
            size=1024,
            folder_id="folder-1",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-06-01T00:00:00Z",
        )
        d = hit.to_dict()
        assert d["mime_type"] == "application/pdf"
        assert d["size"] == 1024
        assert d["folder_id"] == "folder-1"
        assert d["created_at"] == "2024-01-01T00:00:00Z"


class TestSearchResponse:
    """Tests for SearchResponse model."""

    def test_to_dict(self):
        resp = SearchResponse(
            results=[
                SearchHit(
                    id="doc-1",
                    title="Test",
                    content_snippet="",
                    type="document",
                    owner_id="user-1",
                )
            ],
            total=1,
            page=1,
            page_size=20,
            query="test",
        )
        d = resp.to_dict()
        assert d["total"] == 1
        assert d["page"] == 1
        assert d["page_size"] == 20
        assert d["query"] == "test"
        assert len(d["results"]) == 1


class TestSuggestResponse:
    """Tests for SuggestResponse model."""

    def test_to_dict(self):
        resp = SuggestResponse(suggestions=["hello", "help"], query="hel")
        d = resp.to_dict()
        assert d["suggestions"] == ["hello", "help"]
        assert d["query"] == "hel"


class TestAnalyticsData:
    """Tests for AnalyticsData model."""

    def test_to_dict(self):
        data = AnalyticsData(
            popular_queries=[{"query": "test", "count": 5}],
            zero_result_queries=[{"query": "xyz", "count": 2}],
            total_searches=100,
            avg_results_per_query=3.5,
        )
        d = data.to_dict()
        assert d["total_searches"] == 100
        assert d["avg_results_per_query"] == 3.5
        assert len(d["popular_queries"]) == 1
        assert len(d["zero_result_queries"]) == 1
