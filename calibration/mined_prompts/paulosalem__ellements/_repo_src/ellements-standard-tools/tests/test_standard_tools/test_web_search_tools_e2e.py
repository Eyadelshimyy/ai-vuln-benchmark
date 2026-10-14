"""End-to-end tests for web search tools.

These tests make real API calls to DuckDuckGo to verify that the underlying
web search functionality returns expected, non-empty results.
"""

import os

import pytest
from ellements.standard_tools.web.search import SearchResults, WebSearcher

pytestmark = [
    pytest.mark.requires_network,
    pytest.mark.skipif(
        os.getenv("ELLEMENTS_RUN_NETWORK_TESTS") != "1",
        reason="set ELLEMENTS_RUN_NETWORK_TESTS=1 to run live web-search e2e tests",
    ),
]


class TestWebSearcherE2E:
    """End-to-end tests for web searcher with real API calls."""

    @pytest.fixture
    def searcher(self):
        """Get web searcher instance."""
        return WebSearcher()

    def test_search_web_returns_results(self, searcher):
        """Test that search_web returns non-empty search results."""
        result = searcher.search(query="Python programming", max_results=5, time_range="y")

        # Should return SearchResults object
        assert result is not None
        assert isinstance(result, SearchResults)
        assert len(result.results) > 0
        assert result.query == "Python programming"

    def test_search_with_different_query(self, searcher):
        """Test search with a different query."""
        result = searcher.search(query="machine learning algorithms", max_results=5, time_range="y")

        # Should return non-empty results
        assert result is not None
        assert isinstance(result, SearchResults)
        assert len(result.results) > 0

    def test_search_respects_max_results(self, searcher):
        """Test that max_results parameter affects result quantity."""
        # Search with small max_results
        result_small = searcher.search(query="JavaScript frameworks", max_results=3, time_range="y")

        # Search with larger max_results
        result_large = searcher.search(query="JavaScript frameworks", max_results=10, time_range="y")

        # Both should be non-empty
        assert len(result_small.results) > 0
        assert len(result_large.results) > 0

        # Larger result should have more or equal results
        assert len(result_large.results) >= len(result_small.results)

    def test_search_news_returns_results(self, searcher):
        """Test that search_news returns news-specific results."""
        result = searcher.search_news(query="artificial intelligence", max_results=5, time_range="w")

        # Should return SearchResults object
        assert result is not None
        assert isinstance(result, SearchResults)
        assert len(result.results) > 0

    def test_empty_query_handles_gracefully(self, searcher):
        """Test that empty queries are handled gracefully."""
        # The new ddgs library requires non-empty queries
        try:
            result = searcher.search(query="", max_results=5, time_range="y")
            # If it succeeds (shouldn't with ddgs), verify it returns SearchResults
            assert result is not None
            assert isinstance(result, SearchResults)
        except Exception as e:
            # Should get a clear error message about query being mandatory
            assert isinstance(e, Exception)
            assert "query" in str(e).lower() or "mandatory" in str(e).lower()

    @pytest.mark.parametrize("query,time_range", [
        ("Python", "d"),
        ("JavaScript", "w"),
        ("machine learning", "m"),
        ("data science", "y"),
        ("web development", "w"),
    ])
    def test_multiple_queries_and_time_ranges(self, searcher, query, time_range):
        """Test various query and time range combinations."""
        result = searcher.search(query=query, max_results=5, time_range=time_range)

        # Should return SearchResults
        assert result is not None
        assert isinstance(result, SearchResults)
        assert len(result.results) > 0

    def test_web_and_news_search_integration(self, searcher):
        """Test that both web and news search work together."""
        # Search web for general info
        web_result = searcher.search(query="climate change", max_results=5, time_range="y")

        # Search news for recent updates
        news_result = searcher.search_news(query="climate change", max_results=5, time_range="w")

        # Both should return results
        assert len(web_result.results) > 0
        assert len(news_result.results) > 0

    def test_special_characters_in_query(self, searcher):
        """Test that queries with special characters work."""
        result = searcher.search(query="C++ programming", max_results=5, time_range="y")

        # Should handle special characters and return results
        assert result is not None
        assert isinstance(result, SearchResults)
        assert len(result.results) > 0

    def test_search_result_structure(self, searcher):
        """Test that search results have expected structure."""
        result = searcher.search(query="Python documentation", max_results=5, time_range="y")

        # Should have results with titles and URLs
        assert len(result.results) > 0
        first_result = result.results[0]
        assert first_result.title
        assert first_result.url
        assert first_result.snippet
