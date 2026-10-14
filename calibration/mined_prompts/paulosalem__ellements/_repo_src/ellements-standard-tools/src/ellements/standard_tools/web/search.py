"""Web search functionality using DDGS with backend fallback."""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from ellements.core import ToolRegistry
from ellements.core.exceptions import LLMError
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a single search result."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    snippet: str = Field(description="Brief description/snippet of the result")
    source: str = Field(default="google", description="Search backend source")


class SearchResults(BaseModel):
    """Container for multiple search results."""

    query: str = Field(description="The search query that was executed")
    results: list[SearchResult] = Field(description="List of search results")
    total_results: int = Field(description="Total number of results returned")

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)

    def __iter__(self) -> Any:
        """Allow iteration over results."""
        return iter(self.results)

    def __getitem__(self, index: int) -> SearchResult:
        """Allow indexing of results."""
        return self.results[index]


class WebSearcher:
    """Web search client using DDGS."""

    _TEXT_BACKEND_DEFAULTS = ("google", "duckduckgo", "bing", "auto", "html", "lite")
    _NEWS_BACKEND_DEFAULTS = ("duckduckgo", "bing", "yahoo", "auto", "html", "lite")

    def __init__(
        self,
        region: str = "us-en",
        safe_search: str = "moderate",
        timeout: int = 10,
        max_results: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the web searcher.

        Args:
            region: Region for search results (e.g., 'us-en', 'uk-en', 'de-de')
            safe_search: Safe search setting ('strict', 'moderate', 'off')
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return
            **kwargs: Additional configuration for DDGS
        """
        self.region = region
        self.safe_search = safe_search
        self.timeout = timeout
        self.max_results = max_results
        self.config = kwargs

    def _backend_candidates(self, category: str, backend: str | None) -> list[str]:
        if backend:
            return [backend]
        defaults = (
            self._TEXT_BACKEND_DEFAULTS
            if category == "text"
            else self._NEWS_BACKEND_DEFAULTS
        )
        seen: set[str] = set()
        candidates: list[str] = []
        for candidate in defaults:
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
        return candidates

    def _run_ddgs_query(
        self,
        ddgs: DDGS,
        category: str,
        query: str,
        region: str,
        safesearch: str,
        timelimit: str | None,
        max_results: int,
        backend: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        method = ddgs.text if category == "text" else ddgs.news
        common_kwargs = {
            "region": region,
            "safesearch": safesearch,
            "timelimit": timelimit,
            "max_results": max_results,
            "backend": backend,
            **kwargs,
        }
        try:
            return list(method(query, **common_kwargs))
        except TypeError:
            return list(method(keywords=query, **common_kwargs))

    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
        safe_search: str | None = None,
        time_range: str | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> SearchResults:
        """Perform a web search.

        Args:
            query: Search query string
            max_results: Maximum number of results (overrides default)
            region: Region override
            safe_search: Safe search override
            time_range: Time range filter ('d', 'w', 'm', 'y' for day, week, month, year)
            **kwargs: Additional search parameters

        Returns:
            SearchResults object containing the results
        """
        try:
            # Use provided parameters or fall back to instance defaults
            search_region = region or self.region
            search_safe = safe_search or self.safe_search
            search_max = max_results or self.max_results

            backend_candidates = self._backend_candidates("text", backend)
            last_error: Exception | None = None

            with DDGS(timeout=self.timeout, **self.config) as ddgs:
                for candidate in backend_candidates:
                    try:
                        raw_results = self._run_ddgs_query(
                            ddgs,
                            "text",
                            query=query,
                            region=search_region,
                            safesearch=search_safe,
                            timelimit=time_range,
                            max_results=search_max,
                            backend=candidate,
                            **kwargs,
                        )
                    except Exception as exc:
                        last_error = exc
                        continue

                    results = []
                    for result in raw_results:
                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("href", ""),
                            snippet=result.get("body", ""),
                            source=candidate,
                        )
                        results.append(search_result)

                    return SearchResults(
                        query=query, results=results, total_results=len(results)
                    )

            if last_error is not None:
                raise last_error
            return SearchResults(query=query, results=[], total_results=0)

        except Exception as e:
            raise LLMError(f"Web search failed: {e}") from e

    async def search_async(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
        safe_search: str | None = None,
        time_range: str | None = None,
        **kwargs: Any,
    ) -> SearchResults:
        """Async version of search."""
        # Run the synchronous search in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                query=query,
                max_results=max_results,
                region=region,
                safe_search=safe_search,
                time_range=time_range,
                **kwargs,
            ),
        )

    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
        safe_search: str | None = None,
        time_range: str | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> SearchResults:
        """Search for news articles.

        Args:
            query: Search query string
            max_results: Maximum number of results
            region: Region for search
            safe_search: Safe search setting
            time_range: Time range filter
            **kwargs: Additional parameters

        Returns:
            SearchResults with news articles
        """
        try:
            search_region = region or self.region
            search_safe = safe_search or self.safe_search
            search_max = max_results or self.max_results

            backend_candidates = self._backend_candidates("news", backend)
            last_error: Exception | None = None

            with DDGS(timeout=self.timeout, **self.config) as ddgs:
                for candidate in backend_candidates:
                    try:
                        raw_results = self._run_ddgs_query(
                            ddgs,
                            "news",
                            query=query,
                            region=search_region,
                            safesearch=search_safe,
                            timelimit=time_range,
                            max_results=search_max,
                            backend=candidate,
                            **kwargs,
                        )
                    except Exception as exc:
                        last_error = exc
                        continue

                    results = []
                    for result in raw_results:
                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            snippet=result.get("body", ""),
                            source=f"{candidate}_news",
                        )
                        results.append(search_result)

                    return SearchResults(
                        query=query, results=results, total_results=len(results)
                    )

            if last_error is not None:
                raise last_error
            return SearchResults(query=query, results=[], total_results=0)

        except Exception as e:
            raise LLMError(f"News search failed: {e}") from e

    async def search_news_async(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
        safe_search: str | None = None,
        time_range: str | None = None,
        **kwargs: Any,
    ) -> SearchResults:
        """Async version of news search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search_news(
                query=query,
                max_results=max_results,
                region=region,
                safe_search=safe_search,
                time_range=time_range,
                **kwargs,
            ),
        )

    def quick_search(self, query: str, num_results: int = 5) -> list[str]:
        """Quick search returning just URLs.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of URLs
        """
        results = self.search(query, max_results=num_results)
        return [result.url for result in results.results]

    def search_and_summarize(self, query: str, max_results: int = 3) -> str:
        """Search and create a text summary of results.

        Args:
            query: Search query
            max_results: Number of results to include

        Returns:
            Formatted text summary
        """
        results = self.search(query, max_results=max_results)

        summary_lines = [f"Search results for: {query}\n"]

        for i, result in enumerate(results.results, 1):
            summary_lines.append(f"{i}. {result.title}")
            summary_lines.append(f"   URL: {result.url}")
            summary_lines.append(f"   {result.snippet}\n")

        return "\n".join(summary_lines)


# Agent Tools
# ---------------------------------------------------------------------------


def web_search_tools() -> ToolRegistry:
    """Create web search tools for agents.

    These tools are returned as plain callables. Framework-specific backends
    adapt them when needed.

    Returns:
        Dict of plain tool callables ready for adaptation by agent frameworks

    Example:
        >>> tools = web_search_tools()
        >>> agent = AgentBuilder("WebResearcher").with_tools(tools).build()
    """

    # Create searcher instance
    searcher = WebSearcher()

    def search_web(
        query: str, max_results: int = 10, time_range: str | None = None
    ) -> SearchResults:
        """Search the web using DDGS (Google backend by default).

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10)
            time_range: Filter by time range - 'd' (day), 'w' (week), 'm' (month), 'y' (year)

        Returns:
            SearchResults object with web pages matching the query
        """
        return searcher.search(query, max_results=max_results, time_range=time_range)

    def search_news(
        query: str, max_results: int = 10, time_range: str | None = None
    ) -> SearchResults:
        """Search for news articles using DDGS News backends.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10)
            time_range: Filter by time range - 'd' (day), 'w' (week), 'm' (month), 'y' (year)

        Returns:
            SearchResults object with news articles matching the query
        """
        return searcher.search_news(
            query, max_results=max_results, time_range=time_range
        )

    return ToolRegistry.from_mapping(
        {
            "search_web": search_web,
            "search_news": search_news,
        }
    )
