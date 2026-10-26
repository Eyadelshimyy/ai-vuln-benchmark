"""Response models for Search Service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """A single search result."""

    id: str
    title: str
    content_snippet: str
    type: str  # "document" or "file"
    owner_id: str
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    highlights: dict[str, list[str]] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    mime_type: str | None = None
    folder_id: str | None = None
    size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "content_snippet": self.content_snippet,
            "type": self.type,
            "owner_id": self.owner_id,
            "tags": self.tags,
            "score": self.score,
            "highlights": self.highlights,
        }
        if self.created_at is not None:
            result["created_at"] = self.created_at
        if self.updated_at is not None:
            result["updated_at"] = self.updated_at
        if self.mime_type is not None:
            result["mime_type"] = self.mime_type
        if self.folder_id is not None:
            result["folder_id"] = self.folder_id
        if self.size is not None:
            result["size"] = self.size
        return result


@dataclass
class SearchResponse:
    """Paginated search response."""

    results: list[SearchHit]
    total: int
    page: int
    page_size: int
    query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [hit.to_dict() for hit in self.results],
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "query": self.query,
        }


@dataclass
class SuggestResponse:
    """Autocomplete suggestion response."""

    suggestions: list[str]
    query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": self.suggestions,
            "query": self.query,
        }


@dataclass
class AnalyticsData:
    """Search analytics data."""

    popular_queries: list[dict[str, Any]]
    zero_result_queries: list[dict[str, Any]]
    total_searches: int
    avg_results_per_query: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "popular_queries": self.popular_queries,
            "zero_result_queries": self.zero_result_queries,
            "total_searches": self.total_searches,
            "avg_results_per_query": self.avg_results_per_query,
        }
