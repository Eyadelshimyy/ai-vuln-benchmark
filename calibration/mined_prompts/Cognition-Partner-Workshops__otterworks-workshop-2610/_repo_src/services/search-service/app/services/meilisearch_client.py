"""MeiliSearch client for full-text search operations."""

from __future__ import annotations

import threading
import time
from typing import Any

import meilisearch
import structlog

from app.config import MeiliSearchConfig
from app.models.search_result import AnalyticsData, SearchHit, SearchResponse

logger = structlog.get_logger()

FILES_INDEX = "files"
DOCUMENTS_INDEX = "documents"

# In-memory analytics store
_analytics_lock = threading.Lock()
_search_analytics: dict[str, Any] = {
    "queries": [],
    "total_searches": 0,
    "total_results": 0,
}

MAX_ANALYTICS_ENTRIES = 10000


def record_search_analytics(query: str, result_count: int) -> None:
    """Record a search query for analytics purposes."""
    with _analytics_lock:
        _search_analytics["queries"].append(
            {"query": query, "result_count": result_count, "timestamp": time.time()}
        )
        _search_analytics["total_searches"] += 1
        _search_analytics["total_results"] += result_count
        if len(_search_analytics["queries"]) > MAX_ANALYTICS_ENTRIES:
            _search_analytics["queries"] = _search_analytics["queries"][-MAX_ANALYTICS_ENTRIES:]


def get_search_analytics() -> AnalyticsData:
    """Compute search analytics from recorded queries."""
    with _analytics_lock:
        queries = list(_search_analytics["queries"])
        total_searches = _search_analytics["total_searches"]
        total_results = _search_analytics["total_results"]

    query_counts: dict[str, int] = {}
    zero_result_counts: dict[str, int] = {}
    for entry in queries:
        q = entry["query"]
        query_counts[q] = query_counts.get(q, 0) + 1
        if entry["result_count"] == 0:
            zero_result_counts[q] = zero_result_counts.get(q, 0) + 1

    popular = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    zero_results = sorted(zero_result_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    avg_results = total_results / total_searches if total_searches > 0 else 0.0

    return AnalyticsData(
        popular_queries=[{"query": q, "count": c} for q, c in popular],
        zero_result_queries=[{"query": q, "count": c} for q, c in zero_results],
        total_searches=total_searches,
        avg_results_per_query=round(avg_results, 2),
    )


class MeiliSearchService:
    """Client for MeiliSearch search and indexing operations."""

    def __init__(self, config: MeiliSearchConfig) -> None:
        self.config = config
        self.client = meilisearch.Client(config.url, config.api_key or None)
        self.documents_index_name = config.documents_index
        self.files_index_name = config.files_index

    @staticmethod
    def _escape(value: str) -> str:
        """Escape a value for use in MeiliSearch filter expressions."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def ensure_indices(self) -> None:
        """Create indices and configure settings if they don't exist."""
        for index_name in [self.documents_index_name, self.files_index_name]:
            try:
                self.client.get_index(index_name)
            except meilisearch.errors.MeilisearchApiError:
                task = self.client.create_index(index_name, {"primaryKey": "id"})
                self.client.wait_for_task(task.task_uid, timeout_in_ms=30000)
                logger.info("meilisearch_index_created", index=index_name)

        # Configure documents index
        docs_index = self.client.index(self.documents_index_name)
        docs_index.update_searchable_attributes(["title", "content", "tags"])
        docs_index.update_filterable_attributes(["type", "owner_id", "tags", "created_at", "updated_at"])
        docs_index.update_sortable_attributes(["updated_at", "created_at"])
        docs_index.update_ranking_rules([
            "words", "typo", "proximity", "attribute", "sort", "exactness",
        ])

        # Configure files index
        files_index = self.client.index(self.files_index_name)
        files_index.update_searchable_attributes(["name", "tags", "mime_type"])
        files_index.update_filterable_attributes(["type", "owner_id", "mime_type", "folder_id", "tags", "created_at", "updated_at"])
        files_index.update_sortable_attributes(["updated_at", "created_at", "size"])
        files_index.update_ranking_rules([
            "words", "typo", "proximity", "attribute", "sort", "exactness",
        ])

        logger.info("meilisearch_indices_configured")

    def ping(self) -> bool:
        """Check if MeiliSearch is reachable."""
        try:
            self.client.health()
            return True
        except Exception:
            return False

    def _build_search_params(
        self,
        page: int,
        page_size: int,
        filter_parts: list[str],
        multi_index: bool,
    ) -> dict[str, Any]:
        """Build MeiliSearch search parameters with correct pagination."""
        if multi_index:
            fetch_limit = page * page_size
            offset = 0
        else:
            fetch_limit = page_size
            offset = (page - 1) * page_size

        params: dict[str, Any] = {
            "offset": offset,
            "limit": fetch_limit,
            "attributesToHighlight": ["title", "name", "content"],
            "highlightPreTag": "<em>",
            "highlightPostTag": "</em>",
            "attributesToCrop": ["content"],
            "cropLength": 200,
        }
        if filter_parts:
            params["filter"] = " AND ".join(filter_parts)
        return params

    def search(
        self,
        query: str,
        doc_type: str | None = None,
        owner_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> SearchResponse:
        """Full-text search across documents and files."""
        filter_parts: list[str] = []
        if doc_type:
            filter_parts.append(f'type = "{self._escape(doc_type)}"')
        if owner_id:
            filter_parts.append(f'owner_id = "{self._escape(owner_id)}"')

        indices_to_search = self._resolve_indices(doc_type)
        multi_index = len(indices_to_search) > 1
        search_params = self._build_search_params(page, page_size, filter_parts, multi_index)

        all_hits: list[SearchHit] = []
        total = 0

        for index_name in indices_to_search:
            index = self.client.index(index_name)
            try:
                result = index.search(query, search_params)
            except meilisearch.errors.MeilisearchApiError as exc:
                logger.warning("search_filter_error", index=index_name, error=str(exc))
                raise ValueError(f"Invalid search filter: {exc}") from exc
            total += result["estimatedTotalHits"]
            for hit in result["hits"]:
                all_hits.append(self._parse_hit(hit, index_name))

        record_search_analytics(query, total)

        start = (page - 1) * page_size if multi_index else 0
        page_hits = all_hits[start : start + page_size]

        return SearchResponse(
            results=page_hits,
            total=total,
            page=page,
            page_size=page_size,
            query=query,
        )

    def advanced_search(
        self,
        query: str | None = None,
        doc_type: str | None = None,
        owner_id: str | None = None,
        tags: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> SearchResponse:
        """Advanced search with detailed filters."""
        filter_parts: list[str] = []
        if doc_type:
            filter_parts.append(f'type = "{self._escape(doc_type)}"')
        if owner_id:
            filter_parts.append(f'owner_id = "{self._escape(owner_id)}"')
        if tags:
            tag_filters = [f'tags = "{self._escape(tag)}"' for tag in tags]
            filter_parts.append(f'({" OR ".join(tag_filters)})')
        if date_from:
            filter_parts.append(f'created_at >= "{self._escape(date_from)}"')
        if date_to:
            filter_parts.append(f'created_at <= "{self._escape(date_to)}"')

        search_term = query or ""
        indices_to_search = self._resolve_indices(doc_type)
        multi_index = len(indices_to_search) > 1
        search_params = self._build_search_params(page, page_size, filter_parts, multi_index)

        all_hits: list[SearchHit] = []
        total = 0

        for index_name in indices_to_search:
            index = self.client.index(index_name)
            result = index.search(search_term, search_params)
            total += result["estimatedTotalHits"]
            for hit in result["hits"]:
                all_hits.append(self._parse_hit(hit, index_name))

        record_search_analytics(search_term or "*", total)

        start = (page - 1) * page_size if multi_index else 0
        page_hits = all_hits[start : start + page_size]

        return SearchResponse(
            results=page_hits,
            total=total,
            page=page,
            page_size=page_size,
            query=search_term or "*",
        )

    def suggest(self, prefix: str, size: int = 10) -> list[str]:
        """Autocomplete suggestions using MeiliSearch prefix matching."""
        suggestions: list[str] = []
        seen: set[str] = set()

        for index_name in [self.documents_index_name, self.files_index_name]:
            index = self.client.index(index_name)
            result = index.search(prefix, {
                "limit": size,
                "attributesToRetrieve": ["title", "name"],
            })
            for hit in result["hits"]:
                text = hit.get("title") or hit.get("name", "")
                if text and text not in seen:
                    suggestions.append(text)
                    seen.add(text)
                    if len(suggestions) >= size:
                        break
            if len(suggestions) >= size:
                break

        return suggestions

    def _wait_and_check(self, task_uid: int, timeout_in_ms: int = 10000) -> None:
        """Wait for a MeiliSearch task and raise on failure."""
        result = self.client.wait_for_task(task_uid, timeout_in_ms=timeout_in_ms)
        status = getattr(result, "status", None) or (result.get("status") if isinstance(result, dict) else None)
        if status and status != "succeeded":
            error = getattr(result, "error", None) or (result.get("error") if isinstance(result, dict) else None)
            raise RuntimeError(
                f"MeiliSearch task {task_uid} {status}: {error}"
            )

    def index_document(self, document: dict[str, Any]) -> None:
        """Index or update a document."""
        doc = {**document, "type": "document"}
        index = self.client.index(self.documents_index_name)
        task = index.add_documents([doc])
        self._wait_and_check(task.task_uid)
        logger.info("document_indexed", document_id=doc.get("id"))

    def index_file(self, file_data: dict[str, Any]) -> None:
        """Index or update a file.

        If the add fails with an LMDB key-exists error (a known
        Meilisearch bug triggered by delete-then-re-add on the same
        ID), the method retries once after explicitly deleting the
        stale entry.
        """
        doc = {**file_data, "type": "file"}
        index = self.client.index(self.files_index_name)
        task = index.add_documents([doc])
        try:
            self._wait_and_check(task.task_uid)
        except RuntimeError as exc:
            if "MDB_KEYEXIST" in str(exc):
                logger.warning("lmdb_key_exists_retry", file_id=doc.get("id"))
                del_task = index.delete_document(doc["id"])
                self._wait_and_check(del_task.task_uid)
                retry_task = index.add_documents([doc])
                self._wait_and_check(retry_task.task_uid)
            else:
                raise
        logger.info("file_indexed", file_id=doc.get("id"))

    def delete_document(self, doc_type: str, doc_id: str) -> bool:
        """Remove a document or file from the index. Returns False if not found."""
        index_name = self.documents_index_name if doc_type == "document" else self.files_index_name
        index = self.client.index(index_name)
        try:
            index.get_document(doc_id)
        except meilisearch.errors.MeilisearchApiError:
            logger.warning("document_not_found_in_index", doc_id=doc_id, index=index_name)
            return False
        task = index.delete_document(doc_id)
        self._wait_and_check(task.task_uid)
        logger.info("document_removed_from_index", doc_id=doc_id, index=index_name)
        return True

    def reindex(
        self,
        documents: list[dict[str, Any]] | None = None,
        files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Delete indices, recreate them, and optionally repopulate.

        If *documents* or *files* lists are supplied they are bulk-indexed
        after the empty indices are created.  When called without data the
        indices are left empty (callers can populate them afterwards).
        """
        for index_name in [self.documents_index_name, self.files_index_name]:
            try:
                task = self.client.delete_index(index_name)
                self._wait_and_check(task.task_uid, timeout_in_ms=30000)
                logger.info("meilisearch_index_deleted", index=index_name)
            except (meilisearch.errors.MeilisearchApiError, RuntimeError):
                pass
        self.ensure_indices()

        indexed_counts: dict[str, int] = {"documents": 0, "files": 0}
        if documents:
            idx = self.client.index(self.documents_index_name)
            for batch_start in range(0, len(documents), 500):
                batch = documents[batch_start : batch_start + 500]
                task = idx.add_documents(batch)
                self._wait_and_check(task.task_uid, timeout_in_ms=60000)
            indexed_counts["documents"] = len(documents)
        if files:
            idx = self.client.index(self.files_index_name)
            for batch_start in range(0, len(files), 500):
                batch = files[batch_start : batch_start + 500]
                task = idx.add_documents(batch)
                self._wait_and_check(task.task_uid, timeout_in_ms=60000)
            indexed_counts["files"] = len(files)

        return {
            "status": "reindexed",
            "indices": [self.documents_index_name, self.files_index_name],
            "indexed_counts": indexed_counts,
        }

    def _resolve_indices(self, doc_type: str | None) -> list[str]:
        """Determine which indices to search based on type filter."""
        if doc_type == "document":
            return [self.documents_index_name]
        if doc_type == "file":
            return [self.files_index_name]
        return [self.documents_index_name, self.files_index_name]

    def _parse_hit(self, hit: dict[str, Any], index_name: str) -> SearchHit:
        """Convert a MeiliSearch hit to a SearchHit model."""
        formatted = hit.get("_formatted", {})
        highlights: dict[str, list[str]] = {}
        for field in ["title", "name", "content"]:
            if field in formatted and "<em>" in str(formatted[field]):
                highlights[field] = [str(formatted[field])]

        is_doc = index_name == self.documents_index_name
        return SearchHit(
            id=hit.get("id", ""),
            title=hit.get("title", "") if is_doc else hit.get("name", ""),
            content_snippet=str(formatted.get("content", ""))[:200] if is_doc else "",
            type=hit.get("type", "document" if is_doc else "file"),
            owner_id=hit.get("owner_id", ""),
            tags=hit.get("tags", []),
            score=0.0,
            highlights=highlights,
            created_at=hit.get("created_at"),
            updated_at=hit.get("updated_at"),
            mime_type=hit.get("mime_type"),
            folder_id=hit.get("folder_id"),
            size=hit.get("size"),
        )
