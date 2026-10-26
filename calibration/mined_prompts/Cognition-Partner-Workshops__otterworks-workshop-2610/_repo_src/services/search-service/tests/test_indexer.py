"""Tests for the Indexer service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import MeiliSearchConfig
from app.services.indexer import Indexer
from app.services.meilisearch_client import MeiliSearchService


@pytest.fixture()
def mock_ms_service() -> MeiliSearchService:
    """Create a MeiliSearchService with mocked client."""
    with patch("app.services.meilisearch_client.meilisearch.Client") as mock_cls:
        mock_client = MagicMock()

        mock_task = MagicMock()
        mock_task.task_uid = 1
        mock_task_result = MagicMock()
        mock_task_result.status = "succeeded"
        mock_client.wait_for_task.return_value = mock_task_result

        mock_index = MagicMock()
        mock_index.add_documents.return_value = mock_task
        mock_index.delete_document.return_value = mock_task
        mock_index.get_document.return_value = {"id": "doc-1"}
        mock_client.index.return_value = mock_index

        mock_client.create_index.return_value = mock_task
        mock_client.delete_index.return_value = mock_task
        mock_client.get_index.side_effect = None
        mock_client.health.return_value = {"status": "available"}

        mock_cls.return_value = mock_client
        service = MeiliSearchService(MeiliSearchConfig(
            documents_index="test-docs",
            files_index="test-files",
        ))
        yield service


@pytest.fixture()
def indexer(mock_ms_service: MeiliSearchService) -> Indexer:
    return Indexer(mock_ms_service)


class TestIndexer:
    """Tests for Indexer logic."""

    def test_index_document_success(self, indexer: Indexer):
        result = indexer.index_document({
            "id": "doc-1",
            "title": "Test Doc",
            "content": "Hello world",
            "owner_id": "user-1",
        })
        assert result["status"] == "indexed"
        assert result["type"] == "document"

    def test_index_document_missing_id(self, indexer: Indexer):
        with pytest.raises(ValueError, match="id"):
            indexer.index_document({"title": "No ID"})

    def test_index_document_missing_title(self, indexer: Indexer):
        with pytest.raises(ValueError, match="title"):
            indexer.index_document({"id": "doc-1"})

    def test_index_file_success(self, indexer: Indexer):
        result = indexer.index_file({
            "id": "file-1",
            "name": "report.pdf",
            "mime_type": "application/pdf",
            "owner_id": "user-1",
        })
        assert result["status"] == "indexed"
        assert result["type"] == "file"

    def test_index_file_missing_id(self, indexer: Indexer):
        with pytest.raises(ValueError, match="id"):
            indexer.index_file({"name": "report.pdf"})

    def test_index_file_missing_name(self, indexer: Indexer):
        with pytest.raises(ValueError, match="name"):
            indexer.index_file({"id": "file-1"})

    def test_remove_document(self, indexer: Indexer):
        result = indexer.remove("document", "doc-1")
        assert result["status"] == "deleted"
        assert result["type"] == "document"

    def test_remove_invalid_type(self, indexer: Indexer):
        with pytest.raises(ValueError, match="Invalid type"):
            indexer.remove("invalid", "id-1")

    def test_process_event_index_document(self, indexer: Indexer):
        result = indexer.process_event({
            "action": "index_document",
            "data": {"id": "doc-1", "title": "Test"},
        })
        assert result is not None
        assert result["status"] == "indexed"

    def test_process_event_index_file(self, indexer: Indexer):
        result = indexer.process_event({
            "action": "index_file",
            "data": {"id": "file-1", "name": "test.pdf"},
        })
        assert result is not None
        assert result["status"] == "indexed"

    def test_process_event_delete(self, indexer: Indexer):
        result = indexer.process_event({
            "action": "delete",
            "data": {"id": "doc-1", "type": "document"},
        })
        assert result is not None
        assert result["status"] == "deleted"

    def test_process_event_unknown_action(self, indexer: Indexer):
        result = indexer.process_event({"action": "unknown", "data": {}})
        assert result is None
