"""Tests for the Isaacus enrichment API wrapper and response conversion.

The isaacus SDK is a core dependency (always installed).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Skip only on a broken install where the isaacus SDK cannot be imported
pytest.importorskip("isaacus", reason="isaacus SDK not importable (broken install?)")

from womblex.analyse.enrich import (
    _convert_document,
    _to_span,
    _to_span_list,
    enrich_document,
    enrich_documents,
)
from womblex.analyse.models import EnrichmentResult

# ---------------------------------------------------------------------------
# Fixtures: mock SDK objects
# ---------------------------------------------------------------------------


def _mock_span(start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end)


def _make_mock_document(text: str = "Hello world. This is a test document.") -> SimpleNamespace:
    """Build a mock ILGS Document mirroring the Isaacus SDK response."""
    return SimpleNamespace(
        text=text,
        type="other",
        jurisdiction="AU",
        title=_mock_span(0, 11),
        subtitle=None,
        segments=[
            SimpleNamespace(
                id="seg:0",
                kind="unit",
                type="paragraph",
                category="main",
                span=_mock_span(0, len(text)),
                parent=None,
                children=[],
                level=0,
                type_name=None,
                code=None,
                title=None,
            ),
        ],
        crossreferences=[],
        locations=[
            SimpleNamespace(
                id="loc:0",
                name=_mock_span(0, 5),  # "Hello"
                type="city",
                mentions=[_mock_span(0, 5)],
                parent=None,
                children=[],
            ),
        ],
        persons=[
            SimpleNamespace(
                id="per:0",
                name=_mock_span(13, 17),  # "This"
                type="corporate",
                role="other",
                mentions=[_mock_span(13, 17)],
                parent=None,
                children=[],
                residence="loc:0",
            ),
        ],
        emails=[],
        websites=[],
        phone_numbers=[],
        id_numbers=[],
        terms=[
            SimpleNamespace(
                id="term:0",
                name=_mock_span(23, 27),  # "test"
                meaning=_mock_span(23, 36),  # "test document."
                mentions=[],
            ),
        ],
        external_documents=[],
        quotes=[],
        dates=[
            SimpleNamespace(
                value="2024-01-15",
                type="creation",
                mentions=[_mock_span(0, 5)],
                person=None,
            ),
        ],
        headings=[_mock_span(0, 11)],
        junk=[],
    )


def _make_mock_client(text: str = "Hello world. This is a test document.") -> MagicMock:
    """Build a mock Isaacus client that returns a mock enrichment response."""
    doc = _make_mock_document(text)
    response = SimpleNamespace(
        results=[SimpleNamespace(index=0, document=doc)],
        usage=SimpleNamespace(input_tokens=42),
    )
    client = MagicMock()
    client.enrichments.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Tests: span conversion
# ---------------------------------------------------------------------------


class TestSpanConversion:
    def test_to_span(self) -> None:
        raw = _mock_span(10, 20)
        span = _to_span(raw)
        assert span is not None
        assert span.start == 10
        assert span.end == 20

    def test_to_span_none(self) -> None:
        assert _to_span(None) is None

    def test_to_span_list(self) -> None:
        raw = [_mock_span(0, 5), _mock_span(10, 15)]
        spans = _to_span_list(raw)
        assert len(spans) == 2
        assert spans[0].start == 0
        assert spans[1].end == 15

    def test_to_span_list_empty(self) -> None:
        assert _to_span_list(None) == []
        assert _to_span_list([]) == []


# ---------------------------------------------------------------------------
# Tests: document conversion
# ---------------------------------------------------------------------------


class TestDocumentConversion:
    def test_convert_document_basic(self) -> None:
        doc = _make_mock_document()
        result = _convert_document(doc)

        assert isinstance(result, EnrichmentResult)
        assert result.type == "other"
        assert result.jurisdiction == "AU"
        assert result.title is not None
        assert result.title.start == 0
        assert result.title.end == 11

    def test_convert_segments(self) -> None:
        doc = _make_mock_document()
        result = _convert_document(doc)

        assert len(result.segments) == 1
        seg = result.segments[0]
        assert seg.id == "seg:0"
        assert seg.kind == "unit"
        assert seg.type == "paragraph"
        assert seg.category == "main"
        assert seg.level == 0

    def test_convert_persons(self) -> None:
        doc = _make_mock_document()
        result = _convert_document(doc)

        assert len(result.persons) == 1
        per = result.persons[0]
        assert per.id == "per:0"
        assert per.type == "corporate"
        assert per.role == "other"
        assert per.residence == "loc:0"
        assert len(per.mentions) == 1

    def test_convert_locations(self) -> None:
        doc = _make_mock_document()
        result = _convert_document(doc)

        assert len(result.locations) == 1
        loc = result.locations[0]
        assert loc.id == "loc:0"
        assert loc.type == "city"

    def test_convert_terms(self) -> None:
        doc = _make_mock_document()
        result = _convert_document(doc)

        assert len(result.terms) == 1
        term = result.terms[0]
        assert term.id == "term:0"

    def test_convert_dates(self) -> None:
        doc = _make_mock_document()
        result = _convert_document(doc)

        assert len(result.dates) == 1
        assert result.dates[0].value == "2024-01-15"
        assert result.dates[0].type == "creation"

    def test_convert_headings(self) -> None:
        doc = _make_mock_document()
        result = _convert_document(doc)

        assert len(result.headings) == 1
        assert result.headings[0].start == 0


# ---------------------------------------------------------------------------
# Tests: enrichment API calls
# ---------------------------------------------------------------------------


class TestEnrichDocument:
    def test_enrich_single_document(self) -> None:
        client = _make_mock_client()
        result = enrich_document("Hello world. This is a test document.", client)

        assert isinstance(result, EnrichmentResult)
        assert result.type == "other"
        client.enrichments.create.assert_called_once_with(
            model="kanon-2-enricher",
            texts=["Hello world. This is a test document."],
            overflow_strategy="auto",
        )

    def test_enrich_multiple_documents(self) -> None:
        client = _make_mock_client()
        # The mock returns one result, so enrich_documents with one text
        results = enrich_documents(["Hello world. This is a test document."], client)

        assert len(results) == 1
        assert isinstance(results[0], EnrichmentResult)

    def test_enrich_retry_on_rate_limit(self) -> None:
        client = MagicMock()
        # First call raises 429, second succeeds
        doc = _make_mock_document()
        response = SimpleNamespace(
            results=[SimpleNamespace(index=0, document=doc)],
            usage=SimpleNamespace(input_tokens=42),
        )
        client.enrichments.create.side_effect = [
            Exception("429 rate limit exceeded"),
            response,
        ]

        result = enrich_document(
            "Test text",
            client,
            retry_base_delay=0.01,  # fast retries for test
        )
        assert isinstance(result, EnrichmentResult)
        assert client.enrichments.create.call_count == 2

    def test_enrich_non_retryable_error(self) -> None:
        client = MagicMock()
        client.enrichments.create.side_effect = Exception("Internal server error")

        with pytest.raises(RuntimeError, match="Enrichment failed"):
            enrich_document("Test text", client)

    def test_enrich_exhausted_retries(self) -> None:
        client = MagicMock()
        client.enrichments.create.side_effect = Exception("429 rate limit exceeded")

        with pytest.raises(RuntimeError, match="after 4 attempts"):
            enrich_document(
                "Test text",
                client,
                max_retries=3,
                retry_base_delay=0.01,
            )
