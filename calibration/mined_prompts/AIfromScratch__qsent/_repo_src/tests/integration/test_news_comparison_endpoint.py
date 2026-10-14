"""
Integration tests for the /diagnostics/news-comparison endpoint.
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from qsf.api.main import app

client = TestClient(app)

MOCK_COMPARISON_RESULT = {
    "tickers": ["IONQ"],
    "results": [
        {
            "ticker": "IONQ",
            "company": "IonQ",
            "providers": [
                {"name": "NewsAPI", "total": 20, "relevant": 15, "precision": 0.75, "free_tier": "1,000 req/day (30-day history)", "error": None},
                {"name": "Massive", "total": 10, "relevant": 8, "precision": 0.8, "free_tier": "5 req/min, unlimited/day (2yr history)", "error": None},
            ],
        }
    ],
}


@patch("qsf.api.main.run_news_comparison")
def test_valid_request_returns_200_with_correct_shape(mock_runner):
    mock_runner.return_value = MOCK_COMPARISON_RESULT
    response = client.post("/diagnostics/news-comparison", json={"tickers": ["IONQ"]})
    assert response.status_code == 200
    data = response.json()
    assert "tickers" in data
    assert "results" in data
    assert data["results"][0]["ticker"] == "IONQ"
    assert "providers" in data["results"][0]


@patch("qsf.api.main.run_news_comparison")
def test_tickers_are_uppercased(mock_runner):
    mock_runner.return_value = MOCK_COMPARISON_RESULT
    client.post("/diagnostics/news-comparison", json={"tickers": ["ionq", "form"]})
    call_args = mock_runner.call_args[0][0]
    assert "IONQ" in call_args
    assert "FORM" in call_args


@patch("qsf.api.main.run_news_comparison")
def test_whitespace_tickers_filtered_out(mock_runner):
    mock_runner.return_value = MOCK_COMPARISON_RESULT
    client.post("/diagnostics/news-comparison", json={"tickers": ["IONQ", "  ", "FORM"]})
    call_args = mock_runner.call_args[0][0]
    assert "  " not in call_args
    assert len(call_args) == 2


def test_all_empty_tickers_returns_422():
    response = client.post("/diagnostics/news-comparison", json={"tickers": ["  ", ""]})
    assert response.status_code == 422


def test_missing_tickers_field_returns_422():
    response = client.post("/diagnostics/news-comparison", json={})
    assert response.status_code == 422


@patch("qsf.api.main.run_news_comparison")
def test_provider_error_returns_200_with_error_in_payload(mock_runner):
    """Provider-level errors are captured in the payload, not as HTTP errors."""
    result_with_error = {
        "tickers": ["IONQ"],
        "results": [
            {
                "ticker": "IONQ",
                "company": "IonQ",
                "providers": [
                    {"name": "NewsAPI", "total": 0, "relevant": 0, "precision": 0.0,
                     "free_tier": "1,000 req/day (30-day history)", "error": "ValueError: NEWS_API_KEY not set"},
                ],
            }
        ],
    }
    mock_runner.return_value = result_with_error
    response = client.post("/diagnostics/news-comparison", json={"tickers": ["IONQ"]})
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["providers"][0]["error"] is not None


# ---------------------------------------------------------------------------
# SSE streaming endpoint tests
# ---------------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into list of {event, data} dicts."""
    messages = []
    for block in raw.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event is not None and data is not None:
            messages.append({"event": event, "data": data})
    return messages


def _mock_stream(tickers):
    def _sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
    yield _sse("ticker_start", {"ticker": "IONQ", "company": "IonQ"})
    yield _sse("provider_start", {"ticker": "IONQ", "provider": "NewsAPI", "total": 1})
    yield _sse("article_result", {
        "ticker": "IONQ", "provider": "NewsAPI",
        "index": 0, "text": "IonQ article", "date": "2026-02-10", "relevant": True,
    })
    yield _sse("provider_done", {
        "ticker": "IONQ", "provider": "NewsAPI",
        "total": 1, "relevant": 1, "precision": 1.0,
        "free_tier": "1,000 req/day (30-day history)", "error": None,
    })
    yield _sse("done", {})


@patch("qsf.api.main.run_news_comparison_stream")
def test_stream_returns_text_event_stream(mock_stream):
    mock_stream.side_effect = _mock_stream
    response = client.get("/diagnostics/news-comparison/stream?tickers=IONQ")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@patch("qsf.api.main.run_news_comparison_stream")
def test_stream_emits_all_expected_event_types(mock_stream):
    mock_stream.side_effect = _mock_stream
    response = client.get("/diagnostics/news-comparison/stream?tickers=IONQ")
    event_types = [m["event"] for m in _parse_sse(response.text)]
    for expected in ("ticker_start", "provider_start", "article_result", "provider_done", "done"):
        assert expected in event_types


@patch("qsf.api.main.run_news_comparison_stream")
def test_stream_ticker_start_payload(mock_stream):
    mock_stream.side_effect = _mock_stream
    response = client.get("/diagnostics/news-comparison/stream?tickers=IONQ")
    messages = _parse_sse(response.text)
    start = next(m for m in messages if m["event"] == "ticker_start")
    assert start["data"]["ticker"] == "IONQ"
    assert start["data"]["company"] == "IonQ"


@patch("qsf.api.main.run_news_comparison_stream")
def test_stream_article_result_has_required_fields(mock_stream):
    mock_stream.side_effect = _mock_stream
    response = client.get("/diagnostics/news-comparison/stream?tickers=IONQ")
    messages = _parse_sse(response.text)
    ar = next(m for m in messages if m["event"] == "article_result")
    for field in ("ticker", "provider", "index", "text", "date", "relevant"):
        assert field in ar["data"]


@patch("qsf.api.main.run_news_comparison_stream")
def test_stream_done_event_is_last(mock_stream):
    mock_stream.side_effect = _mock_stream
    response = client.get("/diagnostics/news-comparison/stream?tickers=IONQ")
    messages = _parse_sse(response.text)
    assert messages[-1]["event"] == "done"


def test_stream_empty_tickers_returns_422():
    response = client.get("/diagnostics/news-comparison/stream?tickers=&tickers=")
    assert response.status_code == 422


@patch("qsf.api.main.run_news_comparison_stream")
def test_stream_tickers_are_uppercased(mock_stream):
    mock_stream.side_effect = _mock_stream
    client.get("/diagnostics/news-comparison/stream?tickers=ionq")
    call_args = mock_stream.call_args[0][0]
    assert "IONQ" in call_args
