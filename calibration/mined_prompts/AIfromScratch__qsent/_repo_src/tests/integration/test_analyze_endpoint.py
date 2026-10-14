from unittest.mock import patch

from fastapi.testclient import TestClient

from qsf.api.main import app

client = TestClient(app)

MOCK_RESULT = {
    "ticker": "IONQ",
    "last_updated": "2026-02-17T00:00:00",
    "sentiment_score": 0.5,
    "data_points": 10,
    "breakdown": {"news_sentiment": 0.6, "social_sentiment": 0.4, "trend": "stable"},
    "daily_data": [
        {"date": "2026-01-10", "close": 48.0, "volume": 1000000, "ror": None,
         "news_sentiment": 0.5, "social_sentiment": 0.4},
    ],
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("qsf.api.main.pipeline")
def test_analyze_valid_ticker(mock_pipeline):
    mock_pipeline.invoke.return_value = {"result": MOCK_RESULT}

    response = client.post("/analyze", json={"ticker": "IONQ"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "IONQ"
    assert "sentiment_score" in data
    assert "breakdown" in data
    assert "daily_data" in data
    assert data["breakdown"]["trend"] in ("increasing", "decreasing", "stable")


@patch("qsf.api.main.pipeline")
def test_analyze_returns_404_on_error(mock_pipeline):
    # NOTE: written pre-implementation — `detail` becomes a dict (with
    # "error" and "source_status" keys) once code-builder implements
    # per-source status tracking. This test is expected to fail (KeyError
    # on `["error"]`, since `detail` is currently a plain string) until then.
    mock_pipeline.invoke.return_value = {
        "error": "No price data found for 'FAKE123'",
        "source_status": {
            "market": "No price data found for 'FAKE123'",
            "news": "skipped",
            "reddit": "skipped",
            "sentiment": "skipped",
        },
    }

    response = client.post("/analyze", json={"ticker": "FAKE123"})
    assert response.status_code == 404
    assert "FAKE123" in response.json()["detail"]["error"]


@patch("qsf.api.main.pipeline")
def test_analyze_error_response_includes_source_status(mock_pipeline):
    # NOTE: written pre-implementation — expected to fail until code-builder
    # changes the /analyze 404 handler to return a dict `detail` containing
    # both "error" and "source_status".
    mock_pipeline.invoke.return_value = {
        "error": "Failed to fetch Reddit data: received 401 HTTP response",
        "source_status": {
            "market": "ok",
            "news": "ok",
            "reddit": "Failed to fetch Reddit data: received 401 HTTP response",
            "sentiment": "skipped",
        },
    }

    response = client.post("/analyze", json={"ticker": "IONQ"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    assert "error" in detail
    assert "source_status" in detail
    assert isinstance(detail["source_status"], dict)


def test_analyze_missing_ticker():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


@patch("qsf.api.main.pipeline")
def test_analyze_ticker_is_uppercased(mock_pipeline):
    mock_pipeline.invoke.return_value = {"result": MOCK_RESULT}

    client.post("/analyze", json={"ticker": "ionq"})
    call_args = mock_pipeline.invoke.call_args[0][0]
    assert call_args["ticker"] == "IONQ"
