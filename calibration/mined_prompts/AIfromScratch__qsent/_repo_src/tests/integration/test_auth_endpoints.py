"""
Integration tests for auth enforcement on protected endpoints.

These tests intentionally do NOT use the bypass_auth fixture from conftest.py.
The autouse fixture applies only when get_current_user is not overridden —
here we verify real 401 behaviour by clearing any override before each test.
"""
import pytest
from fastapi.testclient import TestClient

from qsf.api.auth import COOKIE_NAME, create_session_token, get_current_user
from qsf.api.main import app


@pytest.fixture(autouse=True)
def clear_auth_override():
    # Remove the bypass set by the integration conftest so real auth runs
    app.dependency_overrides.pop(get_current_user, None)
    yield
    app.dependency_overrides.pop(get_current_user, None)


client = TestClient(app)


def test_analyze_requires_auth():
    response = client.post("/analyze", json={"ticker": "IONQ"})
    assert response.status_code == 401


def test_news_comparison_requires_auth():
    response = client.post("/diagnostics/news-comparison", json={"tickers": ["IONQ"]})
    assert response.status_code == 401


def test_health_no_auth_required():
    response = client.get("/health")
    assert response.status_code == 200


def test_stream_no_auth_required():
    # SSE endpoint is intentionally left open (EventSource doesn't support cookies)
    response = client.get(
        "/diagnostics/news-comparison/stream",
        params={"tickers": "IONQ"},
        headers={"Accept": "text/event-stream"},
    )
    # Will fail with a real error (no pipeline running), but NOT a 401
    assert response.status_code != 401


def test_analyze_succeeds_with_valid_cookie(mocker):
    mocker.patch(
        "qsf.api.main.pipeline",
        **{
            "invoke.return_value": {
                "result": {
                    "ticker": "IONQ",
                    "last_updated": "2026-02-17T00:00:00",
                    "sentiment_score": 0.5,
                    "data_points": 10,
                    "breakdown": {"news_sentiment": 0.6, "social_sentiment": 0.4, "trend": "stable"},
                    "daily_data": [],
                }
            }
        },
    )
    token = create_session_token("user@example.com", "Test User")
    response = client.post(
        "/analyze",
        json={"ticker": "IONQ"},
        cookies={COOKIE_NAME: token},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /auth/avatar
# ---------------------------------------------------------------------------


def test_avatar_requires_auth():
    response = client.get("/auth/avatar")
    assert response.status_code == 401


def test_avatar_returns_404_when_no_picture(mocker):
    token = create_session_token("user@example.com", "Test User", picture="")
    response = client.get("/auth/avatar", cookies={COOKIE_NAME: token})
    assert response.status_code == 404
    assert response.json()["detail"] == "No avatar"


def test_avatar_returns_404_when_picture_key_missing(mocker):
    token = create_session_token("user@example.com", "Test User")
    response = client.get("/auth/avatar", cookies={COOKIE_NAME: token})
    assert response.status_code == 404
    assert response.json()["detail"] == "No avatar"


def test_avatar_returns_404_on_failed_fetch(mocker):
    picture_url = "https://lh3.googleusercontent.com/a/test.jpg"
    token = create_session_token("user@example.com", "Test User", picture=picture_url)

    mock_response = mocker.AsyncMock()
    mock_response.status_code = 404
    mock_client = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch("qsf.api.main.httpx.AsyncClient", return_value=mock_client)

    response = client.get("/auth/avatar", cookies={COOKIE_NAME: token})
    assert response.status_code == 404
    assert response.json()["detail"] == "Avatar unavailable"


def test_avatar_returns_200_with_image_bytes(mocker):
    picture_url = "https://lh3.googleusercontent.com/a/test.jpg"
    token = create_session_token("user@example.com", "Test User", picture=picture_url)

    image_bytes = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
    mock_response = mocker.AsyncMock()
    mock_response.status_code = 200
    mock_response.content = image_bytes
    mock_response.headers = {"content-type": "image/png"}
    mock_client = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch("qsf.api.main.httpx.AsyncClient", return_value=mock_client)

    response = client.get("/auth/avatar", cookies={COOKIE_NAME: token})
    assert response.status_code == 200
    assert response.content == image_bytes
    assert response.headers["content-type"] == "image/png"


def test_avatar_defaults_to_jpeg_when_no_content_type(mocker):
    picture_url = "https://lh3.googleusercontent.com/a/test.jpg"
    token = create_session_token("user@example.com", "Test User", picture=picture_url)

    image_bytes = b"\xff\xd8\xff\xe0"  # JPEG magic bytes
    mock_response = mocker.AsyncMock()
    mock_response.status_code = 200
    mock_response.content = image_bytes
    mock_response.headers = {}
    mock_client = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch("qsf.api.main.httpx.AsyncClient", return_value=mock_client)

    response = client.get("/auth/avatar", cookies={COOKIE_NAME: token})
    assert response.status_code == 200
    assert response.content == image_bytes
    assert response.headers["content-type"] == "image/jpeg"


def test_avatar_follows_redirects(mocker):
    picture_url = "https://lh3.googleusercontent.com/a/test.jpg"
    token = create_session_token("user@example.com", "Test User", picture=picture_url)

    image_bytes = b"\x89PNG\r\n\x1a\n"
    mock_response = mocker.AsyncMock()
    mock_response.status_code = 200
    mock_response.content = image_bytes
    mock_response.headers = {"content-type": "image/png"}
    mock_client = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

    mocker.patch("qsf.api.main.httpx.AsyncClient", return_value=mock_client)

    response = client.get("/auth/avatar", cookies={COOKIE_NAME: token})
    mock_client.get.assert_called_once_with(picture_url, follow_redirects=True, timeout=5)
    assert response.status_code == 200
