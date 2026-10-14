"""
Unit tests for auth helpers and protected routes.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from qsf.api.auth import (
    ALGORITHM,
    COOKIE_NAME,
    SECRET_KEY,
    create_session_token,
    decode_session_token,
)
from qsf.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def test_create_and_decode_session_token():
    token = create_session_token("user@example.com", "Test User")
    payload = decode_session_token(token)
    assert payload["sub"] == "user@example.com"
    assert payload["name"] == "Test User"


def test_create_session_token_with_picture():
    picture_url = "https://example.com/pic.jpg"
    token = create_session_token("user@example.com", "Test User", picture=picture_url)
    payload = decode_session_token(token)
    assert payload["sub"] == "user@example.com"
    assert payload["name"] == "Test User"
    assert payload["picture"] == picture_url


def test_create_session_token_picture_defaults_to_empty():
    token = create_session_token("user@example.com", "Test User")
    payload = decode_session_token(token)
    assert payload["picture"] == ""


def test_create_session_token_stores_google_sub():
    token = create_session_token("user@example.com", "Test User", google_sub="116234567890")
    payload = decode_session_token(token)
    assert payload["google_sub"] == "116234567890"


def test_create_session_token_google_sub_defaults_to_empty():
    token = create_session_token("user@example.com", "Test User")
    payload = decode_session_token(token)
    assert payload["google_sub"] == ""


def test_decode_expired_token_raises():
    from jose import JWTError

    expired = {
        "sub": "user@example.com",
        "name": "Test",
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    token = jwt.encode(expired, SECRET_KEY, algorithm=ALGORITHM)
    with pytest.raises(JWTError):
        decode_session_token(token)


# ---------------------------------------------------------------------------
# /auth/me
# ---------------------------------------------------------------------------


def test_me_requires_auth():
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_returns_user_with_valid_cookie():
    token = create_session_token("user@example.com", "Test User")
    response = client.get("/auth/me", cookies={COOKIE_NAME: token})
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@example.com"
    assert data["name"] == "Test User"
    assert data["picture"] == ""


def test_me_returns_google_sub():
    token = create_session_token("user@example.com", "Test User", google_sub="116234567890")
    response = client.get("/auth/me", cookies={COOKIE_NAME: token})
    assert response.status_code == 200
    assert response.json()["google_sub"] == "116234567890"


def test_me_returns_empty_google_sub_when_absent():
    token = create_session_token("user@example.com", "Test User")
    response = client.get("/auth/me", cookies={COOKIE_NAME: token})
    assert response.status_code == 200
    assert response.json()["google_sub"] == ""


def test_me_returns_picture_when_provided():
    picture_url = "https://example.com/pic.jpg"
    token = create_session_token("user@example.com", "Test User", picture=picture_url)
    response = client.get("/auth/me", cookies={COOKIE_NAME: token})
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@example.com"
    assert data["name"] == "Test User"
    assert data["picture"] == picture_url


def test_me_rejects_invalid_token():
    response = client.get("/auth/me", cookies={COOKIE_NAME: "not.a.valid.token"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# /auth/logout
# ---------------------------------------------------------------------------


def test_logout_clears_cookie():
    token = create_session_token("user@example.com", "Test User")
    response = client.get(
        "/auth/logout",
        cookies={COOKIE_NAME: token},
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert "/login.html" in response.headers["location"]
    # Cookie deletion appears as max-age=0 or expires in the past
    set_cookie = response.headers.get("set-cookie", "")
    assert COOKIE_NAME in set_cookie


# ---------------------------------------------------------------------------
# /auth/callback — CSRF state validation
# ---------------------------------------------------------------------------


def test_callback_rejects_mismatched_state():
    response = client.get(
        "/auth/callback?code=fake-code&state=wrong",
        cookies={"oauth_state": "correct"},
    )
    assert response.status_code == 400


def test_callback_rejects_missing_state_cookie():
    response = client.get("/auth/callback?code=fake-code&state=some-state")
    assert response.status_code == 400


def test_callback_rejects_missing_code():
    state = "test-state-abc"
    response = client.get(
        f"/auth/callback?state={state}",
        cookies={"oauth_state": state},
    )
    assert response.status_code == 400


def test_callback_sets_session_cookie(mocker):
    state = "test-state-xyz"

    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=None)
    mock_instance.fetch_token = AsyncMock(return_value={"access_token": "fake-token"})
    mock_userinfo = MagicMock()
    mock_userinfo.json.return_value = {"email": "user@example.com", "name": "Test User"}
    mock_instance.get = AsyncMock(return_value=mock_userinfo)

    mocker.patch("qsf.api.auth.AsyncOAuth2Client", return_value=mock_instance)

    response = client.get(
        f"/auth/callback?code=fake-code&state={state}",
        cookies={"oauth_state": state},
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert COOKIE_NAME in response.cookies


def test_callback_captures_google_sub(mocker):
    state = "test-state-with-sub"

    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=None)
    mock_instance.fetch_token = AsyncMock(return_value={"access_token": "fake-token"})
    mock_userinfo = MagicMock()
    mock_userinfo.json.return_value = {
        "email": "user@example.com",
        "name": "Test User",
        "sub": "116234567890",
    }
    mock_instance.get = AsyncMock(return_value=mock_userinfo)
    mocker.patch("qsf.api.auth.AsyncOAuth2Client", return_value=mock_instance)

    response = client.get(
        f"/auth/callback?code=fake-code&state={state}",
        cookies={"oauth_state": state},
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    session_cookie = response.cookies[COOKIE_NAME]
    payload = decode_session_token(session_cookie)
    assert payload["google_sub"] == "116234567890"


# ---------------------------------------------------------------------------
# /auth/login
# ---------------------------------------------------------------------------


def test_sanitize_picture_rejects_non_google_url():
    from qsf.api.auth import _sanitize_picture
    assert _sanitize_picture("https://evil.com/img.jpg") == ""


def test_sanitize_picture_rejects_javascript_uri():
    from qsf.api.auth import _sanitize_picture
    assert _sanitize_picture("javascript:alert(1)") == ""


def test_sanitize_picture_accepts_google_url():
    from qsf.api.auth import _sanitize_picture
    url = "https://lh3.googleusercontent.com/a/photo.jpg"
    assert _sanitize_picture(url) == url


def test_sanitize_picture_rejects_empty_string():
    from qsf.api.auth import _sanitize_picture
    assert _sanitize_picture("") == ""


def test_login_redirects_to_google_and_sets_state_cookie(mocker):
    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=None)
    mock_instance.create_authorization_url = MagicMock(
        return_value=("https://accounts.google.com/o/oauth2/v2/auth?client_id=test", "state123")
    )

    mocker.patch("qsf.api.auth.AsyncOAuth2Client", return_value=mock_instance)

    response = client.get("/auth/login", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert "accounts.google.com" in response.headers["location"]
    assert "oauth_state" in response.cookies
