"""``tracebi login`` / ``logout``, end to end against the served app.

The CLI is real: it starts its loopback listener, builds the authorization
URL, exchanges the code, writes the credentials file. Two seams are replaced:
the HTTP transport (routed into the in-process app instead of a socket) and
the browser (``webbrowser.open`` performs the person's part: the server's
/authorize, the fake identity provider, then the redirect to the CLI's
listener, over a real loopback socket).
"""

from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

pytest.importorskip("mcp")
pytest.importorskip("jwt")

from tests.e2e.conftest import run_cli  # noqa: E402
from tests.e2e.test_app_sso_journey import sso  # noqa: E402, F401
from tests.e2e.test_mcp_oauth_journey import (  # noqa: E402, F401
    PUBLIC, idp, provider_login, signin,
)
from tracebi import credentials  # noqa: E402


@pytest.fixture
def cli_env(sso, idp, tmp_path, monkeypatch):
    """The served app, a config directory of its own, and a browser."""
    c = sso.mcp_client
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    calls: list[tuple[str, str]] = []

    def fake_request(method, url, *, form=None, json_body=None, bearer=None):
        assert url.startswith(PUBLIC + "/"), url
        calls.append((method, url[len(PUBLIC):]))
        resp = c.request(method, url[len(PUBLIC):], data=form, json=json_body,
                         headers={"Authorization": f"Bearer {bearer}"} if bearer else {},
                         follow_redirects=False)
        try:
            body = resp.json()
        except ValueError:
            body = {}
        return resp.status_code, body if isinstance(body, dict) else {}

    monkeypatch.setattr(credentials, "_request", fake_request)

    browser = {"claims": {"groups": ["bi-analysts"]}, "visits": 0, "stray": None}

    def open_browser(url):
        browser["visits"] += 1
        target = urllib.parse.urlsplit(url)
        assert target.path == "/authorize"
        auth = c.get(f"{target.path}?{target.query}", follow_redirects=False)
        callback = provider_login(c, idp, auth, **browser["claims"])
        back = callback.headers["location"]
        assert back.startswith("http://127.0.0.1:"), back
        if browser["stray"]:                  # something else knocking on the listener first
            knock = back.split("?")[0] + "?" + browser["stray"]
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(knock, timeout=10)
            assert err.value.code == 404
        urllib.request.urlopen(back, timeout=10).read()
        return True

    monkeypatch.setattr(credentials.webbrowser, "open", open_browser)
    return c, browser, calls


def stored():
    return json.loads(credentials.credentials_path().read_text())["servers"][PUBLIC]


def test_login_signs_the_cli_in_and_keeps_the_tokens_private(cli_env):
    c, browser, calls = cli_env
    code, out = run_cli("login", "--server", PUBLIC)
    assert code == 0, out
    assert f"Signed in to {PUBLIC} as alice@example.com (analyst)." in out

    path = credentials.credentials_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    rec = stored()
    assert rec["access_token"].startswith("tbat_") and rec["refresh_token"].startswith("tbrt_")
    assert rec["client_id"] and rec["expires_at"] > time.time() + 3000
    assert rec["scope"] == "mcp offline_access"

    # What the file holds works on the app's API, as the person, with their role.
    me = c.get("/api/me", headers={"Authorization": f"Bearer {credentials.access_token(PUBLIC)}"})
    assert me.json() == {"actor": "alice@example.com", "role": "analyst", "sign_in": "oidc"}

    # The second sign-in reuses the client registration.
    registrations = [m for m, p in calls if (m, p) == ("POST", "/register")]
    assert len(registrations) == 1
    assert run_cli("login", "--server", PUBLIC)[0] == 0
    assert len([1 for m, p in calls if (m, p) == ("POST", "/register")]) == 1
    assert stored()["client_id"] == rec["client_id"]
    assert stored()["access_token"] != rec["access_token"]


def test_the_listener_ignores_anything_but_its_own_state(cli_env):
    """A stray local request (another page probing localhost ports) must neither
    complete the sign-in nor be accepted as the code."""
    _, browser, _ = cli_env
    browser["stray"] = "code=planted&state=not-the-state"
    code, out = run_cli("login", "--server", PUBLIC)
    assert code == 0, out
    assert stored()["access_token"].startswith("tbat_")


def test_an_expired_token_is_refreshed_and_the_old_refresh_token_is_spent(cli_env):
    c, _, _ = cli_env
    assert run_cli("login", "--server", PUBLIC)[0] == 0
    first = stored()
    assert credentials.access_token(PUBLIC) == first["access_token"]      # fresh: no refresh

    data = json.loads(credentials.credentials_path().read_text())
    data["servers"][PUBLIC]["expires_at"] = int(time.time()) + 10          # inside the margin
    credentials.credentials_path().write_text(json.dumps(data))
    token = credentials.access_token(PUBLIC)
    assert token != first["access_token"]
    second = stored()
    assert second["access_token"] == token and second["refresh_token"] != first["refresh_token"]
    assert second["expires_at"] > time.time() + 3000
    assert stat.S_IMODE(credentials.credentials_path().stat().st_mode) == 0o600
    assert c.get("/api/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    # Presenting the spent refresh token again ends the sign-in server-side,
    # and the CLI says to sign in again instead of looping.
    status, reply = credentials._request("POST", f"{PUBLIC}/token", form={
        "grant_type": "refresh_token", "refresh_token": first["refresh_token"],
        "client_id": first["client_id"]})
    assert status == 400 and reply["error"] == "invalid_grant"
    data["servers"][PUBLIC] = {**second, "expires_at": 1}
    credentials.credentials_path().write_text(json.dumps(data))
    with pytest.raises(credentials.NotSignedIn, match="tracebi login"):
        credentials.access_token(PUBLIC)
    assert "access_token" not in stored() and stored()["client_id"] == first["client_id"]


def test_access_token_without_a_login_says_how_to_log_in(cli_env):
    with pytest.raises(credentials.NotSignedIn, match="tracebi login --server"):
        credentials.access_token(PUBLIC)


def test_logout_revokes_at_the_server_and_forgets_the_tokens(cli_env):
    c, _, _ = cli_env
    assert run_cli("login", "--server", PUBLIC)[0] == 0
    rec = stored()
    auth = {"Authorization": f"Bearer {rec['access_token']}"}
    assert c.get("/api/me", headers=auth).status_code == 200

    code, out = run_cli("logout", "--server", PUBLIC)
    assert code == 0 and "Signed out" in out
    assert c.get("/api/me", headers=auth).status_code == 401
    refreshed = c.post("/token", data={"grant_type": "refresh_token",
                                       "client_id": rec["client_id"],
                                       "refresh_token": rec["refresh_token"]})
    assert refreshed.status_code == 400
    assert stored() == {"client_id": rec["client_id"]}
    with pytest.raises(credentials.NotSignedIn):
        credentials.access_token(PUBLIC)
    assert "Not signed in" in run_cli("logout", "--server", PUBLIC)[1]


def test_login_refuses_an_address_that_would_send_tokens_in_clear(cli_env):
    code, out = run_cli("login", "--server", "http://bi.example.com")
    assert code == 1 and "plain http" in out
    assert not credentials.credentials_path().exists()
    assert credentials.server_key("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    for bad in ("bi.example.com", "https://bi.example.com/app", "ftp://x", "https://u:p@x.example"):
        with pytest.raises(credentials.CredentialsError):
            credentials.server_key(bad)


def test_a_browser_that_never_answers_times_out_and_forgets_the_client(cli_env, monkeypatch):
    _, browser, _ = cli_env
    assert run_cli("login", "--server", PUBLIC)[0] == 0
    cached = stored()["client_id"]
    monkeypatch.setattr(credentials.webbrowser, "open", lambda url: True)
    code, out = run_cli("login", "--server", PUBLIC, "--timeout", "0.3")
    assert code == 1 and "timed out" in out
    assert "client_id" not in stored() and stored()["access_token"]      # tokens untouched
    assert cached


def test_a_server_whose_metadata_names_another_address_is_refused(cli_env, monkeypatch):
    real = credentials._request

    def lying(method, url, **kw):
        status, body = real(method, url, **kw)
        if url.endswith("/.well-known/oauth-authorization-server"):
            body = {**body, "token_endpoint": "https://evil.example/token"}
        return status, body

    monkeypatch.setattr(credentials, "_request", lying)
    code, out = run_cli("login", "--server", PUBLIC)
    assert code == 1 and "another host" in out
    assert not credentials.credentials_path().exists() or "access_token" not in json.dumps(
        json.loads(credentials.credentials_path().read_text()))


def test_a_corrupt_credentials_file_is_reported_not_overwritten(cli_env):
    path = credentials.credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    code, out = run_cli("login", "--server", PUBLIC)
    assert code == 1 and "cannot be read" in out
    assert path.read_text() == "{not json"
    assert os.path.exists(path)
