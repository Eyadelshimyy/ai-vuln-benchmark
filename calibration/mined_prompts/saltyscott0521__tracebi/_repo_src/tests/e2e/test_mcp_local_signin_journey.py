"""The MCP gateway's built-in sign-in (no identity provider), end to end.

``TRACEBI_DEMO_LOGIN`` gives a one-click demo user and ``TRACEBI_LOCAL_USERS``
named users with passwords. The person signs in on a page TraceBi serves at
``/oauth/callback``; everything after (code, /token, /mcp, the role) is the
same authorization server the identity-provider journey drives, so this file
checks only what is different: the page, the passwords, the throttle, the
browser binding and startup refusing a half-set configuration.
"""

from __future__ import annotations

import re
import urllib.parse

import pytest

pytest.importorskip("mcp")
pytest.importorskip("jwt")

from tests.e2e.test_mcp_oauth_journey import (  # noqa: E402
    MCP_URL,
    PUBLIC,
    authorize,
    cookie_header,
    exchange,
    mcp_session,
    register,
    returned_code,
    tool,
)
from tracebi import mcp_oauth  # noqa: E402

SECRET = "x" * 40
USERS = "alice:analyst:alice-password,vera:viewer:vera-password"


@pytest.fixture
def local(scaffolded, monkeypatch):
    """The scaffolded project served with the built-in sign-in turned on."""
    from fastapi.testclient import TestClient

    from tracebi.registry import registry
    from tracebi.web import discovery
    from tracebi.web.api import main

    for attr in ("_connectors", "_models", "_report_factories",
                 "_scheduled_factories", "_pipelines"):
        monkeypatch.setattr(registry, attr, {})
    monkeypatch.setattr(registry, "_default_model_name", None)
    for attr, empty in (("_discovered", {}), ("_live_reports", {}),
                        ("_failed_sources", {}), ("_live_models", set()),
                        ("_live_pipelines", set()), ("_outcomes", [])):
        monkeypatch.setattr(discovery, attr, empty)
    for name, value in {
            "TRACEBI_PUBLIC_URL": PUBLIC, "TRACEBI_OAUTH_SECRET": SECRET,
            "TRACEBI_LOCAL_USERS": USERS, "TRACEBI_DEMO_LOGIN": "analyst"}.items():
        monkeypatch.setenv(name, value)
    for name in ("TRACEBI_OIDC_ISSUER", "TRACEBI_PUBLIC_MCP_URL",
                 "TRACEBI_OAUTH_REDIRECTS", "TRACEBI_MCP_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    from tests.e2e.conftest import run_cli
    code, out = run_cli("run-transform", "sample_transform")
    assert code == 0, out
    discovery.rescan("reports", "models", "pipelines")

    main.serve_gateway(oauth=mcp_oauth.GatewayOAuth(mcp_oauth.OAuthConfig.from_env()))
    client = TestClient(main.app)
    try:
        with client:
            yield client
    finally:
        main.unserve_gateway()


def begin(c):
    """A connector sends the person to /authorize; the browser lands on the
    sign-in page. Returns (client_id, verifier, state, the cookie header)."""
    client_id = register(c).json()["client_id"]
    auth, verifier = authorize(c, client_id)
    assert auth.status_code == 302, auth.text
    target = urllib.parse.urlsplit(auth.headers["location"])
    assert f"{target.scheme}://{target.netloc}{target.path}" == f"{PUBLIC}/oauth/callback"
    state = dict(urllib.parse.parse_qsl(target.query))["state"]
    return client_id, verifier, state, {"Cookie": cookie_header(auth)}


def submit(c, state, cookie, **fields):
    return c.post("/oauth/callback", data={"state": state, **fields},
                  headers=cookie, follow_redirects=False)


def tokens_for(c, **fields):
    client_id, verifier, state, cookie = begin(c)
    done = submit(c, state, cookie, **fields)
    resp = exchange(c, client_id, returned_code(done), verifier)
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_the_demo_user_connects_in_one_click_and_can_draft(local, scaffolded):
    c = local
    client_id, verifier, state, cookie = begin(c)

    page = c.get("/oauth/callback", params={"state": state}, headers=cookie)
    assert page.status_code == 200
    assert "Continue as the demo user" in page.text
    assert "The demo user is an analyst." in page.text
    assert "Password" in page.text            # the named users' form is there too
    assert page.headers["cache-control"] == "no-store"
    assert page.headers["x-frame-options"] == "DENY"

    done = submit(c, state, cookie, demo="1")
    resp = exchange(c, client_id, returned_code(done), verifier)
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]

    session = mcp_session(c, token)
    path = "sample_model/sample_dashboard"
    started = tool(c, token, session, "start_draft", kind="reports", path=path,
                   from_published=True)["structuredContent"]
    assert started["ok"], started
    assert started["owner"] == "demo"
    assert (scaffolded / "drafts" / "demo" / "reports" / path).is_dir()


def test_a_named_user_signs_in_with_their_password_and_gets_their_role(local):
    c = local
    token = tokens_for(c, username="vera", password="vera-password")
    session = mcp_session(c, token)
    assert tool(c, token, session, "list_models")["structuredContent"]
    refused = tool(c, token, session, "start_draft", kind="reports", path="x/y")
    assert refused["isError"] is True and "viewer" in refused["content"][0]["text"]

    token = tokens_for(c, username="alice", password="alice-password")
    session = mcp_session(c, token)
    assert tool(c, token, session, "start_draft", kind="reports",
                path="sample_model/sample_dashboard", from_published=True)["structuredContent"]["ok"]


def test_a_wrong_password_signs_nobody_in_and_too_many_are_throttled(local):
    c = local
    _, _, state, cookie = begin(c)
    for bad in ({"username": "alice", "password": "nope-nope"},
                {"username": "nobody", "password": "alice-password"},
                {"username": "alice", "password": ""}):
        resp = submit(c, state, cookie, **bad)
        assert resp.status_code == 401 and "do not match" in resp.text
        assert "location" not in resp.headers and "alice-password" not in resp.text

    # Ten wrong passwords for one name stop that name, even for the right one.
    for _ in range(10):
        submit(c, state, cookie, username="alice", password="wrong-wrong")
    blocked = submit(c, state, cookie, username="alice", password="alice-password")
    assert blocked.status_code == 429 and "Too many attempts" in blocked.text
    assert "location" not in blocked.headers
    # Someone else is not locked out by it.
    assert submit(c, state, cookie, username="vera", password="vera-password").status_code == 302


def test_the_demo_button_is_absent_when_no_demo_user_is_configured(local, monkeypatch):
    c = local
    from tracebi.web.api import main
    main.unserve_gateway()
    cfg = mcp_oauth.OAuthConfig.from_env({**_env(), "TRACEBI_DEMO_LOGIN": ""})
    main.serve_gateway(oauth=mcp_oauth.GatewayOAuth(cfg))
    _, _, state, cookie = begin(c)
    page = c.get("/oauth/callback", params={"state": state}, headers=cookie)
    assert "demo user" not in page.text and "Password" in page.text
    # A forged demo field does nothing when there is no demo user.
    assert submit(c, state, cookie, demo="1").status_code == 401


def test_a_sign_in_must_finish_in_the_browser_that_began_it(local):
    """The sign-in link can be forwarded. Without the cookie of the browser that
    began it, neither the page nor the answer is served."""
    c = local
    _, _, state, cookie = begin(c)
    stranger = c.get("/oauth/callback", params={"state": state})
    assert stranger.status_code == 400 and "different browser" in stranger.text
    posted = c.post("/oauth/callback", data={"state": state, "demo": "1"},
                    follow_redirects=False)
    assert posted.status_code == 400 and "location" not in posted.headers

    altered = {"Cookie": cookie["Cookie"][:-3] + "abc"}
    assert submit(c, state, altered, demo="1").status_code == 400
    assert c.get("/oauth/callback", params={"state": "made-up"}).status_code == 400


def test_the_page_escapes_what_it_is_given(local):
    c = local
    _, _, state, cookie = begin(c)
    resp = submit(c, state, cookie, username="<script>x</script>", password="whatever1")
    assert resp.status_code == 401 and "<script>x" not in resp.text
    assert re.search(r'name="state" value="[^"<>]+"', resp.text)


def _env(**over):
    return {"TRACEBI_PUBLIC_URL": PUBLIC, "TRACEBI_OAUTH_SECRET": SECRET,
            "TRACEBI_LOCAL_USERS": USERS, "TRACEBI_DEMO_LOGIN": "analyst", **over}


def test_a_half_set_configuration_stops_startup():
    assert mcp_oauth.OAuthConfig.from_env({}) is None
    good = mcp_oauth.OAuthConfig.from_env(_env())
    assert (good.mcp_url, good.issuer, good.local, good.demo_role) == (
        MCP_URL, PUBLIC, True, "analyst")
    assert good.local_users["alice"] == ("analyst", "alice-password")
    assert mcp_oauth.OAuthConfig.from_env(_env(TRACEBI_LOCAL_USERS="")).local_users == {}

    for broken, mention in [
            (_env(TRACEBI_OAUTH_SECRET=""), "TRACEBI_OAUTH_SECRET"),
            (_env(TRACEBI_OAUTH_SECRET="short"), "TRACEBI_OAUTH_SECRET"),
            (_env(TRACEBI_PUBLIC_URL=""), "TRACEBI_PUBLIC_MCP_URL"),
            (_env(TRACEBI_PUBLIC_URL="http://bi.example.com"), "https"),
            (_env(TRACEBI_DEMO_LOGIN="root"), "TRACEBI_DEMO_LOGIN"),
            (_env(TRACEBI_LOCAL_USERS="alice:superuser:long-enough"), "TRACEBI_LOCAL_USERS"),
            (_env(TRACEBI_LOCAL_USERS="alice:analyst:short"), "TRACEBI_LOCAL_USERS"),
            (_env(TRACEBI_LOCAL_USERS="alice:analyst:long-enough,alice:viewer:long-enough"), "twice"),
            (_env(TRACEBI_OIDC_ISSUER="https://idp.example.com"), "choose one way")]:
        with pytest.raises(mcp_oauth.OAuthConfigError, match=mention):
            mcp_oauth.OAuthConfig.from_env(broken)
