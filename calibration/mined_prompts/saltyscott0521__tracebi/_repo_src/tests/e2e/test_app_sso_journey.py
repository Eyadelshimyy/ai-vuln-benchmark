"""The web app's own sign-in, end to end through the real app.

The same fake identity provider as the MCP journey stands in for Entra or
Okta. A browser is a TestClient with a cookie jar on the public https origin;
a CLI or connector is a bearer token minted by the MCP leg. Every refusal
below guards one check in ``tracebi/web/api/sso.py`` (or the authorizer it
feeds); each test names the bug it would catch.
"""

from __future__ import annotations

import hashlib
import secrets
import urllib.parse

import pytest
from sqlalchemy import text

pytest.importorskip("mcp")
pytest.importorskip("jwt")

from tests.e2e.test_mcp_oauth_journey import (  # noqa: E402, F401
    CLAUDE, CLIENT_ID, IDP, PUBLIC, _challenge, authorize, cookie_header, idp,
    provider_login, register, returned_code, exchange, sign_in, signin,
)
from tracebi import mcp_oauth  # noqa: E402

ANALYST_ROUTE = "/api/reports/sample_model/sample_dashboard/run"     # needs analyst
ADMIN_ROUTE = "/api/pipelines/nope/run"                              # needs admin


@pytest.fixture
def sso(signin):
    """The scaffolded project with app sign-in on; yields a browser factory."""
    from fastapi.testclient import TestClient

    from tracebi.web.api import main

    main.serve_sso(mcp_oauth.GatewayOAuth(mcp_oauth.OAuthConfig.from_env()))

    def browser():
        return TestClient(main.app, base_url=PUBLIC)
    browser.mcp_client = signin          # the plain-http client the MCP journey uses
    try:
        yield browser
    finally:
        main.unserve_sso()


def begin(b, next=None):
    """GET /login; returns (response, provider query)."""
    resp = b.get("/login", params={"next": next} if next is not None else {},
                 follow_redirects=False)
    assert resp.status_code == 302, resp.text
    target = urllib.parse.urlsplit(resp.headers["location"])
    assert f"{target.scheme}://{target.netloc}{target.path}" == f"{IDP}/authorize"
    q = dict(urllib.parse.parse_qsl(target.query))
    assert q["redirect_uri"] == f"{PUBLIC}/login/callback"
    assert q["code_challenge_method"] == "S256" and q["response_type"] == "code"
    return resp, q


def answer(b, idp, q, *, cookies=None, **claims):
    """*cookies* are sent as the whole Cookie header, as a hand-built request."""
    """The provider authenticates the person; the browser returns to the app."""
    code = secrets.token_urlsafe(12)
    idp.pending[code] = {"challenge": q["code_challenge"],
                         "id_token": idp.id_token(nonce=q["nonce"], **claims)}
    headers = {"Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items())} if cookies else {}
    return b.get("/login/callback", params={"code": code, "state": q["state"]},
                 headers=headers, follow_redirects=False)


def sign_in_browser(b, idp, next=None, **claims):
    _, q = begin(b, next)
    resp = answer(b, idp, q, **claims)
    assert resp.status_code == 302, resp.text
    return resp


def sessions():
    from tracebi import state
    with state.ensure().connect() as conn:
        return [dict(r) for r in conn.execute(text("SELECT * FROM tracebi_sessions")).mappings()]


# ── the browser journey ──────────────────────────────────────────────────

def test_a_person_signs_in_in_the_browser_and_is_themselves(sso, idp):
    """/login -> provider -> /login/callback -> a session whose cookie carries
    the right flags, and only its hash is stored."""
    b = sso()
    assert b.get("/api/me").status_code == 401

    resp = sign_in_browser(b, idp, next="/reports/sales?tab=1", groups=["bi-analysts"])
    assert resp.headers["location"] == "/reports/sales?tab=1"
    cookie = next(v for k, v in resp.headers.multi_items()
                  if k == "set-cookie" and v.startswith("tracebi_session="))
    flags = cookie.lower()
    assert "httponly" in flags and "secure" in flags
    assert "samesite=lax" in flags and "path=/" in flags
    raw = b.cookies.get("tracebi_session")
    assert raw and len(raw) >= 40
    # The binding cookie has done its work and is dropped.
    assert any(v.startswith("tb_login_") and "max-age=0" in v.lower()
               for k, v in resp.headers.multi_items() if k == "set-cookie")

    [row] = sessions()
    assert row["session_hash"] == hashlib.sha256(raw.encode()).hexdigest()
    assert raw not in "".join(str(v) for v in row.values())
    assert (row["actor"], row["role"]) == ("alice@example.com", "analyst")
    assert row["absolute_expires_at"] - row["created_at"] == 7 * 86400
    assert row["expires_at"] - row["created_at"] == 12 * 3600

    me = b.get("/api/me").json()
    assert me == {"actor": "alice@example.com", "role": "analyst", "sign_in": "oidc"}
    assert b.get("/api/health").status_code == 200


def test_the_role_decides_what_a_person_may_do(sso, idp):
    """Role enforcement is on in OIDC mode: a viewer reads, an analyst runs, only
    an admin touches pipelines. Catches a middleware that authenticates but
    skips the authorizer (everyone admin)."""
    viewer, analyst, admin = sso(), sso(), sso()
    sign_in_browser(viewer, idp, sub="u-v", email="v@example.com")           # default role
    sign_in_browser(analyst, idp, sub="u-a", email="a@example.com", groups=["bi-analysts"])
    sign_in_browser(admin, idp, sub="u-ad", email="ad@example.com", groups=["bi-admins"])

    assert viewer.get("/api/me").json()["role"] == "viewer"
    assert viewer.get("/api/models").status_code == 200
    denied = viewer.post(ANALYST_ROUTE)
    assert denied.status_code == 403
    assert denied.json()["detail"]["required_role"] == "analyst"
    assert analyst.post(ANALYST_ROUTE).status_code != 403
    assert analyst.post(ADMIN_ROUTE).status_code == 403
    assert admin.post(ADMIN_ROUTE).status_code != 403


def test_what_a_person_does_is_recorded_under_their_name(sso, idp):
    """The actor ContextVar is set around the request, as in the other modes."""
    b = sso()
    sign_in_browser(b, idp, groups=["bi-analysts"])
    resp = b.post(ANALYST_ROUTE)
    assert resp.status_code == 200, resp.text
    runs = b.get("/api/runs").json()
    assert runs and {(r["actor"], r["actor_role"]) for r in runs
                     if r.get("actor")} == {("alice@example.com", "analyst")}


def test_logout_ends_the_session_for_good(sso, idp):
    """A copy of the cookie taken before logout must stop working: the server
    forgets the session, not only the browser."""
    b = sso()
    sign_in_browser(b, idp)
    stolen = b.cookies.get("tracebi_session")
    assert b.post("/logout").json() == {"ok": True}
    assert b.get("/api/me").status_code == 401
    other = sso()
    other.cookies.set("tracebi_session", stolen, domain="bi.example.com")
    assert other.get("/api/me").status_code == 401
    assert sessions()[0]["revoked_at"] is not None


def test_logout_refuses_a_cross_site_post(sso, idp):
    b = sso()
    sign_in_browser(b, idp)
    assert b.post("/logout", headers={"Origin": "https://evil.example"}).status_code == 403
    assert b.get("/api/me").status_code == 200


def test_a_cookie_cannot_ride_a_cross_site_write(sso, idp):
    """Cookie writes from another site are refused even without an Origin the
    CSRF guard would see (Sec-Fetch-Site says cross-site)."""
    b = sso()
    sign_in_browser(b, idp, groups=["bi-analysts"])
    assert b.post(ANALYST_ROUTE, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert b.post(ANALYST_ROUTE, headers={"Sec-Fetch-Site": "same-origin"}).status_code != 403


# ── sessions that must not work ──────────────────────────────────────────

def test_forged_unknown_expired_and_revoked_sessions_are_refused(sso, idp):
    b = sso()
    sign_in_browser(b, idp)
    good = b.cookies.get("tracebi_session")

    def status(cookie):
        """/api/me as a browser holding exactly *cookie*."""
        fresh = sso()
        fresh.cookies.set("tracebi_session", cookie, domain="bi.example.com")
        return fresh.get("/api/me").status_code

    assert status(good) == 200
    for forged in ("x", secrets.token_urlsafe(32), good[:-1] + ("A" if good[-1] != "A" else "B"),
                   hashlib.sha256(good.encode()).hexdigest()):
        assert status(forged) == 401, forged

    from tracebi import state
    eng = state.ensure()

    def set_column(column, value):
        with eng.begin() as conn:
            conn.execute(text(f"UPDATE tracebi_sessions SET {column} = :v"), {"v": value})

    set_column("expires_at", 1)                      # idle too long
    assert status(good) == 401
    set_column("expires_at", 9999999999)
    assert status(good) == 200
    set_column("absolute_expires_at", 1)             # a week is a week
    assert status(good) == 401
    set_column("absolute_expires_at", 9999999999)
    set_column("revoked_at", 1)
    assert status(good) == 401


def test_a_used_session_stays_open_but_never_past_its_week(sso, idp):
    """The idle deadline slides with use, capped at the absolute one."""
    b = sso()
    sign_in_browser(b, idp)
    from tracebi import state
    near = int(__import__("time").time()) + 600
    with state.ensure().begin() as conn:
        conn.execute(text("UPDATE tracebi_sessions SET expires_at = :e, absolute_expires_at = :e"),
                     {"e": near})
    assert b.get("/api/me").status_code == 200
    assert sessions()[0]["expires_at"] == near


# ── the way back from the provider ───────────────────────────────────────

@pytest.mark.parametrize("hostile", [
    "//evil.example/x", "/\\evil.example", "https://evil.example", "javascript:alert(1)",
    "evil.example", "/\t/evil.example", "/ok\r\nSet-Cookie: x=1", "/login", "/logout", "",
])
def test_next_only_ever_leads_to_a_page_of_this_app(sso, idp, hostile):
    """An open redirect after sign-in is a phishing gift."""
    b = sso()
    resp = sign_in_browser(b, idp, next=hostile)
    assert resp.headers["location"] == "/"


def test_next_keeps_a_local_path_and_its_query(sso, idp):
    b = sso()
    resp = sign_in_browser(b, idp, next="/models/sales_model/explore?measure=revenue")
    assert resp.headers["location"] == "/models/sales_model/explore?measure=revenue"


def test_a_callback_from_another_browser_is_refused(sso, idp):
    """The sign-in link someone else began, opened by a victim: no matching
    cookie, no session."""
    attacker, victim = sso(), sso()
    _, q = begin(attacker)
    resp = answer(victim, idp, q)
    assert resp.status_code == 400 and "different browser" in resp.text
    assert "tracebi_session" not in victim.cookies and sessions() == []


def test_the_binding_cookie_must_belong_to_this_sign_in(sso, idp):
    b = sso()
    _, q = begin(b)
    flow = q["state"].partition(".")[0]
    resp = answer(sso(), idp, q, cookies={f"tb_login_{flow}": secrets.token_urlsafe(32)})
    assert resp.status_code == 400 and sessions() == []


def test_a_state_can_be_used_once(sso, idp):
    b = sso()
    _, q = begin(b)
    assert answer(b, idp, q).status_code == 302
    again = answer(b, idp, q)
    assert again.status_code == 400
    assert len(sessions()) == 1


def test_an_unknown_or_tampered_state_is_refused(sso, idp):
    b = sso()
    _, q = begin(b)
    flow = q["state"].partition(".")[0]
    for state in (q["state"] + "x", f"{flow}.{secrets.token_urlsafe(32)}", "nonsense", "", "."):
        resp = b.get("/login/callback", params={"code": "c", "state": state},
                     follow_redirects=False)
        assert resp.status_code == 400
    assert sessions() == []


def test_a_gateway_sign_in_state_cannot_open_an_app_session(sso, idp):
    """The MCP flow shares the pending table. Its state, replayed here with a
    binding cookie renamed to match, must not become a browser session."""
    c = sso.mcp_client
    client_id = register(c).json()["client_id"]
    auth, _ = authorize(c, client_id)
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(auth.headers["location"]).query))
    flow = q["state"].partition(".")[0]
    binder = auth.cookies[f"tb_oauth_{flow}"]
    b = sso()
    resp = answer(b, idp, q, cookies={f"tb_login_{flow}": binder})
    assert resp.status_code == 400 and sessions() == []


def test_an_id_token_the_provider_did_not_sign_is_refused(sso, idp):
    b = sso()
    _, q = begin(b)
    code = secrets.token_urlsafe(12)
    idp.pending[code] = {"challenge": q["code_challenge"], "id_token": idp.id_token(
        nonce=q["nonce"], signer=idp.other_key)}
    resp = b.get("/login/callback", params={"code": code, "state": q["state"]},
                 follow_redirects=False)
    assert resp.status_code == 403 and sessions() == []


def test_an_id_token_for_another_sign_in_is_refused(sso, idp):
    """Nonce binds the token to this attempt (a captured token cannot be replayed)."""
    b = sso()
    _, q = begin(b)
    resp = answer(b, idp, {**q, "nonce": "not-the-one-we-sent"})
    assert resp.status_code == 403 and sessions() == []


def test_a_denied_sign_in_makes_no_session(sso, idp):
    b = sso()
    _, q = begin(b)
    resp = b.get("/login/callback", params={"error": "access_denied", "state": q["state"]},
                 follow_redirects=False)
    assert resp.status_code == 403 and sessions() == []


def test_a_strangers_sign_in_attempts_write_nothing(sso):
    """/login is open to strangers: what the return trip needs travels in a
    signed cookie in their browser, so no number of attempts fills a table or
    locks anyone out."""
    import sqlalchemy as sa

    from tracebi import state

    b = sso()
    for _ in range(25):
        assert b.get("/login", follow_redirects=False).status_code == 302
    with state.ensure().connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM tracebi_oauth_pending")).scalar() == 0
    assert sessions() == []


def test_a_tampered_or_expired_sign_in_cookie_is_refused(sso, idp, monkeypatch):
    from tracebi.web.api import sso as sso_module

    b = sso()
    resp, q = begin(b)
    name = next(c for c in b.cookies.keys() if c.startswith("tb_login_"))
    body, _, mac = b.cookies.get(name).partition(".")
    b.cookies.clear()     # send only the forged copy, whatever the jar's domain rules
    forged = answer(b, idp, q, cookies={name: body + "." + ("0" * len(mac))})
    assert forged.status_code == 400 and "not valid" in forged.text and sessions() == []

    b = sso()
    monkeypatch.setattr(sso_module, "_LOGIN_TTL", -1)
    resp, q = begin(b)
    assert answer(b, idp, q).status_code == 400 and sessions() == []


# ── who gets in without a session ────────────────────────────────────────

def test_a_stranger_gets_401_from_the_api_and_a_redirect_from_a_page(sso):
    b = sso()
    assert b.get("/api/health").status_code == 200
    for path in ("/api/models", "/api/reports", "/api/runs", "/api/me", "/api/discovery",
                 "/api/schema"):
        resp = b.get(path)
        assert resp.status_code == 401, path
        assert resp.json()["detail"]["login"] == "/login"
        assert "www-authenticate" in resp.headers
    assert b.post(ANALYST_ROUTE).status_code == 401

    html = {"Accept": "text/html,application/xhtml+xml"}
    nav = b.get("/", headers=html, follow_redirects=False)
    assert nav.status_code == 302 and nav.headers["location"] == "/login?next=%2F"
    nav = b.get("/reports/sales?x=1", headers=html, follow_redirects=False)
    assert nav.headers["location"] == "/login?next=%2Freports%2Fsales%3Fx%3D1"
    share = b.get("/r/sample_model/sample_dashboard", headers=html, follow_redirects=False)
    assert share.status_code == 302 and share.headers["location"].startswith("/login?next=")
    # The bundle's own files and the sign-in routes answer without a session.
    assert b.get("/assets/app.js", headers=html, follow_redirects=False).status_code != 302
    assert b.get("/login", follow_redirects=False).status_code == 302
    # A fetch for JSON is not a navigation.
    assert b.get("/", follow_redirects=False).status_code != 302


def test_a_basic_header_is_not_a_login(sso):
    import base64
    b = sso()
    basic = base64.b64encode(b"alice:secret").decode()
    assert b.get("/api/models", headers={"Authorization": f"Basic {basic}"}).status_code == 401


def test_a_stale_cookie_is_cleared_on_the_401(sso):
    b = sso()
    resp = b.get("/api/me", headers={"Cookie": "tracebi_session=stale"})
    assert resp.status_code == 401
    assert "tracebi_session=" in resp.headers.get("set-cookie", "")


# ── access tokens on the API (what `tracebi login` sends) ────────────────

def test_a_tracebi_access_token_acts_as_its_person_on_the_api(sso, idp):
    c = sso.mcp_client
    client_id, tokens = sign_in(c, idp, groups=["bi-analysts"])
    auth = {"Authorization": f"Bearer {tokens['access_token']}"}
    b = sso()
    assert b.get("/api/me", headers=auth).json() == {
        "actor": "alice@example.com", "role": "analyst", "sign_in": "oidc"}
    assert b.post(ANALYST_ROUTE, headers=auth).status_code != 403
    assert b.post(ADMIN_ROUTE, headers=auth).status_code == 403

    _, viewer = sign_in(c, idp, sub="u-v", email="v@example.com")
    assert b.post(ANALYST_ROUTE, headers={
        "Authorization": f"Bearer {viewer['access_token']}"}).status_code == 403


def test_a_bad_or_revoked_token_is_refused_and_does_not_fall_back_to_the_cookie(sso, idp):
    c = sso.mcp_client
    client_id, tokens = sign_in(c, idp, groups=["bi-analysts"])
    auth = {"Authorization": f"Bearer {tokens['access_token']}"}
    b = sso()
    assert b.get("/api/me", headers=auth).status_code == 200

    resp = c.post("/revoke", data={"client_id": client_id, "token": tokens["refresh_token"]})
    assert resp.status_code == 200
    assert b.get("/api/me", headers=auth).status_code == 401

    for bad in ("tbat_" + secrets.token_urlsafe(32), "tbrt_x", tokens["refresh_token"], "static"):
        assert b.get("/api/me", headers={"Authorization": f"Bearer {bad}"}).status_code == 401, bad
    # A valid session cookie does not rescue a bad token sent beside it.
    sign_in_browser(b, idp)
    assert b.get("/api/me").status_code == 200
    assert b.get("/api/me", headers={"Authorization": "Bearer tbat_nope"}).status_code == 401


# ── startup and posture ──────────────────────────────────────────────────

def test_oidc_beside_basic_or_proxy_auth_refuses_to_start(monkeypatch):
    from fastapi import FastAPI

    from tracebi.web.api.auth import install_if_configured

    monkeypatch.setenv("TRACEBI_OIDC_ISSUER", IDP)
    monkeypatch.delenv("TRACEBI_AUTH_USER", raising=False)
    monkeypatch.delenv("TRACEBI_AUTH_PASS", raising=False)
    monkeypatch.delenv("TRACEBI_AUTH_PROXY_HEADER", raising=False)
    assert install_if_configured(FastAPI()) == "oidc"

    monkeypatch.setenv("TRACEBI_AUTH_USER", "admin")
    monkeypatch.setenv("TRACEBI_AUTH_PASS", "pw")
    with pytest.raises(ValueError, match="cannot run beside"):
        install_if_configured(FastAPI())
    monkeypatch.delenv("TRACEBI_AUTH_USER")
    monkeypatch.delenv("TRACEBI_AUTH_PASS")
    monkeypatch.setenv("TRACEBI_AUTH_PROXY_HEADER", "X-Forwarded-User")
    with pytest.raises(ValueError, match="cannot run beside"):
        install_if_configured(FastAPI())


def test_without_sso_the_app_is_as_it_was(monkeypatch):
    """/api/me exists in every mode; with nothing configured it says so, and
    the sign-in routes are not there."""
    from fastapi.testclient import TestClient

    from tracebi.web.api import main

    for name in ("TRACEBI_AUTH_USER", "TRACEBI_AUTH_PASS", "TRACEBI_AUTH_PROXY_HEADER"):
        monkeypatch.delenv(name, raising=False)
    c = TestClient(main.app)
    me = c.get("/api/me")
    if main._auth_mode is None:
        assert me.json() == {"actor": None, "role": "admin", "sign_in": "none"}
    assert c.post("/logout").status_code in (404, 405)
    assert c.get("/login", follow_redirects=False).status_code != 302
