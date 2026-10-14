"""The MCP gateway's per-person sign-in, end to end through the real app.

A fake company identity provider (an RSA key, discovery, JWKS, a token
endpoint) stands in for Entra or Okta by replacing the two outbound fetches.
Everything else is real: the app's routes, the SDK's OAuth handlers, the run
store, the /mcp transport and the tools. A connector's whole journey is driven
by hand: 401, metadata, registration, /authorize, the provider, /oauth/callback,
/token, then tool calls as the person.

Each refusal below guards one check in ``tracebi/mcp_oauth.py`` (or the SDK
handler it relies on); the module docstring of each test names the bug it
would catch.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import urllib.parse

import pytest

pytest.importorskip("mcp")
pytest.importorskip("jwt")

import jwt  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

from tracebi import mcp_oauth  # noqa: E402

PUBLIC = "https://bi.example.com"
MCP_URL = f"{PUBLIC}/mcp"
IDP = "https://idp.example.com"
CLIENT_ID = "tracebi-app"
CLAUDE = mcp_oauth.CLAUDE_REDIRECT


class FakeIdP:
    """An OIDC provider that answers the two fetches TraceBi makes."""

    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.pending: dict[str, dict] = {}     # code -> {"id_token", "challenge"}
        self.token_posts: list[dict] = []

    def jwk(self):
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        return {**jwk, "kid": "k1", "use": "sig", "alg": "RS256"}

    def id_token(self, *, nonce, sub="u-alice", email="alice@example.com",
                 groups=(), aud=CLIENT_ID, iss=IDP, ttl=300, signer=None, **extra):
        now = int(time.time())
        claims = {"iss": iss, "aud": aud, "sub": sub, "iat": now, "exp": now + ttl,
                  "nonce": nonce, "email": email, "groups": list(groups), **extra}
        return jwt.encode(claims, signer or self.key, algorithm="RS256",
                          headers={"kid": "k1"})

    def fetch(self, url, form=None):
        if url == f"{IDP}/.well-known/openid-configuration":
            return {"issuer": IDP, "authorization_endpoint": f"{IDP}/authorize",
                    "token_endpoint": f"{IDP}/token", "jwks_uri": f"{IDP}/jwks"}
        if url == f"{IDP}/jwks":
            return {"keys": [self.jwk()]}
        if url == f"{IDP}/token":
            self.token_posts.append(form)
            grant = self.pending.pop(form["code"], None)
            assert grant, "unknown provider code"
            assert form["client_id"] == CLIENT_ID and form["client_secret"] == "idp-secret"
            assert form["redirect_uri"] in (f"{PUBLIC}/oauth/callback",
                                            f"{PUBLIC}/login/callback")
            assert _challenge(form["code_verifier"]) == grant["challenge"], "provider PKCE"
            return {"id_token": grant["id_token"]}
        raise AssertionError(f"unexpected fetch {url}")


def _challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")


@pytest.fixture
def idp(monkeypatch):
    fake = FakeIdP()
    monkeypatch.setattr(mcp_oauth, "_fetch_json", fake.fetch)
    return fake


@pytest.fixture
def signin(scaffolded, monkeypatch, idp):
    """The scaffolded project served with per-person sign-in turned on."""
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
            "TRACEBI_OIDC_ISSUER": IDP, "TRACEBI_OIDC_CLIENT_ID": CLIENT_ID,
            "TRACEBI_OIDC_CLIENT_SECRET": "idp-secret",
            "TRACEBI_PUBLIC_URL": PUBLIC,
            "TRACEBI_OIDC_ROLE_MAP": "bi-admins:admin,bi-analysts:analyst"}.items():
        monkeypatch.setenv(name, value)
    for name in ("TRACEBI_PUBLIC_MCP_URL", "TRACEBI_OAUTH_REDIRECTS",
                 "TRACEBI_MCP_TOKEN", "TRACEBI_OIDC_DEFAULT_ROLE"):
        monkeypatch.delenv(name, raising=False)

    from tests.e2e.conftest import run_cli
    code, out = run_cli("run-transform", "sample_transform")
    assert code == 0, out
    discovery.rescan("reports", "models", "pipelines")

    cfg = mcp_oauth.OAuthConfig.from_env()
    main.serve_gateway(oauth=mcp_oauth.GatewayOAuth(cfg))
    client = TestClient(main.app)
    try:
        with client:
            yield client
    finally:
        main.unserve_gateway()


# ── driving a connector by hand ──────────────────────────────────────────

def register(c, redirect=CLAUDE, **extra):
    resp = c.post("/register", json={
        "client_name": "Claude", "redirect_uris": [redirect],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"], **extra})
    return resp


def authorize(c, client_id, redirect=CLAUDE, verifier=None, **extra):
    verifier = verifier or secrets.token_urlsafe(48)
    params = {"response_type": "code", "client_id": client_id, "redirect_uri": redirect,
              "code_challenge": _challenge(verifier), "code_challenge_method": "S256",
              "state": "client-state", "scope": "mcp offline_access",
              "resource": MCP_URL, **extra}
    return c.get("/authorize", params=params, follow_redirects=False), verifier


def cookie_header(resp):
    return "; ".join(f"{k}={v}" for k, v in resp.cookies.items())


def provider_login(c, idp, auth_resp, **claims):
    """The person logs in at the provider and the browser returns to TraceBi.
    Returns the callback response."""
    assert auth_resp.status_code == 302, auth_resp.text
    target = urllib.parse.urlsplit(auth_resp.headers["location"])
    assert f"{target.scheme}://{target.netloc}{target.path}" == f"{IDP}/authorize"
    q = dict(urllib.parse.parse_qsl(target.query))
    assert q["response_type"] == "code" and q["client_id"] == CLIENT_ID
    assert q["redirect_uri"] == f"{PUBLIC}/oauth/callback"
    assert q["code_challenge_method"] == "S256" and q["scope"].split()[0] == "openid"
    code = secrets.token_urlsafe(12)
    idp.pending[code] = {"challenge": q["code_challenge"],
                         "id_token": idp.id_token(nonce=q["nonce"], **claims)}
    return c.get("/oauth/callback", params={"code": code, "state": q["state"]},
                 headers={"Cookie": cookie_header(auth_resp)}, follow_redirects=False)


def returned_code(callback_resp):
    assert callback_resp.status_code == 302, callback_resp.text
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(callback_resp.headers["location"]).query))
    assert q.get("state") == "client-state"
    assert "code" in q, q
    return q["code"]


def exchange(c, client_id, code, verifier, redirect=CLAUDE):
    return c.post("/token", data={
        "grant_type": "authorization_code", "code": code, "client_id": client_id,
        "redirect_uri": redirect, "code_verifier": verifier})


def sign_in(c, idp, redirect=CLAUDE, client_id=None, **claims):
    client_id = client_id or register(c, redirect).json()["client_id"]
    auth, verifier = authorize(c, client_id, redirect)
    code = returned_code(provider_login(c, idp, auth, **claims))
    resp = exchange(c, client_id, code, verifier, redirect)
    assert resp.status_code == 200, resp.text
    return client_id, resp.json()


def _mcp_call(c, method, params=None, session=None, token=None, id=1):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if id is not None:
        body["id"] = id
    headers = {"Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session:
        headers["mcp-session-id"] = session
    resp = c.post("/mcp", json=body, headers=headers)
    data = [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]
    return resp, (data[-1].get("result") if data else None)


def mcp_session(c, token):
    resp, _ = _mcp_call(c, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "journey", "version": "0"}}, token=token)
    assert resp.status_code == 200, resp.text
    session = resp.headers["mcp-session-id"]
    _mcp_call(c, "notifications/initialized", session=session, token=token, id=None)
    return session


def tool(c, token, session, name, **arguments):
    resp, got = _mcp_call(c, "tools/call", {"name": name, "arguments": arguments},
                          session=session, token=token)
    assert resp.status_code == 200, resp.text
    return got


# ── discovery ────────────────────────────────────────────────────────────

def test_a_connector_discovers_the_sign_in_from_a_401(signin):
    c = signin
    resp = _mcp_call(c, "initialize")[0]
    assert resp.status_code == 401
    challenge = resp.headers["www-authenticate"]
    assert challenge.startswith("Bearer ")
    assert f'resource_metadata="{PUBLIC}/.well-known/oauth-protected-resource/mcp"' in challenge
    assert _mcp_call(c, "initialize", token="not-a-token")[0].status_code == 401

    for path in ("/.well-known/oauth-protected-resource/mcp",
                 "/.well-known/oauth-protected-resource"):
        prm = c.get(path).json()
        assert prm["resource"] == MCP_URL
        assert prm["authorization_servers"][0] == PUBLIC     # TraceBi's own issuer first
        assert "offline_access" in prm["scopes_supported"]

    meta = c.get("/.well-known/oauth-authorization-server").json()
    assert meta["issuer"] == PUBLIC
    assert meta["authorization_endpoint"] == f"{PUBLIC}/authorize"
    assert meta["token_endpoint"] == f"{PUBLIC}/token"
    assert meta["registration_endpoint"] == f"{PUBLIC}/register"
    assert meta["code_challenge_methods_supported"] == ["S256"]
    assert meta["client_id_metadata_document_supported"] is True
    assert "none" in meta["token_endpoint_auth_methods_supported"]
    assert "offline_access" in meta["scopes_supported"]
    assert set(meta["grant_types_supported"]) == {"authorization_code", "refresh_token"}


def test_the_oauth_endpoints_sit_outside_the_apps_own_auth(signin, monkeypatch):
    """The connector holds no Basic credential: metadata, register and token must
    answer without one while /api still asks. A prefix added to the protected
    list would lock every connector out."""
    from tracebi.web.api.auth import _PROTECTED_PREFIXES

    for path in ("/authorize", "/token", "/register", "/oauth/callback", "/mcp",
                 "/.well-known/oauth-authorization-server"):
        assert not path.startswith(_PROTECTED_PREFIXES), path


# ── the whole journey ────────────────────────────────────────────────────

def test_a_person_signs_in_and_their_work_is_theirs(signin, idp, scaffolded):
    c = signin
    client_id, tokens = sign_in(c, idp, groups=["bi-analysts"])
    assert tokens["token_type"].lower() == "bearer" and tokens["expires_in"] == 3600
    assert tokens["access_token"].startswith("tbat_") and tokens["refresh_token"]

    path = "sample_model/sample_dashboard"
    token = tokens["access_token"]
    session = mcp_session(c, token)
    started = tool(c, token, session, "start_draft", kind="reports", path=path,
                   from_published=True)["structuredContent"]
    assert started["ok"], started
    assert started["owner"] == "alice@example.com"
    assert (scaffolded / "drafts" / "alice@example.com" / "reports" / path).is_dir()
    assert not (scaffolded / "drafts" / "agent").exists()

    done = tool(c, token, session, "publish_draft", kind="reports", path=path,
                note="by alice")["structuredContent"]
    assert done["ok"], done
    [row] = c.get("/api/runs", params={"kind": "publish"}).json()
    assert (row["actor"], row["actor_role"]) == ("alice@example.com", "analyst")

    # Two people on one server stay two people.
    _, other = sign_in(c, idp, sub="u-bob", email="bob@example.com", groups=["bi-admins"])
    bob = mcp_session(c, other["access_token"])
    assert tool(c, other["access_token"], bob, "list_drafts")["structuredContent"]["drafts"] == []
    mine = tool(c, token, session, "list_drafts")["structuredContent"]["drafts"]
    assert [d["owner"] for d in mine] == ["alice@example.com"]


def test_a_refresh_token_rotates_and_an_old_one_ends_the_sign_in(signin, idp):
    """Refresh tokens are single use. A rotated one presented again must fail
    with invalid_grant, and ends the sign-in (the new tokens stop working), so a
    leaked old token cannot quietly outlive the rotation."""
    c = signin
    client_id, first = sign_in(c, idp, groups=["bi-analysts"])

    def refresh(token):
        return c.post("/token", data={"grant_type": "refresh_token", "client_id": client_id,
                                      "refresh_token": token})

    second = refresh(first["refresh_token"])
    assert second.status_code == 200, second.text
    second = second.json()
    assert second["refresh_token"] != first["refresh_token"]
    assert second["access_token"] != first["access_token"]
    assert _mcp_call(c, "initialize", token=second["access_token"])[0].status_code == 200

    again = refresh(first["refresh_token"])
    assert again.status_code == 400 and again.json()["error"] == "invalid_grant"
    # The reuse revoked the family: the rotated pair is dead too.
    assert refresh(second["refresh_token"]).json()["error"] == "invalid_grant"
    assert _mcp_call(c, "initialize", token=second["access_token"])[0].status_code == 401


def test_a_refresh_token_of_another_client_is_not_found(signin, idp):
    """invalid_grant, not a token: a refresh token is bound to the client it was
    issued to."""
    c = signin
    _, tokens = sign_in(c, idp)
    stranger = register(c).json()["client_id"]
    resp = c.post("/token", data={"grant_type": "refresh_token", "client_id": stranger,
                                  "refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_grant"


def test_the_role_comes_from_the_groups_and_a_viewer_only_reads(signin, idp):
    c = signin
    _, viewer = sign_in(c, idp, sub="u-v", email="vera@example.com", groups=["staff"])
    token = viewer["access_token"]
    session = mcp_session(c, token)

    assert tool(c, token, session, "list_models")["structuredContent"]
    assert tool(c, token, session, "list_drafts")["structuredContent"]["ok"]
    got = tool(c, token, session, "start_draft", kind="reports", path="x/y")
    assert got["isError"] is True
    assert "viewer" in got["content"][0]["text"] and "start_draft" in got["content"][0]["text"]

    # Every tool that writes is on the viewer's refused list.
    _, listing = _mcp_call(c, "tools/list", session=session, token=token)
    writers = {t["name"] for t in listing["tools"]
               if not (t.get("annotations") or {}).get("readOnlyHint")}
    from tracebi.mcp_server import _WRITE_TOOLS
    assert writers <= _WRITE_TOOLS, writers - _WRITE_TOOLS
    arguments = {
        "render_report_spec": {"spec": {}}, "resolve_pin": {"report": "r", "pin_id": "p"},
        "build_report": {"report": "r"}, "start_draft": {"kind": "reports", "path": "x"},
        "write_draft_file": {"kind": "reports", "path": "x", "file": "report.json", "content": ""},
        "preview_draft": {"kind": "reports", "path": "x"},
        "publish_draft": {"kind": "reports", "path": "x"}}
    assert set(arguments) == _WRITE_TOOLS
    for name, args in arguments.items():
        refused = tool(c, token, session, name, **args)
        assert refused["isError"] is True, name
        assert "your role (viewer)" in refused["content"][0]["text"], (name, refused)

    # The default role is configurable and an unmapped person is not an analyst.
    assert tool(c, token, session, "list_drafts")["structuredContent"]["drafts"] == []


def test_the_static_token_still_works_beside_sign_in(scaffolded, monkeypatch, idp):
    """Automation keeps its shared token (as TRACEBI_MCP_ACTOR, role analyst);
    a wrong token and a person's token for another resource are refused."""
    from fastapi.testclient import TestClient

    from tracebi.web.api import main

    for name, value in {"TRACEBI_OIDC_ISSUER": IDP, "TRACEBI_OIDC_CLIENT_ID": CLIENT_ID,
                        "TRACEBI_OIDC_CLIENT_SECRET": "idp-secret",
                        "TRACEBI_PUBLIC_URL": PUBLIC}.items():
        monkeypatch.setenv(name, value)
    main.serve_gateway("s3cret", oauth=mcp_oauth.GatewayOAuth(mcp_oauth.OAuthConfig.from_env()))
    try:
        with TestClient(main.app) as c:
            assert _mcp_call(c, "initialize", token="wrong")[0].status_code == 401
            session = mcp_session(c, "s3cret")
            got = tool(c, "s3cret", session, "start_draft", kind="models", path="regions")
            assert got["structuredContent"]["owner"] == "agent"
    finally:
        main.unserve_gateway()
    assert (scaffolded / "drafts" / "agent" / "models").is_dir()


# ── Client ID Metadata Documents ─────────────────────────────────────────

CIMD = "https://chatgpt.com/oauth/client-metadata.json"


def _document(**over):
    doc = {"client_id": CIMD, "client_name": "ChatGPT",
           "redirect_uris": [mcp_oauth.CHATGPT_REDIRECT], "token_endpoint_auth_method": "none"}
    doc.update(over)
    return doc


def test_a_client_that_names_itself_by_url_signs_in_without_registering(signin, idp, monkeypatch):
    c = signin
    seen = []
    monkeypatch.setattr(mcp_oauth, "_fetch_client_document",
                        lambda url: seen.append(url) or _document())
    _, tokens = sign_in(c, idp, redirect=mcp_oauth.CHATGPT_REDIRECT, client_id=CIMD)
    assert tokens["access_token"]
    assert len(seen) == 1, "the document is cached, not fetched on every request"


@pytest.mark.parametrize("doc, why", [
    (_document(client_id="https://chatgpt.com/other.json"), "client_id is not its own URL"),
    (_document(redirect_uris=["https://evil.example/cb"]), "redirect not on the allowlist"),
    (_document(token_endpoint_auth_method="client_secret_post"), "a document client is public"),
    (_document(client_secret="x"), "a document carries no secret"),
    ("not a document", "not an object"),
])
def test_a_bad_client_document_is_an_unknown_client(signin, monkeypatch, doc, why):
    c = signin
    monkeypatch.setattr(mcp_oauth, "_fetch_client_document", lambda url: doc)
    resp, _ = authorize(c, CIMD, redirect=mcp_oauth.CHATGPT_REDIRECT)
    assert resp.status_code == 400, why
    assert "location" not in resp.headers


def test_an_unreachable_client_document_is_an_unknown_client(signin, monkeypatch):
    def boom(url):
        raise OSError("no route")
    monkeypatch.setattr(mcp_oauth, "_fetch_client_document", boom)
    resp, _ = authorize(signin, CIMD, redirect=mcp_oauth.CHATGPT_REDIRECT)
    assert resp.status_code == 400


def test_a_client_document_fetch_never_reaches_a_private_address(monkeypatch):
    """The URL is chosen by whoever calls /authorize, so a name that resolves to
    an internal address (the cloud metadata service, localhost) is refused
    before any connection is made."""
    import socket

    def resolves_to(address):
        return lambda host, port, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    for address in ("169.254.169.254", "127.0.0.1", "10.0.0.5", "192.168.1.1", "::1"):
        monkeypatch.setattr(socket, "getaddrinfo", resolves_to(address))
        monkeypatch.setattr(socket, "create_connection",
                            lambda *a, **k: pytest.fail("connected to a private address"))
        with pytest.raises(ValueError, match="public"):
            mcp_oauth._fetch_client_document("https://client.example/doc.json")
    with pytest.raises(ValueError, match="https"):
        mcp_oauth._fetch_client_document("http://client.example/doc.json")


# ── refusals ─────────────────────────────────────────────────────────────

def test_the_wrong_pkce_verifier_gets_no_tokens(signin, idp):
    """PKCE binds the code to the client that asked: a stolen code without the
    verifier is worthless."""
    c = signin
    client_id = register(c).json()["client_id"]
    auth, verifier = authorize(c, client_id)
    code = returned_code(provider_login(c, idp, auth))
    wrong = exchange(c, client_id, code, secrets.token_urlsafe(48))
    assert wrong.status_code == 400 and wrong.json()["error"] == "invalid_grant"
    assert exchange(c, client_id, code, verifier).status_code == 200


def test_an_authorization_code_works_once_and_a_replay_ends_the_sign_in(signin, idp):
    c = signin
    client_id = register(c).json()["client_id"]
    auth, verifier = authorize(c, client_id)
    code = returned_code(provider_login(c, idp, auth))
    first = exchange(c, client_id, code, verifier)
    assert first.status_code == 200
    replay = exchange(c, client_id, code, verifier)
    assert replay.status_code == 400 and replay.json()["error"] == "invalid_grant"
    # What the first redemption made is revoked: the code may have been stolen.
    assert _mcp_call(c, "initialize", token=first.json()["access_token"])[0].status_code == 401


def test_an_expired_code_is_refused(signin, idp):
    c = signin
    client_id = register(c).json()["client_id"]
    auth, verifier = authorize(c, client_id)
    code = returned_code(provider_login(c, idp, auth))
    _age("tracebi_oauth_codes", "expires_at")
    resp = exchange(c, client_id, code, verifier)
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_grant"


def test_another_clients_code_is_not_found(signin, idp):
    c = signin
    one, other = register(c).json()["client_id"], register(c).json()["client_id"]
    auth, verifier = authorize(c, one)
    code = returned_code(provider_login(c, idp, auth))
    resp = exchange(c, other, code, verifier)
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_grant"


def test_pkce_is_required_and_only_s256(signin):
    c = signin
    client_id = register(c).json()["client_id"]
    base = {"response_type": "code", "client_id": client_id, "redirect_uri": CLAUDE}
    missing = c.get("/authorize", params=base, follow_redirects=False)
    assert "error=invalid_request" in missing.headers["location"]
    assert "idp.example.com" not in missing.headers["location"]
    plain = c.get("/authorize", params={**base, "code_challenge": "x" * 43,
                                        "code_challenge_method": "plain"}, follow_redirects=False)
    assert "idp.example.com" not in plain.headers.get("location", "")
    short = c.get("/authorize", params={**base, "code_challenge": "short",
                                        "code_challenge_method": "S256"}, follow_redirects=False)
    assert "idp.example.com" not in short.headers.get("location", "")


def test_a_redirect_uri_off_the_allowlist_is_refused_everywhere(signin, monkeypatch):
    """Registration, authorize, and the per-client match: a code must never be
    sent to an address nobody approved."""
    c = signin
    for bad in ("https://evil.example/cb", "http://localhost.evil.example/cb",
                "http://localhost@evil.example/cb", "https://claude.ai/api/mcp/auth_callback/x",
                "http://claude.ai/api/mcp/auth_callback", "https://localhost:3000/cb",
                "javascript:alert(1)"):
        resp = register(c, bad)
        assert resp.status_code == 400, bad
        assert resp.json()["error"] == "invalid_redirect_uri", bad

    client_id = register(c, CLAUDE).json()["client_id"]
    resp, _ = authorize(c, client_id, redirect="https://evil.example/cb")
    assert resp.status_code == 400 and "location" not in resp.headers

    # A client registered for one allowed address cannot use another.
    resp, _ = authorize(c, client_id, redirect=mcp_oauth.CHATGPT_REDIRECT)
    assert resp.status_code == 400 and "location" not in resp.headers

    # An operator can add exact addresses.
    monkeypatch.setenv("TRACEBI_OAUTH_REDIRECTS", "https://agent.corp.example/cb")
    assert register(c, "https://agent.corp.example/cb").status_code == 201
    assert register(c, "https://agent.corp.example/cb2").status_code == 400


def test_a_loopback_redirect_may_use_any_port(signin, idp):
    """Claude Code and `tracebi login` listen on a random local port."""
    c = signin
    client_id = register(c, "http://127.0.0.1:41234/callback").json()["client_id"]
    redirect = "http://127.0.0.1:50999/callback"
    auth, verifier = authorize(c, client_id, redirect=redirect)
    done = provider_login(c, idp, auth)
    assert done.headers["location"].startswith(redirect + "?")
    code = returned_code(done)
    assert exchange(c, client_id, code, verifier, redirect).status_code == 200
    # Only the port is free: path and host must still match.
    resp, _ = authorize(c, client_id, redirect="http://127.0.0.1:50999/elsewhere")
    assert resp.status_code == 400
    resp, _ = authorize(c, client_id, redirect="http://localhost:50999/callback")
    assert resp.status_code == 400


@pytest.mark.parametrize("uri, allowed", [
    (CLAUDE, True), (mcp_oauth.CHATGPT_REDIRECT, True),
    ("http://localhost:1/x", True), ("http://127.0.0.1/x", True),
    ("http://[::1]:8000/x", False), ("http://127.0.0.1.evil.example/x", False),
    ("https://127.0.0.1:1/x", False), ("http://u:p@localhost:1/x", False),
    ("http://localhost:1/x#frag", False), ("https://claude.ai/other", False),
    ("", False), ("not a url", False),
])
def test_the_redirect_allowlist(uri, allowed):
    assert mcp_oauth.redirect_allowed(uri) is allowed


def test_an_unknown_resource_is_refused(signin):
    c = signin
    client_id = register(c).json()["client_id"]
    resp, _ = authorize(c, client_id, resource="https://elsewhere.example/mcp")
    assert "error=invalid_target" in resp.headers["location"]
    assert "idp.example.com" not in resp.headers["location"]


def test_registration_is_for_public_clients_only(signin):
    c = signin
    resp = register(c, token_endpoint_auth_method="client_secret_post")
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_client_metadata"
    # A client that says nothing is made public, not given a secret.
    silent = c.post("/register", json={"redirect_uris": [CLAUDE]})
    assert silent.status_code == 201
    assert silent.json()["token_endpoint_auth_method"] == "none"
    assert silent.json().get("client_secret") is None


@pytest.mark.parametrize("kind", ["signature", "audience", "nonce", "expired", "issuer",
                                  "alg_none", "no_nonce"])
def test_an_id_token_that_does_not_check_out_signs_nobody_in(signin, idp, kind):
    """Each case is a way a forged or replayed ID token could otherwise log
    someone in as someone else."""
    c = signin
    client_id = register(c).json()["client_id"]
    auth, _ = authorize(c, client_id)
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(auth.headers["location"]).query))
    nonce = q["nonce"]
    token = {
        "signature": lambda: idp.id_token(nonce=nonce, signer=idp.other_key),
        "audience": lambda: idp.id_token(nonce=nonce, aud="some-other-app"),
        "nonce": lambda: idp.id_token(nonce="not-the-nonce"),
        "expired": lambda: idp.id_token(nonce=nonce, ttl=-3600),
        "issuer": lambda: idp.id_token(nonce=nonce, iss="https://evil.example.com"),
        "alg_none": lambda: jwt.encode({"iss": IDP, "aud": CLIENT_ID, "sub": "u", "nonce": nonce,
                                        "exp": int(time.time()) + 300}, None, algorithm="none"),
        "no_nonce": lambda: jwt.encode({"iss": IDP, "aud": CLIENT_ID, "sub": "u",
                                        "exp": int(time.time()) + 300}, idp.key, algorithm="RS256",
                                       headers={"kid": "k1"}),
    }[kind]()
    code = "idp-code"
    idp.pending[code] = {"challenge": q["code_challenge"], "id_token": token}
    done = c.get("/oauth/callback", params={"code": code, "state": q["state"]},
                 headers={"Cookie": cookie_header(auth)}, follow_redirects=False)
    assert done.status_code == 302
    back = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(done.headers["location"]).query))
    assert back["error"] == "access_denied" and "code" not in back
    assert back["state"] == "client-state"
    # The sign-in is spent: the browser is told to drop its cookie, and a
    # replay that kept it gets no code either (the provider's code is spent).
    assert 'tb_oauth_' in done.headers["set-cookie"] and "Max-Age=0" in done.headers["set-cookie"]
    again = c.get("/oauth/callback", params={"code": code, "state": q["state"]},
                  headers={"Cookie": cookie_header(auth)}, follow_redirects=False)
    assert "code=" not in again.headers.get("location", "")


def test_a_sign_in_must_finish_in_the_browser_that_began_it(signin, idp):
    """The link to the provider can be forwarded to someone else. If their
    completing it handed the code to the original requester, anyone could be
    made to sign an attacker's connector in as themselves."""
    c = signin
    client_id = register(c).json()["client_id"]
    auth, _ = authorize(c, client_id)
    assert "tb_oauth_" in auth.headers["set-cookie"] and "HttpOnly" in auth.headers["set-cookie"]
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(auth.headers["location"]).query))
    idp.pending["c1"] = {"challenge": q["code_challenge"], "id_token": idp.id_token(nonce=q["nonce"])}
    stranger = c.get("/oauth/callback", params={"code": "c1", "state": q["state"]},
                     follow_redirects=False)
    assert stranger.status_code == 400 and "different browser" in stranger.text
    assert "c1" not in stranger.text and "location" not in stranger.headers


def test_a_strangers_authorize_requests_write_nothing(signin):
    """/authorize is open to anyone. If each request kept a row until it
    expired, a stranger could grow the store (or, with a cap, lock everyone
    out of sign-in); the in-flight state lives in the browser instead."""
    from sqlalchemy import text

    from tracebi import state

    c = signin
    client_id = register(c).json()["client_id"]
    for _ in range(25):
        assert authorize(c, client_id)[0].status_code == 302
    with state.ensure().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tracebi_oauth_pending")).scalar() == 0


def test_a_tampered_sign_in_cookie_is_refused(signin, idp):
    """The cookie carries the grant (redirect, PKCE challenge); editing it
    must not get past the callback."""
    c = signin
    client_id = register(c).json()["client_id"]
    auth, _ = authorize(c, client_id)
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(auth.headers["location"]).query))
    idp.pending["c1"] = {"challenge": q["code_challenge"], "id_token": idp.id_token(nonce=q["nonce"])}
    (name, value), = auth.cookies.items()
    body, _, mac = value.partition(".")
    grant = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    grant["p"]["state"] = "attacker-state"
    edited = base64.urlsafe_b64encode(json.dumps(grant).encode()).decode().rstrip("=")
    forged = f"{edited}.{mac}"
    done = c.get("/oauth/callback", params={"code": "c1", "state": q["state"]},
                 headers={"Cookie": f"{name}={forged}"}, follow_redirects=False)
    assert done.status_code == 400 and "not valid" in done.text


def test_a_made_up_state_and_an_idp_error_are_handled(signin, idp):
    c = signin
    assert c.get("/oauth/callback", params={"code": "x", "state": "nope"}).status_code == 400
    assert c.get("/oauth/callback").status_code == 400
    client_id = register(c).json()["client_id"]
    auth, _ = authorize(c, client_id)
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(auth.headers["location"]).query))
    denied = c.get("/oauth/callback", params={"error": "access_denied", "state": q["state"]},
                   headers={"Cookie": cookie_header(auth)}, follow_redirects=False)
    assert "error=access_denied" in denied.headers["location"]
    assert denied.headers["location"].startswith(CLAUDE)


def test_an_expired_access_token_is_a_401(signin, idp):
    c = signin
    _, tokens = sign_in(c, idp)
    assert _mcp_call(c, "initialize", token=tokens["access_token"])[0].status_code == 200
    _age("tracebi_oauth_tokens", "expires_at")
    resp = _mcp_call(c, "initialize", token=tokens["access_token"])[0]
    assert resp.status_code == 401
    assert "resource_metadata" in resp.headers["www-authenticate"]


def test_revoking_a_token_ends_the_sign_in(signin, idp):
    c = signin
    client_id, tokens = sign_in(c, idp)
    resp = c.post("/revoke", data={"client_id": client_id, "token": tokens["refresh_token"]})
    assert resp.status_code == 200
    assert _mcp_call(c, "initialize", token=tokens["access_token"])[0].status_code == 401
    refresh = c.post("/token", data={"grant_type": "refresh_token", "client_id": client_id,
                                     "refresh_token": tokens["refresh_token"]})
    assert refresh.json()["error"] == "invalid_grant"


def test_only_hashes_are_stored(signin, idp):
    """A copy of the database must not be a set of working credentials."""
    from tracebi import state

    c = signin
    client_id, tokens = sign_in(c, idp)
    secrets_seen = [tokens["access_token"], tokens["refresh_token"]]
    from sqlalchemy import text
    with state.ensure().connect() as conn:
        dump = ""
        for table in ("clients", "pending", "codes", "tokens"):
            for row in conn.execute(text(f"SELECT * FROM tracebi_oauth_{table}")).fetchall():
                dump += " ".join(str(v) for v in row) + "\n"
    assert dump.strip()
    for secret in secrets_seen:
        assert secret not in dump
        assert hashlib.sha256(secret.encode()).hexdigest() in dump
    assert dump.count("tbat_") == 0 and dump.count("tbrt_") == 0


def test_the_connectors_post_is_not_refused_as_cross_site(signin):
    """/register and /token carry no cookie or Basic credential, so a browser
    client (an inspector) with an Origin must still get through; the app's own
    /api writes keep the guard."""
    c = signin
    resp = c.post("/register", json={"redirect_uris": [CLAUDE]},
                  headers={"Origin": "https://inspector.example"})
    assert resp.status_code == 201
    tok = c.post("/token", data={"grant_type": "refresh_token", "client_id": "x",
                                 "refresh_token": "y"},
                 headers={"Origin": "https://inspector.example"})
    assert tok.status_code != 403
    assert c.post("/api/reports/anything/runs", headers={"Origin": "https://inspector.example"}
                  ).status_code == 403


# ── configuration fails loudly ───────────────────────────────────────────

def test_sign_in_misconfiguration_stops_startup():
    base = {"TRACEBI_OIDC_ISSUER": IDP, "TRACEBI_OIDC_CLIENT_ID": "id",
            "TRACEBI_OIDC_CLIENT_SECRET": "s", "TRACEBI_PUBLIC_URL": PUBLIC}
    assert mcp_oauth.OAuthConfig.from_env({}) is None
    good = mcp_oauth.OAuthConfig.from_env(base)
    assert (good.mcp_url, good.issuer) == (MCP_URL, PUBLIC)
    assert mcp_oauth.OAuthConfig.from_env(
        {**base, "TRACEBI_PUBLIC_MCP_URL": "https://bi.example.com/mcp"}).mcp_url == MCP_URL

    for broken, mention in [
            ({**base, "TRACEBI_OIDC_CLIENT_SECRET": ""}, "TRACEBI_OIDC_CLIENT_SECRET"),
            ({**base, "TRACEBI_PUBLIC_URL": ""}, "TRACEBI_PUBLIC_MCP_URL"),
            ({**base, "TRACEBI_PUBLIC_URL": "http://bi.example.com"}, "https"),
            ({**base, "TRACEBI_OIDC_ISSUER": "http://idp.example.com"}, "https"),
            ({**base, "TRACEBI_OIDC_ROLE_MAP": "staff:superuser"}, "TRACEBI_OIDC_ROLE_MAP"),
            ({**base, "TRACEBI_OIDC_ROLE_MAP": "nocolon"}, "TRACEBI_OIDC_ROLE_MAP"),
            ({**base, "TRACEBI_OIDC_DEFAULT_ROLE": "root"}, "TRACEBI_OIDC_DEFAULT_ROLE")]:
        with pytest.raises(mcp_oauth.OAuthConfigError, match=mention):
            mcp_oauth.OAuthConfig.from_env(broken)


def test_groups_map_to_the_highest_role():
    cfg = mcp_oauth.OAuthConfig.from_env({
        "TRACEBI_OIDC_ISSUER": IDP, "TRACEBI_OIDC_CLIENT_ID": "id",
        "TRACEBI_OIDC_CLIENT_SECRET": "s", "TRACEBI_PUBLIC_URL": PUBLIC,
        "TRACEBI_OIDC_ROLE_MAP": "a:b:admin, analysts:analyst"})
    oidc = mcp_oauth._OIDC(cfg)
    who = lambda groups, **kw: oidc.identity({"sub": "s1", "groups": groups, **kw})   # noqa: E731
    assert who(["analysts", "a:b"])["role"] == "admin"
    assert who(["analysts"])["role"] == "analyst"
    assert who(["unknown"])["role"] == "viewer"
    assert who("analysts")["role"] == "analyst"
    assert who(None)["role"] == "viewer"
    assert who([], email="x@example.com", email_verified=False)["actor"] == "s1"
    assert who([], email="x@example.com")["actor"] == "x@example.com"


def _age(table: str, column: str) -> None:
    """Make every row of *table* long expired."""
    from sqlalchemy import text

    from tracebi import state
    with state.ensure().begin() as conn:
        conn.execute(text(f"UPDATE {table} SET {column} = 1"))


def test_two_requests_racing_for_one_code_or_refresh_token_get_one_answer(signin, idp):
    """The loads above are reads. If two requests both pass them before either
    redeems, the redemption itself must be single use, or both walk away with
    tokens."""
    import asyncio

    from tracebi.mcp_oauth import TokenError

    c = signin
    provider = mcp_oauth_provider()
    client_id = register(c).json()["client_id"]
    auth, verifier = authorize(c, client_id)
    code = returned_code(provider_login(c, idp, auth))

    async def race():
        client = await provider.get_client(client_id)
        loaded = await provider.load_authorization_code(client, code)
        first = await provider.exchange_authorization_code(client, loaded)
        with pytest.raises(TokenError):
            await provider.exchange_authorization_code(client, loaded)

        held = await provider.load_refresh_token(client, first.refresh_token)
        await provider.exchange_refresh_token(client, held, held.scopes)
        with pytest.raises(TokenError):
            await provider.exchange_refresh_token(client, held, held.scopes)
    asyncio.run(race())


def mcp_oauth_provider():
    """The provider the running app was built with."""
    from tracebi.web.api import main
    return main._mcp_server._token_verifier.provider
