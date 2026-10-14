"""
A sign-in per person for the MCP gateway.

TraceBi is its own OAuth 2.1 authorization server for ``/mcp``: ChatGPT and
Claude connectors discover it through protected-resource metadata, register
(a Client ID Metadata Document, or Dynamic Client Registration), and send the
person to ``/authorize``. TraceBi does not hold passwords. It hands the login
to the company identity provider as an ordinary confidential OIDC client
(authorization code + PKCE, the ID token checked against the provider's JWKS),
maps the person's groups to ``viewer`` / ``analyst`` / ``admin``, and issues
its own short-lived access token and a rotating refresh token.

For a demo or a small team with no identity provider there is a built-in
sign-in instead (``TRACEBI_DEMO_LOGIN`` and/or ``TRACEBI_LOCAL_USERS``): the
same authorization server, but the person signs in on a page TraceBi serves at
``/oauth/callback``. Everything after the sign-in is unchanged.

The SDK (``mcp.server.auth``) supplies the endpoint handlers, request
validation and the bearer check. This module supplies the provider behind
them, the OIDC leg, the stores, and the three things the SDK's defaults do
not cover: Client ID Metadata Document clients, the redirect allowlist
(loopback ports ignored), and binding a sign-in to the browser that began it.

Everything stored is in the run store (``tracebi_oauth_*`` tables). Access
tokens, refresh tokens and authorization codes are stored as SHA-256 hashes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import html
import http.client
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Optional

try:
    import jwt
    from mcp.server.auth.handlers.metadata import ProtectedResourceMetadataHandler
    from mcp.server.auth.handlers.register import RegistrationHandler
    from mcp.server.auth.provider import (
        AccessToken,
        AuthorizationCode,
        AuthorizationParams,
        AuthorizeError,
        RefreshToken,
        RegistrationError,
        TokenError,
    )
    from mcp.server.auth.routes import (
        build_metadata,
        cors_middleware,
        create_auth_routes,
    )
    from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
    from mcp.server.transport_security import (
        DEFAULT_MAX_REQUEST_BODY_SIZE,
        RequestBodyLimitMiddleware,
    )
    from mcp.shared.auth import (
        InvalidRedirectUriError,
        OAuthClientInformationFull,
        OAuthMetadata,
        OAuthToken,
        ProtectedResourceMetadata,
    )
    from pydantic import AnyUrl, ValidationError
    from sqlalchemy import text
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse, RedirectResponse, Response
    from starlette.routing import Route, request_response
except ImportError as exc:  # pragma: no cover — exercised by hand
    raise ImportError(
        "Per-person sign-in for the MCP gateway needs the 'mcp' extra "
        "(the MCP SDK, PyJWT with cryptography) and the run store's "
        "SQLAlchemy. Install it with: pip install 'tracebi[mcp,pipeline]'"
    ) from exc

log = logging.getLogger("tracebi.oauth")

ROLES = ("viewer", "analyst", "admin")
_RANK = {r: i for i, r in enumerate(ROLES)}

SCOPES = ("mcp", "offline_access")

ACCESS_TTL = 3600                 # one hour
REFRESH_TTL = 7 * 86400           # sliding, per rotation ...
SESSION_TTL = 30 * 86400          # ... but a sign-in is good for 30 days at most
CODE_TTL = 300
PENDING_TTL = 600
_REGISTRATIONS_PER_HOUR = 100
_MIN_PASSWORD = 8
_MIN_SECRET = 32
_FAIL_WINDOW = 600             # seconds a failed password is remembered
_FAILS_PER_NAME = 10
_FAILS_PER_ADDRESS = 30

CLAUDE_REDIRECT = "https://claude.ai/api/mcp/auth_callback"
# ChatGPT's connector callback. Verify against current OpenAI docs.
CHATGPT_REDIRECT = "https://chatgpt.com/connector_platform_oauth_redirect"

#: Only asymmetric algorithms are accepted for an ID token: never ``none``,
#: never an HMAC keyed by something public.
_ID_TOKEN_ALGS = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512",
                  "ES256", "ES384", "ES512")
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1")
_CHALLENGE = re.compile(r"[A-Za-z0-9_-]{43}")      # base64url(sha256), unpadded
_FLOW = re.compile(r"[A-Za-z0-9_-]{8,32}")
_MAX_DOC_BYTES = 64 * 1024
_MAX_COOKIE = 3800                 # browsers keep at most ~4 KB per cookie
_FETCH_TIMEOUT = 5


class OAuthConfigError(ValueError):
    """The OIDC settings are missing or wrong. Raised at startup, never later."""


def _now() -> float:
    return time.time()


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seal(key: bytes, payload: dict) -> str:
    """*payload* as a tamper-evident cookie value: base64url(JSON) + HMAC."""
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{body}.{hmac.new(key, body.encode(), 'sha256').hexdigest()}"


def unseal(key: bytes, sealed: str) -> Optional[dict]:
    """The payload :func:`seal` made with *key*, or None if it was altered."""
    body, _, mac = (sealed or "").partition(".")
    good = hmac.new(key, body.encode(), "sha256").hexdigest()
    if not body or not hmac.compare_digest(mac, good):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _loopback_url(url: str) -> bool:
    return (urllib.parse.urlsplit(url).hostname or "").lower() in (
        *_LOOPBACK_HOSTS, "::1")


# ── Configuration ────────────────────────────────────────────────────────

def parse_role_map(raw: str) -> dict[str, str]:
    """``bi-admins:admin,bi-analysts:analyst`` -> {group: role}.

    Not ``auth._Authorizer._parse_map``: that one warns and drops, and names
    TRACEBI_AUTH_ROLE_MAP. This map decides who may write, so a bad entry stops
    startup. A group name may itself contain ':' (the role is after the last).
    """
    out: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        group, _, role = entry.rpartition(":")
        group, role = group.strip(), role.strip()
        if not group or role not in _RANK:
            raise OAuthConfigError(
                f"TRACEBI_OIDC_ROLE_MAP entry {entry!r} is not group:role "
                f"with a role from {ROLES}.")
        out[group] = role
    return out


def parse_local_users(raw: str) -> dict[str, tuple[str, str]]:
    """``alice:analyst:her-password,bob:viewer:his-password`` -> {name: (role, password)}.

    The password is everything after the second colon, so it may contain ':'
    but not ','. Like the role map, a bad entry stops startup.
    """
    out: dict[str, tuple[str, str]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, rest = entry.partition(":")
        role, _, password = rest.partition(":")
        name, role = name.strip(), role.strip()
        if not name or role not in _RANK or len(password) < _MIN_PASSWORD:
            raise OAuthConfigError(
                f"TRACEBI_LOCAL_USERS entry for {name or entry!r} is not "
                f"name:role:password with a role from {ROLES} and a password "
                f"of at least {_MIN_PASSWORD} characters.")
        if name in out:
            raise OAuthConfigError(f"TRACEBI_LOCAL_USERS lists {name!r} twice.")
        out[name] = (role, password)
    return out


@dataclass(frozen=True)
class OAuthConfig:
    idp_issuer: str = ""
    client_id: str = ""
    client_secret: str = ""
    scopes: str = "openid email profile"
    groups_claim: str = "groups"
    role_map: dict = dc_field(default_factory=dict)
    default_role: str = "viewer"
    mcp_url: str = ""       # the public MCP URL: the protected resource
    issuer: str = ""        # TraceBi's own issuer: the public origin
    # The built-in sign-in, used when there is no identity provider.
    local_users: dict = dc_field(default_factory=dict)   # name -> (role, password)
    demo_role: Optional[str] = None                   # the role of the one-click demo user
    local_secret: str = ""                            # signs the in-flight sign-in

    @property
    def local(self) -> bool:
        """The built-in sign-in is on (and the identity provider is not)."""
        return not self.idp_issuer

    @property
    def callback_url(self) -> str:
        return f"{self.issuer}/oauth/callback"

    @property
    def login_callback_url(self) -> str:
        """The app's own sign-in return address: a second redirect URI the
        admin registers at the identity provider (see tracebi/web/api/sso.py)."""
        return f"{self.issuer}/login/callback"

    @classmethod
    def from_env(cls, env=None) -> Optional["OAuthConfig"]:
        env = os.environ if env is None else env
        idp = (env.get("TRACEBI_OIDC_ISSUER") or "").strip()
        users_raw = (env.get("TRACEBI_LOCAL_USERS") or "").strip()
        demo = (env.get("TRACEBI_DEMO_LOGIN") or "").strip()
        if not (idp or users_raw or demo):
            return None
        problems = []
        if idp and (users_raw or demo):
            problems.append(
                "TRACEBI_OIDC_ISSUER cannot be combined with TRACEBI_LOCAL_USERS "
                "or TRACEBI_DEMO_LOGIN: choose one way to sign in")
        mcp_url = (env.get("TRACEBI_PUBLIC_MCP_URL") or "").strip()
        if not mcp_url:
            public = (env.get("TRACEBI_PUBLIC_URL") or "").strip()
            if not public:
                problems.append(
                    "set TRACEBI_PUBLIC_MCP_URL (or TRACEBI_PUBLIC_URL) to "
                    "the address people reach this server at")
            else:
                parts = urllib.parse.urlsplit(public)
                mcp_url = f"{parts.scheme}://{parts.netloc}/mcp"
        issuer = ""
        if mcp_url:
            parts = urllib.parse.urlsplit(mcp_url)
            secure = parts.scheme == "https" or (
                parts.scheme == "http" and _loopback_url(mcp_url))
            if not secure or not parts.netloc or parts.query or parts.fragment:
                problems.append(
                    f"the public MCP URL {mcp_url!r} must be an https URL "
                    "with no query or fragment")
            issuer = f"{parts.scheme}://{parts.netloc}".lower()

        if not idp:
            return cls._local_from_env(env, users_raw, demo, mcp_url, issuer, problems)

        if not (idp.startswith("https://") or (
                idp.startswith("http://") and _loopback_url(idp))):
            problems.append("TRACEBI_OIDC_ISSUER must be an https URL")
        client_id = (env.get("TRACEBI_OIDC_CLIENT_ID") or "").strip()
        secret = (env.get("TRACEBI_OIDC_CLIENT_SECRET") or "").strip()
        for name, value in (("TRACEBI_OIDC_CLIENT_ID", client_id),
                            ("TRACEBI_OIDC_CLIENT_SECRET", secret)):
            if not value:
                problems.append(f"{name} is not set")
        scopes = (env.get("TRACEBI_OIDC_SCOPES") or "openid email profile").split()
        if "openid" not in scopes:
            scopes.insert(0, "openid")
        default_role = (env.get("TRACEBI_OIDC_DEFAULT_ROLE") or "viewer").strip()
        if default_role not in _RANK:
            problems.append(
                f"TRACEBI_OIDC_DEFAULT_ROLE must be one of {ROLES}, "
                f"got {default_role!r}")
        try:
            role_map = parse_role_map(env.get("TRACEBI_OIDC_ROLE_MAP") or "")
        except OAuthConfigError as exc:
            problems.append(str(exc))
            role_map = {}
        if problems:
            raise OAuthConfigError(
                "Per-person MCP sign-in (TRACEBI_OIDC_ISSUER is set) is "
                "misconfigured: " + "; ".join(problems) + ".")
        return cls(
            idp_issuer=idp, client_id=client_id, client_secret=secret,
            scopes=" ".join(scopes),
            groups_claim=(env.get("TRACEBI_OIDC_GROUPS_CLAIM") or "groups").strip(),
            role_map=role_map, default_role=default_role,
            mcp_url=mcp_url, issuer=issuer)

    @classmethod
    def _local_from_env(cls, env, users_raw, demo, mcp_url, issuer, problems):
        try:
            users = parse_local_users(users_raw)
        except OAuthConfigError as exc:
            problems.append(str(exc))
            users = {}
        if demo and demo not in _RANK:
            problems.append(
                f"TRACEBI_DEMO_LOGIN must be a role from {ROLES}, got {demo!r}")
        secret = (env.get("TRACEBI_OAUTH_SECRET") or "").strip()
        if len(secret) < _MIN_SECRET:
            problems.append(
                f"TRACEBI_OAUTH_SECRET must be a random secret of at least "
                f"{_MIN_SECRET} characters (it signs the sign-in in flight)")
        if problems:
            raise OAuthConfigError(
                "Built-in MCP sign-in (TRACEBI_LOCAL_USERS / TRACEBI_DEMO_LOGIN "
                "is set) is misconfigured: " + "; ".join(problems) + ".")
        return cls(mcp_url=mcp_url, issuer=issuer, local_users=users,
                   demo_role=demo or None, local_secret=secret)


# ── Redirect allowlist ───────────────────────────────────────────────────

def _canon(uri: str) -> str:
    return str(AnyUrl(uri))


def _fixed_redirects() -> set[str]:
    extra = [u.strip() for u in os.environ.get(
        "TRACEBI_OAUTH_REDIRECTS", "").split(",") if u.strip()]
    out = set()
    for uri in (CLAUDE_REDIRECT, CHATGPT_REDIRECT, *extra):
        try:
            out.add(_canon(uri))
        except ValidationError:
            log.warning("TRACEBI_OAUTH_REDIRECTS entry %r is not a URL; ignored", uri)
    return out


def _loopback_parts(uri: str):
    """(host, path, query) when *uri* is a plain-http loopback redirect."""
    parts = urllib.parse.urlsplit(uri)
    if (parts.scheme != "http" or parts.username or parts.password
            or parts.fragment
            or (parts.hostname or "").lower() not in _LOOPBACK_HOSTS):
        return None
    return parts.hostname.lower(), parts.path or "/", parts.query


def redirect_allowed(uri: str) -> bool:
    """Exactly the connectors' callbacks, loopback on any port, or an entry in
    TRACEBI_OAUTH_REDIRECTS. Never a fragment, never userinfo."""
    try:
        canon = _canon(uri)
    except ValidationError:
        return False
    if urllib.parse.urlsplit(canon).fragment:
        return False
    return canon in _fixed_redirects() or _loopback_parts(canon) is not None


def _same_redirect(requested: str, registered: str) -> bool:
    """Exact, except that a loopback redirect may differ in port."""
    if requested == registered:
        return True
    a, b = _loopback_parts(requested), _loopback_parts(registered)
    return a is not None and a == b


# ── The two outbound fetches (tests replace these) ───────────────────────

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _read_capped(resp, limit: int) -> bytes:
    body = resp.read(limit + 1)
    if len(body) > limit:
        raise ValueError("response too large")
    return body


def _fetch_json(url: str, form: Optional[dict] = None) -> dict:
    """GET (or form POST) the identity provider. No redirects, size-capped.
    The provider's address is operator configuration, so private networks are
    allowed here, and an egress proxy from the environment is honoured."""
    if not (url.startswith("https://") or (
            url.startswith("http://") and _loopback_url(url))):
        raise ValueError(f"refusing non-https URL {url!r}")
    data = urllib.parse.urlencode(form).encode() if form is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(req, timeout=_FETCH_TIMEOUT) as resp:
        return json.loads(_read_capped(resp, _MAX_DOC_BYTES))


class _PinnedHTTPS(http.client.HTTPSConnection):
    """Connect to an address already checked, still verifying the host name."""

    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout,
                         context=ssl.create_default_context())
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _fetch_client_document(url: str) -> dict:
    """GET a Client ID Metadata Document. The URL is chosen by whoever calls
    /authorize, so: https only, public addresses only (checked on the address
    actually connected to), no redirects, size and time capped."""
    parts = urllib.parse.urlsplit(url)
    host, port = parts.hostname, parts.port or 443
    if parts.scheme != "https" or not host:
        raise ValueError("a client metadata document must be an https URL")
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses = [i[4][0] for i in infos]
    if not addresses or not all(ipaddress.ip_address(a).is_global for a in addresses):
        raise ValueError("client metadata document host is not a public address")
    conn = _PinnedHTTPS(host, port, addresses[0], _FETCH_TIMEOUT)
    try:
        target = parts.path + (f"?{parts.query}" if parts.query else "")
        conn.request("GET", target, headers={
            "Host": parts.netloc, "Accept": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise ValueError(f"client metadata document answered {resp.status}")
        return json.loads(_read_capped(resp, _MAX_DOC_BYTES))
    finally:
        conn.close()


# ── The identity provider (OIDC) ─────────────────────────────────────────

class SignInError(Exception):
    """The provider's answer could not be trusted. The person sees a generic
    refusal; the reason goes to the server log only."""


def _endpoint_ok(url: Any) -> bool:
    return isinstance(url, str) and (
        url.startswith("https://") or (url.startswith("http://") and _loopback_url(url)))


class _OIDC:
    def __init__(self, cfg: OAuthConfig):
        self.cfg = cfg
        self._doc: Optional[tuple[float, dict]] = None
        self._keys: Optional[tuple[float, list]] = None
        self._keys_fetched = 0.0

    def discovery(self) -> dict:
        if self._doc and self._doc[0] > _now():
            return self._doc[1]
        base = self.cfg.idp_issuer.rstrip("/")
        doc = _fetch_json(f"{base}/.well-known/openid-configuration")
        if str(doc.get("issuer", "")).rstrip("/") != base:
            raise SignInError("discovery document names a different issuer")
        for field in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not _endpoint_ok(doc.get(field)):
                raise SignInError(f"discovery document has no usable {field}")
        self._doc = (_now() + 3600, doc)
        return doc

    def _jwks(self, force: bool = False) -> list:
        if not force and self._keys and self._keys[0] > _now():
            return self._keys[1]
        # An unknown key id is attacker-controlled input: refetch at most
        # once a minute, so it cannot be used to hammer the provider.
        if force and _now() - self._keys_fetched < 60:
            return self._keys[1] if self._keys else []
        keys = _fetch_json(self.discovery()["jwks_uri"]).get("keys", [])
        self._keys_fetched = _now()
        self._keys = (_now() + 3600, keys)
        return keys

    def authorize_url(self, state: str, nonce: str, challenge: str,
                      redirect_uri: Optional[str] = None) -> str:
        query = urllib.parse.urlencode({
            "response_type": "code", "client_id": self.cfg.client_id,
            "redirect_uri": redirect_uri or self.cfg.callback_url,
            "scope": self.cfg.scopes,
            "state": state, "nonce": nonce,
            "code_challenge": challenge, "code_challenge_method": "S256"})
        return f"{self.discovery()['authorization_endpoint']}?{query}"

    def _key_for(self, token: str):
        header = jwt.get_unverified_header(token)
        alg = header.get("alg")
        if alg not in _ID_TOKEN_ALGS:
            raise SignInError(f"ID token signed with unsupported algorithm {alg!r}")
        kid = header.get("kid")

        def find(keys):
            matches = [k for k in keys if k.get("kid") == kid and
                       k.get("use", "sig") == "sig"] if kid else (
                           keys if len(keys) == 1 else [])
            return matches[0] if matches else None

        jwk = find(self._jwks()) or find(self._jwks(force=True))
        if jwk is None:
            raise SignInError("ID token key id is not in the provider's key set")
        return alg, jwt.PyJWK(jwk).key

    def verify_id_token(self, token: str, nonce: str) -> dict:
        try:
            alg, key = self._key_for(token)
            claims = jwt.decode(
                token, key, algorithms=[alg],
                audience=self.cfg.client_id,
                issuer=self.discovery()["issuer"],
                leeway=60,
                options={"require": ["exp", "iss", "aud", "sub"]})
        except SignInError:
            raise
        except jwt.PyJWTError as exc:
            raise SignInError(f"ID token rejected: {exc}") from exc
        got = claims.get("nonce")
        if not isinstance(got, str) or not hmac.compare_digest(got, nonce):
            raise SignInError("ID token nonce does not match this sign-in")
        aud = claims["aud"]
        if isinstance(aud, list) and len(aud) > 1 and claims.get("azp") != self.cfg.client_id:
            raise SignInError("ID token authorized party is not this app")
        return claims

    def sign_in(self, code: str, verifier: str, nonce: str,
                redirect_uri: Optional[str] = None) -> dict:
        try:
            reply = _fetch_json(self.discovery()["token_endpoint"], form={
                "grant_type": "authorization_code", "code": code,
                "redirect_uri": redirect_uri or self.cfg.callback_url,
                "client_id": self.cfg.client_id,
                "client_secret": self.cfg.client_secret,
                "code_verifier": verifier})
        except (urllib.error.URLError, ValueError, OSError) as exc:
            raise SignInError(f"code exchange failed: {exc}") from exc
        token = reply.get("id_token")
        if not isinstance(token, str):
            raise SignInError("token response has no id_token")
        return self.verify_id_token(token, nonce)

    def identity(self, claims: dict) -> dict:
        email = claims.get("email")
        if not isinstance(email, str) or claims.get("email_verified") is False:
            email = None
        actor = email or claims.get("preferred_username") or claims.get("upn") \
            or claims["sub"]
        groups = claims.get(self.cfg.groups_claim) or []
        if isinstance(groups, str):
            groups = [groups]
        role = self.cfg.default_role
        for group in groups if isinstance(groups, list) else []:
            mapped = self.cfg.role_map.get(group)
            if mapped and _RANK[mapped] > _RANK[role]:
                role = mapped
        return {"sub": str(claims["sub"]), "actor": str(actor)[:200], "role": role}


# ── Models the SDK hands back to us ──────────────────────────────────────

class _Client(OAuthClientInformationFull):
    """A registered client whose redirect check is the allowlist and whose
    scopes are the ones TraceBi knows (others are dropped, not an error: the
    token's reach is the person's role, not a scope)."""

    def validate_redirect_uri(self, redirect_uri):
        registered = [str(u) for u in (self.redirect_uris or [])]
        if redirect_uri is None:
            if len(registered) != 1:
                raise InvalidRedirectUriError(
                    "redirect_uri must be specified unless the client has "
                    "exactly one registered URI")
            chosen = registered[0]
        else:
            chosen = str(redirect_uri)
            if not any(_same_redirect(chosen, r) for r in registered):
                raise InvalidRedirectUriError(
                    f"Redirect URI '{chosen}' not registered for client")
        if not redirect_allowed(chosen):
            raise InvalidRedirectUriError(
                f"Redirect URI '{chosen}' is not allowed by this server")
        return AnyUrl(chosen)

    def validate_scope(self, requested_scope):
        if requested_scope is None:
            return None
        return [s for s in requested_scope.split() if s in SCOPES] or None


class _Code(AuthorizationCode):
    family_id: str
    sub: str
    actor: str
    role: str
    session_expires_at: int


class _Refresh(RefreshToken):
    family_id: str
    actor: str
    role: str
    session_expires_at: int


# ── The authorization server ─────────────────────────────────────────────

#: Set by the /authorize wrapper so the provider (which only returns a URL)
#: can bind the sign-in to the browser that began it.
_flow: ContextVar[Optional[dict]] = ContextVar("tracebi_oauth_flow", default=None)


def _client_doc_url(client_id: str) -> bool:
    parts = urllib.parse.urlsplit(client_id)
    return (parts.scheme == "https" and bool(parts.hostname)
            and parts.path not in ("", "/") and not parts.fragment
            and not parts.username and len(client_id) <= 2048
            and "/./" not in parts.path and "/../" not in parts.path)


class Provider:
    """OAuthAuthorizationServerProvider backed by the run store."""

    def __init__(self, cfg: OAuthConfig, oidc: Optional[_OIDC] = None):
        self.cfg = cfg
        self.oidc = oidc or _OIDC(cfg)
        self._documents: dict[str, tuple[float, Optional[_Client]]] = {}
        # Failed built-in sign-ins: key -> times. In this process only, which
        # is how the built-in sign-in is meant to run (one worker).
        self._failures: dict[str, list[float]] = {}

    # -- store ----------------------------------------------------------
    @staticmethod
    def _engine():
        from tracebi import state
        return state.ensure()

    def _exec(self, sql: str, **p) -> int:
        with self._engine().begin() as conn:
            return conn.execute(text(sql), p).rowcount

    def _row(self, sql: str, **p) -> Optional[dict]:
        with self._engine().connect() as conn:
            row = conn.execute(text(sql), p).mappings().first()
        return dict(row) if row else None

    def _purge(self) -> None:
        now = int(_now())
        self._exec("DELETE FROM tracebi_oauth_codes WHERE expires_at < :n", n=now - 3600)
        self._exec("DELETE FROM tracebi_oauth_tokens WHERE expires_at < :n", n=now - 86400)

    def _sign_in_key(self) -> bytes:
        """Signs the in-flight sign-in cookie. Derived from the identity
        provider client secret (or, for the built-in sign-in, TRACEBI_OAUTH_SECRET),
        which every worker shares and nobody else has."""
        secret = self.cfg.local_secret if self.cfg.local else self.cfg.client_secret
        return hashlib.sha256(b"tracebi-oauth:" + secret.encode()).digest()

    def _revoke_family(self, family_id: str) -> None:
        self._exec("UPDATE tracebi_oauth_tokens SET revoked_at = :n "
                   "WHERE family_id = :f AND revoked_at IS NULL",
                   n=int(_now()), f=family_id)

    # -- clients --------------------------------------------------------
    async def get_client(self, client_id: str):
        if _client_doc_url(client_id):
            return await asyncio.to_thread(self._document_client, client_id)
        row = await asyncio.to_thread(
            self._row, "SELECT info FROM tracebi_oauth_clients WHERE client_id = :c",
            c=client_id)
        return _Client.model_validate_json(row["info"]) if row else None

    def _document_client(self, url: str) -> Optional[_Client]:
        cached = self._documents.get(url)
        if cached and cached[0] > _now():
            return cached[1]
        client = None
        try:
            client = self._parse_document(url, _fetch_client_document(url))
        except Exception as exc:  # noqa: BLE001 — an unreachable or bad document is an unknown client
            log.warning("client metadata document %s refused: %s", url, exc)
        if len(self._documents) >= 256:
            self._documents.pop(min(self._documents, key=lambda k: self._documents[k][0]))
        # A refusal is remembered briefly, so /authorize cannot be used to make
        # this server fetch the same address over and over.
        self._documents[url] = (_now() + (3600 if client else 60), client)
        return client

    @staticmethod
    def _parse_document(url: str, doc: Any) -> Optional[_Client]:
        if not isinstance(doc, dict) or doc.get("client_id") != url:
            raise ValueError("document client_id is not its own URL")
        if doc.get("token_endpoint_auth_method", "none") != "none" \
                or "client_secret" in doc:
            raise ValueError("a metadata-document client must be public")
        uris = doc.get("redirect_uris")
        if not isinstance(uris, list) or not uris:
            raise ValueError("document lists no redirect_uris")
        good = [u for u in uris if isinstance(u, str) and redirect_allowed(u)]
        if not good:
            raise ValueError("none of the document's redirect_uris is allowed")
        return _Client(
            client_id=url, redirect_uris=[AnyUrl(u) for u in good],
            token_endpoint_auth_method="none",
            grant_types=["authorization_code", "refresh_token"],
            client_name=str(doc.get("client_name") or "")[:200] or None,
            scope=" ".join(SCOPES))

    async def register_client(self, client_info) -> None:
        if client_info.token_endpoint_auth_method != "none":
            raise RegistrationError(
                "invalid_client_metadata",
                "only public clients are supported: use "
                "token_endpoint_auth_method 'none' (PKCE protects the code)")
        uris = [str(u) for u in (client_info.redirect_uris or [])]
        if not uris:
            raise RegistrationError("invalid_redirect_uri", "redirect_uris is required")
        for uri in uris:
            if not redirect_allowed(uri):
                raise RegistrationError(
                    "invalid_redirect_uri", f"redirect URI {uri} is not allowed")
        record = _Client.model_validate(client_info.model_dump(mode="json"))
        record.scope = " ".join(SCOPES)

        def store():
            now = int(_now())
            recent = self._row(
                "SELECT COUNT(*) AS n FROM tracebi_oauth_clients WHERE created_at > :t",
                t=now - 3600)["n"]
            if recent >= _REGISTRATIONS_PER_HOUR:
                raise RegistrationError(
                    "invalid_client_metadata",
                    "too many registrations just now; try again later")
            self._exec("INSERT INTO tracebi_oauth_clients (client_id, info, created_at) "
                       "VALUES (:c, :i, :t)", c=record.client_id,
                       i=record.model_dump_json(), t=now)
        await asyncio.to_thread(store)

    # -- authorize: the person is sent to the identity provider ---------
    async def authorize(self, client, params: AuthorizationParams) -> str:
        if params.resource and params.resource.rstrip("/") != self.cfg.mcp_url.rstrip("/"):
            raise AuthorizeError("invalid_target", "unknown resource")
        if not _CHALLENGE.fullmatch(params.code_challenge or ""):
            raise AuthorizeError("invalid_request", "code_challenge must be an S256 challenge")
        flow = _flow.get()
        if not flow:
            raise AuthorizeError("server_error", "sign-in is not available here")
        nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        state = f"{flow['id']}.{secrets.token_urlsafe(32)}"
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        grant = {
            "redirect_uri": str(params.redirect_uri),
            "explicit": params.redirect_uri_provided_explicitly,
            "code_challenge": params.code_challenge,
            "scopes": params.scopes or ["mcp"], "state": params.state,
            "resource": self.cfg.mcp_url}

        # What the return trip needs travels in a signed cookie in this
        # browser, so a stranger's /authorize writes nothing on the server.
        sealed = seal(self._sign_in_key(), {
            "s": _h(state), "c": client.client_id, "p": grant, "n": nonce,
            "v": verifier, "e": int(_now()) + PENDING_TTL})
        if len(sealed) > _MAX_COOKIE:
            raise AuthorizeError("invalid_request", "the request is too large")
        if self.cfg.local:
            # No identity provider: the person signs in on TraceBi's own page,
            # which is the same address the provider would have returned to.
            flow["sealed"] = sealed
            return f"{self.cfg.issuer}/oauth/callback?{urllib.parse.urlencode({'state': state})}"
        try:
            url = await asyncio.to_thread(self.oidc.authorize_url, state, nonce, challenge)
        except (SignInError, OSError, ValueError, urllib.error.URLError) as exc:
            log.error("cannot reach the identity provider: %s", exc)
            raise AuthorizeError("server_error", "the identity provider is unreachable") from exc
        flow["sealed"] = sealed
        return url

    # -- the sign-in comes back -----------------------------------------
    def _pending(self, request: Request, state: str):
        """The in-flight sign-in this browser began, as ``(row, cookie_name)``,
        or ``(None, error response)`` when the link is not this browser's own,
        has expired or was altered."""
        flow_id = state.partition(".")[0]
        if not _FLOW.fullmatch(flow_id):
            return None, _bad("This sign-in link is not valid. Start again from your AI app.")
        cookie_name = f"tb_oauth_{flow_id}"
        sealed = request.cookies.get(cookie_name)
        if not sealed:
            # The sign-in was begun in a different browser: a link someone
            # else made, not this person's own request.
            log.warning("sign-in refused: callback came from a browser that did not begin it")
            return None, _bad("This sign-in was started in a different browser. "
                              "Start again from your AI app.")
        row = unseal(self._sign_in_key(), sealed)
        if (row is None or not hmac.compare_digest(str(row.get("s", "")), _h(state))
                or int(row.get("e", 0)) < _now()):
            return None, _bad("This sign-in link is not valid or has expired. "
                              "Start again from your AI app.")
        if not redirect_allowed(row["p"]["redirect_uri"]):
            return None, _bad("The redirect address is no longer allowed.")
        return row, cookie_name

    @staticmethod
    def _back(row: dict, cookie_name: str, **params) -> Response:
        grant = row["p"]
        resp = RedirectResponse(
            _with_query(grant["redirect_uri"], state=grant.get("state"), **params),
            status_code=302, headers={"Cache-Control": "no-store"})
        resp.delete_cookie(cookie_name, path="/oauth/callback")
        return resp

    async def _issue_code(self, row: dict, who: dict) -> str:
        """An authorization code for the person *who* signed in."""
        issued = secrets.token_urlsafe(32)
        now = int(_now())
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO tracebi_oauth_codes (code_hash, client_id, params, identity, "
            "family_id, expires_at, used_at) VALUES (:h, :c, :p, :i, :f, :e, NULL)",
            h=_h(issued), c=row["c"], p=json.dumps(row["p"]),
            i=json.dumps({**who, "session_expires_at": now + SESSION_TTL}),
            f=secrets.token_urlsafe(16), e=now + CODE_TTL)
        return issued

    async def callback(self, request: Request) -> Response:
        """The identity provider answers."""
        q = request.query_params
        row, cookie_name = self._pending(request, q.get("state") or "")
        if row is None:
            return cookie_name
        code = q.get("code")
        if q.get("error") or not code:
            return self._back(row, cookie_name, error="access_denied",
                              error_description="sign-in was not completed")
        try:
            claims = await asyncio.to_thread(
                self.oidc.sign_in, code, row["v"], row["n"])
            who = self.oidc.identity(claims)
        except Exception as exc:  # noqa: BLE001 — whatever went wrong, the answer is no
            log.warning("sign-in refused: %s", exc)
            return self._back(row, cookie_name, error="access_denied",
                              error_description="sign-in could not be verified")
        return self._back(row, cookie_name, code=await self._issue_code(row, who))

    # -- the built-in sign-in -------------------------------------------
    def _throttled(self, *keys: str) -> bool:
        now = _now()
        for key, limit in zip(keys, (_FAILS_PER_NAME, _FAILS_PER_ADDRESS)):
            recent = [t for t in self._failures.get(key, []) if t > now - _FAIL_WINDOW]
            if recent:
                self._failures[key] = recent
            else:
                self._failures.pop(key, None)
            if len(recent) >= limit:
                return True
        return False

    def _failed(self, *keys: str) -> None:
        now = _now()
        if len(self._failures) > 10_000:       # an attacker inventing names
            self._failures.clear()
        for key in keys:
            self._failures.setdefault(key, []).append(now)

    def _check_password(self, name: str, password: str) -> Optional[dict]:
        """The person *name* with *password*, or None. Constant-time, and an
        unknown name costs the same as a wrong password."""
        role, expected = self.cfg.local_users.get(name, (None, secrets.token_urlsafe(16)))
        ok = hmac.compare_digest(password.encode(), expected.encode())
        if role is None or not ok:
            return None
        return {"sub": f"local:{name}", "actor": name[:200], "role": role}

    async def local_login(self, request: Request) -> Response:
        """The built-in sign-in page (GET) and its answer (POST)."""
        posted = request.method == "POST"
        form = await request.form() if posted else {}
        state = str(form.get("state") if posted else request.query_params.get("state") or "")
        row, cookie_name = self._pending(request, state)
        if row is None:
            return cookie_name
        if not posted:
            return _login_page(state, self.cfg)
        address = request.client.host if request.client else "?"
        name = str(form.get("username") or "")[:200]
        if form.get("demo") and self.cfg.demo_role:
            who = {"sub": "local:demo", "actor": "demo", "role": self.cfg.demo_role}
        else:
            keys = (f"name:{name.lower()}", f"address:{address}")
            if self._throttled(*keys):
                log.warning("built-in sign-in throttled for %r from %s", name, address)
                return _login_page(state, self.cfg, "Too many attempts. Wait a few minutes and try again.", 429)
            who = self._check_password(name, str(form.get("password") or ""))
            if who is None:
                self._failed(*keys)
                log.warning("built-in sign-in refused for %r from %s", name, address)
                return _login_page(state, self.cfg, "That name and password do not match.", 401)
        return self._back(row, cookie_name, code=await self._issue_code(row, who))

    # -- authorization code -> tokens -----------------------------------
    async def load_authorization_code(self, client, authorization_code: str):
        def load():
            row = self._row("SELECT * FROM tracebi_oauth_codes WHERE code_hash = :h",
                            h=_h(authorization_code))
            if row is None or row["client_id"] != client.client_id:
                return None
            if row["used_at"] is not None:
                # A code presented twice: whoever holds the tokens it made
                # may be the thief, so they all stop working (RFC 6749 4.1.2).
                self._revoke_family(row["family_id"])
                return None
            grant, who = json.loads(row["params"]), json.loads(row["identity"])
            return _Code(
                code=authorization_code, scopes=grant["scopes"],
                expires_at=row["expires_at"], client_id=row["client_id"],
                code_challenge=grant["code_challenge"],
                redirect_uri=AnyUrl(grant["redirect_uri"]),
                redirect_uri_provided_explicitly=grant["explicit"],
                resource=grant["resource"], subject=who["sub"],
                family_id=row["family_id"], sub=who["sub"], actor=who["actor"],
                role=who["role"], session_expires_at=who["session_expires_at"])
        return await asyncio.to_thread(load)

    async def exchange_authorization_code(self, client, authorization_code) -> OAuthToken:
        def exchange():
            used = self._exec(
                "UPDATE tracebi_oauth_codes SET used_at = :n WHERE code_hash = :h "
                "AND used_at IS NULL AND expires_at >= :n",
                n=int(_now()), h=_h(authorization_code.code))
            if used != 1:
                raise TokenError("invalid_grant", "authorization code is no longer valid")
            return self._issue(
                authorization_code.family_id, client.client_id,
                authorization_code.sub, authorization_code.actor,
                authorization_code.role, authorization_code.scopes,
                authorization_code.resource, authorization_code.session_expires_at)
        return await asyncio.to_thread(exchange)

    def _issue(self, family, client_id, sub, actor, role, scopes, resource,
               session_expires_at) -> OAuthToken:
        now = int(_now())
        access, refresh = "tbat_" + secrets.token_urlsafe(32), "tbrt_" + secrets.token_urlsafe(32)
        who = json.dumps({"sub": sub, "actor": actor, "role": role})
        refresh_expires = min(now + REFRESH_TTL, session_expires_at)
        sql = ("INSERT INTO tracebi_oauth_tokens (token_hash, kind, family_id, client_id, "
               "identity, scopes, resource, expires_at, session_expires_at) VALUES "
               "(:h, :k, :f, :c, :i, :s, :r, :e, :x)")
        for token, kind, expires in ((access, "access", now + ACCESS_TTL),
                                     (refresh, "refresh", refresh_expires)):
            self._exec(sql, h=_h(token), k=kind, f=family, c=client_id, i=who,
                       s=" ".join(scopes), r=resource, e=expires, x=session_expires_at)
        return OAuthToken(access_token=access, expires_in=ACCESS_TTL,
                          scope=" ".join(scopes), refresh_token=refresh)

    # -- refresh --------------------------------------------------------
    async def load_refresh_token(self, client, refresh_token: str):
        def load():
            row = self._row("SELECT * FROM tracebi_oauth_tokens WHERE token_hash = :h "
                            "AND kind = 'refresh'", h=_h(refresh_token))
            if row is None or row["client_id"] != client.client_id or row["revoked_at"]:
                return None
            if row["rotated_at"] is not None:
                # An old refresh token again: either it leaked or a client
                # kept it. End the whole sign-in; the person signs in again.
                self._revoke_family(row["family_id"])
                return None
            who = json.loads(row["identity"])
            return _Refresh(
                token=row["token_hash"], client_id=row["client_id"],
                scopes=row["scopes"].split(), expires_at=row["expires_at"],
                resource=row["resource"], subject=who["sub"],
                family_id=row["family_id"], actor=who["actor"], role=who["role"],
                session_expires_at=row["session_expires_at"])
        return await asyncio.to_thread(load)

    async def exchange_refresh_token(self, client, refresh_token, scopes) -> OAuthToken:
        def rotate():
            used = self._exec(
                "UPDATE tracebi_oauth_tokens SET rotated_at = :n WHERE token_hash = :h "
                "AND kind = 'refresh' AND rotated_at IS NULL AND revoked_at IS NULL "
                "AND expires_at >= :n", n=int(_now()), h=refresh_token.token)
            if used != 1:
                raise TokenError("invalid_grant", "refresh token is no longer valid")
            return self._issue(
                refresh_token.family_id, client.client_id, refresh_token.subject,
                refresh_token.actor, refresh_token.role, scopes or refresh_token.scopes,
                refresh_token.resource, refresh_token.session_expires_at)
        return await asyncio.to_thread(rotate)

    # -- access tokens --------------------------------------------------
    async def load_access_token(self, token: str):
        def load():
            row = self._row("SELECT * FROM tracebi_oauth_tokens WHERE token_hash = :h "
                            "AND kind = 'access'", h=_h(token))
            if row is None or row["revoked_at"] or row["expires_at"] < _now():
                return None
            who = json.loads(row["identity"])
            return AccessToken(
                token=row["token_hash"], client_id=row["client_id"],
                scopes=row["scopes"].split(), expires_at=row["expires_at"],
                resource=row["resource"], subject=who["sub"],
                claims={"iss": self.cfg.issuer, "tracebi_actor": who["actor"],
                        "tracebi_role": who["role"]})
        return await asyncio.to_thread(load)

    async def revoke_token(self, token) -> None:
        def revoke():
            # Both token kinds carry their stored hash, never the raw value.
            row = self._row("SELECT family_id FROM tracebi_oauth_tokens WHERE token_hash = :h",
                            h=token.token)
            if row:
                self._revoke_family(row["family_id"])
        await asyncio.to_thread(revoke)


def _login_page(state: str, cfg: OAuthConfig, message: str = "", status: int = 200) -> Response:
    """The built-in sign-in page: a name and password when TRACEBI_LOCAL_USERS
    is set, and a one-click button when TRACEBI_DEMO_LOGIN is."""
    q = html.escape
    fields = ""
    if cfg.local_users:
        fields += (
            '<label>Name<input name="username" autocomplete="username" required autofocus></label>'
            '<label>Password<input name="password" type="password" '
            'autocomplete="current-password" required></label>'
            '<button type="submit">Sign in</button>')
    demo = ""
    if cfg.demo_role:
        article = "an" if cfg.demo_role[0] in "aeiou" else "a"
        demo = (
            ('<p class="or">or</p>' if cfg.local_users else "")
            + '<button type="submit" name="demo" value="1" formnovalidate class="demo">'
            f'Continue as the demo user</button><p class="note">The demo user is {article} {q(cfg.demo_role)}.</p>')
    body = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Sign in to TraceBi</title><style>'
        'body{margin:0;min-height:100vh;display:grid;place-items:center;background:#fafafa;'
        'font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:#171717}'
        'main{width:min(380px,92vw);background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:28px}'
        'h1{margin:0 0 4px;font-size:20px}p{margin:0 0 16px;color:#404040}'
        'label{display:block;margin:0 0 14px;font-size:13px;font-weight:600}'
        'input{display:block;width:100%;box-sizing:border-box;margin-top:6px;height:40px;padding:0 10px;'
        'border:1px solid #d4d4d4;border-radius:8px;font:inherit}'
        'button{width:100%;height:40px;border:0;border-radius:8px;background:#2f5bea;color:#fff;'
        'font:inherit;font-weight:600;cursor:pointer}'
        'button.demo{background:#fff;color:#171717;border:1px solid #d4d4d4}'
        '.or{text-align:center;margin:14px 0}.note{margin:8px 0 0;font-size:13px;color:#666}'
        '.err{color:#b91c1c;font-weight:600}'
        ':focus-visible{outline:2px solid #2f5bea;outline-offset:2px}'
        '</style></head><body><main><h1>Sign in to TraceBi</h1>'
        '<p>An AI app is asking to connect to this server as you.</p>'
        + (f'<p class="err" role="alert">{q(message)}</p>' if message else "")
        + f'<form method="post" action="/oauth/callback"><input type="hidden" name="state" value="{q(state)}">'
        + fields + demo + '</form></main></body></html>')
    return Response(body, status_code=status, media_type="text/html", headers={
        "Cache-Control": "no-store", "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
        "Referrer-Policy": "no-referrer"})


def _bad(message: str) -> Response:
    return PlainTextResponse(message, status_code=400, headers={"Cache-Control": "no-store"})


def _with_query(url: str, **params) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query += [(k, v) for k, v in params.items() if v is not None]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


# ── The verifier and the routes the app serves ───────────────────────────

class GatewayTokenVerifier:
    """TraceBi access tokens, plus the static automation token when set."""

    def __init__(self, provider: Provider, mcp_url: str, static=None):
        self.provider, self.mcp_url, self.static = provider, mcp_url, static

    async def verify_token(self, token: str):
        if self.static is not None:
            found = await self.static.verify_token(token)
            if found:
                return found.model_copy(update={
                    "resource": self.mcp_url,
                    "claims": {"tracebi_role": "analyst"}})
        if not token.startswith("tbat_"):
            return None
        return await self.provider.load_access_token(token)


class GatewayOAuth:
    """Everything the web app needs to serve /mcp with per-person sign-in."""

    def __init__(self, cfg: OAuthConfig, provider: Optional[Provider] = None):
        self.cfg = cfg
        self.provider = provider or Provider(cfg)

    def verifier(self, static=None) -> GatewayTokenVerifier:
        return GatewayTokenVerifier(self.provider, self.cfg.mcp_url, static)

    def auth_settings(self):
        from mcp.server.auth.settings import AuthSettings
        return AuthSettings(
            issuer_url=self.cfg.issuer, resource_server_url=self.cfg.mcp_url,
            validate_token_resource=True)

    def routes(self) -> list:
        cfg, provider = self.cfg, self.provider
        # Through AuthSettings: it keeps a path-less issuer exactly as written
        # (no trailing slash), and clients compare issuers as strings.
        issuer = self.auth_settings().issuer_url
        registration = ClientRegistrationOptions(
            enabled=True, valid_scopes=list(SCOPES), default_scopes=["mcp"])
        revocation = RevocationOptions(enabled=True)
        sdk = create_auth_routes(
            provider=provider, issuer_url=issuer,
            client_registration_options=registration, revocation_options=revocation)
        ours = {"/.well-known/oauth-authorization-server", "/authorize", "/register",
                "/revoke"}
        routes = [r for r in sdk if r.path not in ours]

        base = build_metadata(issuer, None, registration, revocation).model_dump(mode="json")
        metadata = OAuthMetadata.model_validate({
            **base, "issuer": cfg.issuer, "scopes_supported": list(SCOPES),
            "token_endpoint_auth_methods_supported": ["none"],
            "revocation_endpoint_auth_methods_supported": ["none"],
            "client_id_metadata_document_supported": True,
            "code_challenge_methods_supported": ["S256"]})
        from mcp.server.auth.handlers.metadata import MetadataHandler
        routes.append(Route(
            "/.well-known/oauth-authorization-server",
            endpoint=cors_middleware(MetadataHandler(metadata).handle, ["GET", "OPTIONS"]),
            methods=["GET", "OPTIONS"]))

        resource = ProtectedResourceMetadata.model_validate({
            "resource": cfg.mcp_url, "authorization_servers": [cfg.issuer],
            "scopes_supported": list(SCOPES), "resource_name": "TraceBi"})
        handler = ProtectedResourceMetadataHandler(resource)
        path = urllib.parse.urlsplit(cfg.mcp_url).path
        for suffix in {"", path if path != "/" else ""}:
            routes.append(Route(
                f"/.well-known/oauth-protected-resource{suffix}",
                endpoint=cors_middleware(handler.handle, ["GET", "OPTIONS"]),
                methods=["GET", "OPTIONS"]))

        authorize = _authorize_view(provider, cfg.issuer.startswith("https://"))
        routes.append(Route("/authorize", endpoint=RequestBodyLimitMiddleware(
            request_response(authorize), DEFAULT_MAX_REQUEST_BODY_SIZE),
            methods=["GET", "POST"]))
        if cfg.local:
            routes.append(Route("/oauth/callback", endpoint=RequestBodyLimitMiddleware(
                request_response(provider.local_login), DEFAULT_MAX_REQUEST_BODY_SIZE),
                methods=["GET", "POST"]))
        else:
            routes.append(Route("/oauth/callback", endpoint=provider.callback, methods=["GET"]))
        routes.append(Route("/revoke", endpoint=CORSMiddleware(
            RequestBodyLimitMiddleware(request_response(_revoke_view(provider)),
                                       DEFAULT_MAX_REQUEST_BODY_SIZE),
            allow_origins="*", allow_methods=["POST", "OPTIONS"],
            allow_headers=["mcp-protocol-version"]), methods=["POST", "OPTIONS"]))
        routes.append(Route("/register", endpoint=CORSMiddleware(
            RequestBodyLimitMiddleware(request_response(_register_view(provider, registration)),
                                       DEFAULT_MAX_REQUEST_BODY_SIZE),
            allow_origins="*", allow_methods=["POST", "OPTIONS"],
            allow_headers=["mcp-protocol-version"]), methods=["POST", "OPTIONS"]))
        return routes


def _authorize_view(provider: Provider, secure: bool):
    from mcp.server.auth.handlers.authorize import AuthorizationHandler
    handler = AuthorizationHandler(provider)

    async def view(request: Request) -> Response:
        flow = {"id": secrets.token_urlsafe(9), "sealed": None}
        token = _flow.set(flow)
        try:
            resp = await handler.handle(request)
        finally:
            _flow.reset(token)
        if flow["sealed"]:
            # Ties the return trip to this browser: a sign-in link made by
            # someone else, opened by a victim, has no cookie to match.
            resp.set_cookie(f"tb_oauth_{flow['id']}", flow["sealed"], max_age=PENDING_TTL,
                            httponly=True, secure=secure, samesite="lax",
                            path="/oauth/callback")
        return resp
    return view


def _replayed(request: Request, body: bytes) -> Request:
    """*request* with its body replaced, for handing on to an SDK handler."""
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}
    return Request(request.scope, receive)


def _register_view(provider: Provider, options):
    handler = RegistrationHandler(provider, options)

    async def view(request: Request) -> Response:
        body = await request.body()
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        # The SDK defaults a silent client to a confidential one with a
        # secret; every client here is public and PKCE-protected.
        if isinstance(data, dict) and data.get("token_endpoint_auth_method") is None:
            data["token_endpoint_auth_method"] = "none"
            body = json.dumps(data).encode()
        return await handler.handle(_replayed(request, body))
    return view


def _revoke_view(provider: Provider):
    from mcp.server.auth.handlers.revoke import RevocationHandler
    from mcp.server.auth.middleware.client_auth import ClientAuthenticator
    handler = RevocationHandler(provider, ClientAuthenticator(provider))

    async def view(request: Request) -> Response:
        # The SDK's request model requires a client_secret field; a public
        # client has none to send.
        form = dict(await request.form())
        form.setdefault("client_secret", "")
        return await handler.handle(_replayed(request, urllib.parse.urlencode(form).encode()))
    return view
