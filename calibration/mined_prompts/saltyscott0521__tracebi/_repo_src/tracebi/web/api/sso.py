"""
The web app's own sign-in: one login per person, through the company's OIDC
provider, with the same configuration the MCP gateway already uses.

On when ``TRACEBI_OIDC_ISSUER`` is set (see ``tracebi.mcp_oauth``). TraceBi is
an ordinary OIDC client of the provider here too (``_OIDC`` does the code
exchange and every ID-token check; nothing is repeated). A person who signs in
gets a server-side session: a random id in an ``HttpOnly`` cookie, of which
only the SHA-256 is stored (``tracebi_sessions``). The role comes from the
provider's groups (``TRACEBI_OIDC_ROLE_MAP``) and is enforced on every request
by the same ``_Authorizer`` the other auth modes use.

Routes (outside the protected prefixes, so they answer a stranger):

    GET  /login?next=/path   -> the provider (state + nonce + PKCE, bound to
                                this browser by a short-lived cookie)
    GET  /login/callback     -> checks all of that, opens the session, redirects
                                to *next* (a same-origin path, nothing else)
    POST /logout             -> ends the session

The static app shell (``/``, ``/assets``) holds no data and stays public so it
can load; a browser *navigation* to it without a session is redirected to
``/login`` instead, and every API call without one is a 401 JSON. A TraceBi
access token (``Authorization: Bearer tbat_...``, the kind ``tracebi login``
and the MCP connectors hold) is accepted on the protected prefixes and acts as
the person it was issued to, with their role.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
import urllib.parse
from typing import Optional

from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

from tracebi import state
from tracebi.audit import actor as audit_actor
from tracebi.mcp_oauth import seal, unseal
from tracebi.web.api.auth import _EXEMPT_PATHS, _PROTECTED_PREFIXES, _Authorizer

log = logging.getLogger("tracebi.sso")

COOKIE = "tracebi_session"
IDLE_TTL = 12 * 3600            # a session unused this long ends ...
ABSOLUTE_TTL = 7 * 86400        # ... and none outlives a week
_TOUCH_EVERY = 300              # extend the idle deadline at most this often
_LOGIN_TTL = 600                # a sign-in must come back within ten minutes
_FLOW = re.compile(r"[A-Za-z0-9_-]{8,32}")
_NEXT = re.compile(r"/(?![/\\])[^\x00-\x20\x7f\\]*")

#: Paths a stranger's browser must be able to reach: the sign-in itself, the
#: bundle that renders the app, and the endpoints other clients use.
_OPEN_PREFIXES = ("/login", "/logout", "/assets/", "/.well-known/", "/mcp", "/authorize",
                  "/token", "/register", "/revoke", "/oauth/", "/docs", "/redoc",
                  "/openapi.json")


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_next(value) -> str:
    """*value* when it is a same-origin path, else ``/``.

    Refuses ``//host`` and ``/\\host`` (a browser reads both as another site),
    absolute URLs, control characters and spaces.
    """
    if not isinstance(value, str) or len(value) > 2048 or not _NEXT.fullmatch(value):
        return "/"
    parts = urllib.parse.urlsplit(value)
    if parts.scheme or parts.netloc:
        return "/"
    if parts.path in ("/login", "/logout") or parts.path.startswith(("/login/", "/logout/")):
        return "/"
    return value


# ── The session store ────────────────────────────────────────────────────

class AppSSO:
    """Sign-in routes and the session store, over a ``GatewayOAuth``."""

    def __init__(self, oauth):
        self.cfg = oauth.cfg
        self.provider = oauth.provider
        self.oidc = oauth.provider.oidc
        self.secure_cookies = self.cfg.issuer.startswith("https://")

    # -- store ----------------------------------------------------------
    @staticmethod
    def _exec(sql: str, **p) -> int:
        with state.ensure().begin() as conn:
            return conn.execute(text(sql), p).rowcount

    @staticmethod
    def _row(sql: str, **p) -> Optional[dict]:
        with state.ensure().connect() as conn:
            row = conn.execute(text(sql), p).mappings().first()
        return dict(row) if row else None

    def create_session(self, who: dict) -> str:
        """Open a session for *who* (sub/actor/role); returns the raw id."""
        raw = secrets.token_urlsafe(32)
        now = int(time.time())
        self._exec("DELETE FROM tracebi_sessions WHERE absolute_expires_at < :n",
                   n=now - 86400)
        self._exec(
            "INSERT INTO tracebi_sessions (session_hash, sub, actor, role, created_at, "
            "expires_at, absolute_expires_at, revoked_at) VALUES "
            "(:h, :s, :a, :r, :c, :e, :x, NULL)",
            h=_h(raw), s=who["sub"], a=who["actor"], r=who["role"], c=now,
            e=now + IDLE_TTL, x=now + ABSOLUTE_TTL)
        return raw

    def load_session(self, raw: str) -> Optional[dict]:
        if not raw or len(raw) > 200:
            return None
        row = self._row("SELECT * FROM tracebi_sessions WHERE session_hash = :h", h=_h(raw))
        now = time.time()
        if (row is None or row["revoked_at"] is not None
                or row["expires_at"] < now or row["absolute_expires_at"] < now):
            return None
        if row["expires_at"] - now < IDLE_TTL - _TOUCH_EVERY:
            self._exec("UPDATE tracebi_sessions SET expires_at = :e WHERE session_hash = :h",
                       e=int(min(now + IDLE_TTL, row["absolute_expires_at"])), h=_h(raw))
        return {"sub": row["sub"], "actor": row["actor"], "role": row["role"]}

    def end_session(self, raw: str) -> None:
        if raw:
            self._exec("UPDATE tracebi_sessions SET revoked_at = :n "
                       "WHERE session_hash = :h AND revoked_at IS NULL",
                       n=int(time.time()), h=_h(raw))

    # -- who is asking --------------------------------------------------
    async def identify(self, request: Request) -> Optional[dict]:
        """The person behind *request*: a TraceBi access token, else a session
        cookie. A bearer token that does not check out is a refusal, never a
        fall-through to the cookie."""
        scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
        if scheme.lower() == "bearer":
            token = token.strip()
            if not token.startswith("tbat_"):
                return None
            found = await self.provider.load_access_token(token)
            if found is None:
                return None
            return {"actor": found.claims["tracebi_actor"],
                    "role": found.claims["tracebi_role"], "via": "bearer"}
        raw = request.cookies.get(COOKIE)
        who = await asyncio.to_thread(self.load_session, raw) if raw else None
        return {**who, "via": "cookie"} if who else None

    # -- routes ---------------------------------------------------------
    def routes(self) -> list:
        return [Route("/login", self.login, methods=["GET"]),
                Route("/login/callback", self.callback, methods=["GET"]),
                Route("/logout", self.logout, methods=["POST"])]

    def _login_key(self) -> bytes:
        """Signs the in-flight sign-in cookie. Derived from the identity
        provider client secret, which every worker shares and nobody else has."""
        return hashlib.sha256(b"tracebi-login:" + self.cfg.client_secret.encode()).digest()

    def _seal(self, payload: dict) -> str:
        return seal(self._login_key(), payload)

    def _open(self, sealed: str) -> Optional[dict]:
        return unseal(self._login_key(), sealed)

    async def login(self, request: Request) -> Response:
        """Send the browser to the identity provider. What the return trip needs
        (nonce, PKCE verifier, where to land) travels in a signed cookie in this
        browser, so a stranger's /login writes nothing on the server."""
        flow_id = secrets.token_urlsafe(9)
        nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        state = f"{flow_id}.{secrets.token_urlsafe(32)}"
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        nxt = safe_next(request.query_params.get("next"))
        try:
            url = await asyncio.to_thread(
                self.oidc.authorize_url, state, nonce, challenge,
                redirect_uri=self.cfg.login_callback_url)
        except Exception as exc:  # noqa: BLE001 — provider down: say so, leak nothing
            log.error("cannot reach the identity provider: %s", exc)
            return _page("The identity provider is unreachable. Try again shortly.", 502)
        sealed = self._seal({"s": _h(state), "n": nonce, "v": verifier, "x": nxt,
                             "e": int(time.time()) + _LOGIN_TTL})
        resp = RedirectResponse(url, status_code=302, headers={"Cache-Control": "no-store"})
        # Ties the return trip to this browser: a sign-in link made by someone
        # else and opened by a victim has no matching cookie.
        resp.set_cookie(f"tb_login_{flow_id}", sealed, max_age=_LOGIN_TTL, httponly=True,
                        secure=self.secure_cookies or request.url.scheme == "https",
                        samesite="lax", path="/login/callback")
        return resp

    async def callback(self, request: Request) -> Response:
        q = request.query_params
        state = q.get("state") or ""
        flow_id = state.partition(".")[0]
        if not _FLOW.fullmatch(flow_id):
            return _page("This sign-in link is not valid. Start again.", 400)
        cookie_name = f"tb_login_{flow_id}"
        sealed = request.cookies.get(cookie_name)
        if not sealed:
            log.warning("sign-in refused: callback came from a browser that did not begin it")
            return _page("This sign-in was started in a different browser. Start again.", 400)
        pending = self._open(sealed)
        if (pending is None or not hmac.compare_digest(str(pending.get("s", "")), _h(state))
                or int(pending.get("e", 0)) < time.time()):
            return _page("This sign-in link is not valid or has expired. Start again.", 400)
        code = q.get("code")
        if q.get("error") or not code:
            return _page("Sign-in was not completed.", 403)
        try:
            claims = await asyncio.to_thread(
                self.oidc.sign_in, code, pending["v"], pending["n"],
                self.cfg.login_callback_url)
            who = self.oidc.identity(claims)
        except Exception as exc:  # noqa: BLE001 — whatever went wrong, the answer is no
            log.warning("sign-in refused: %s", exc)
            return _page("Sign-in could not be verified.", 403)
        raw = await asyncio.to_thread(self.create_session, who)
        resp = RedirectResponse(safe_next(pending.get("x")),
                                status_code=302, headers={"Cache-Control": "no-store"})
        resp.delete_cookie(cookie_name, path="/login/callback")
        self._set_session_cookie(resp, raw, request)
        return resp

    async def logout(self, request: Request) -> Response:
        await asyncio.to_thread(self.end_session, request.cookies.get(COOKIE) or "")
        resp = JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})
        resp.delete_cookie(COOKIE, path="/", httponly=True, samesite="lax",
                           secure=self.secure_cookies or request.url.scheme == "https")
        return resp

    def _set_session_cookie(self, resp: Response, raw: str, request: Request) -> None:
        resp.set_cookie(COOKIE, raw, max_age=ABSOLUTE_TTL, httponly=True,
                        secure=self.secure_cookies or request.url.scheme == "https",
                        samesite="lax", path="/")


def _page(message: str, status: int) -> Response:
    return PlainTextResponse(f"{message}\n\nSign in again: /login\n", status_code=status,
                             headers={"Cache-Control": "no-store"})


# ── Which SSO is on ──────────────────────────────────────────────────────

_active: Optional[AppSSO] = None


def active() -> Optional[AppSSO]:
    return _active


def enable(sso: Optional[AppSSO]) -> None:
    global _active
    _active = sso


# ── The middleware ───────────────────────────────────────────────────────

def _accepts_html(request: Request) -> bool:
    return "text/html" in ",".join(request.headers.getlist("accept")).lower()


def _is_navigation(request: Request) -> bool:
    return request.method in ("GET", "HEAD") and _accepts_html(request)


def _is_shell_page(path: str) -> bool:
    """A page of the app shell: not an endpoint, not a file."""
    if path == "/":
        return True
    return not path.startswith(_OPEN_PREFIXES) and "." not in path.rsplit("/", 1)[-1]


class SessionAuthMiddleware(BaseHTTPMiddleware):
    """Authenticates by session cookie or TraceBi access token when app SSO is
    on; does nothing otherwise (the Basic and proxy middlewares are separate,
    and the three are never installed together)."""

    def __init__(self, app) -> None:
        super().__init__(app)
        # Not a trusted role header: the identity provider's groups decide the
        # role, and the principal never gets to state their own.
        self._authz = _Authorizer(trust_role_header=False)

    async def dispatch(self, request: Request, call_next):
        sso = _active
        if sso is None:
            return await call_next(request)
        path = request.url.path
        protected = path.startswith(_PROTECTED_PREFIXES) and path not in _EXEMPT_PATHS
        shell = not protected and _is_navigation(request) and _is_shell_page(path)
        if not (protected or shell):
            return await call_next(request)

        who = await sso.identify(request)
        if who is None:
            return self._unauthenticated(request)
        if shell:
            return await call_next(request)
        if (who["via"] == "cookie" and request.method not in ("GET", "HEAD", "OPTIONS")
                and request.headers.get("sec-fetch-site", "same-origin")
                not in ("same-origin", "none")):
            # A cookie rides along on cross-site requests; a token does not.
            return JSONResponse(status_code=403, content={"detail": {
                "message": "cross-site request refused"}})
        request.state.user = who["actor"]
        request.state.sign_in = "oidc"
        denied = self._authz.check(request, who["actor"], role=who["role"])
        if denied is not None:
            return denied
        with audit_actor(who["actor"], who["role"]):
            return await call_next(request)

    @staticmethod
    def _unauthenticated(request: Request) -> Response:
        stale = COOKIE in request.cookies
        if _is_navigation(request) and not request.url.path.startswith("/api/"):
            target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
            resp: Response = RedirectResponse(
                "/login?" + urllib.parse.urlencode({"next": safe_next(target)}),
                status_code=302, headers={"Cache-Control": "no-store"})
        else:
            resp = JSONResponse(
                status_code=401, headers={"WWW-Authenticate": 'Bearer realm="TraceBi"'},
                content={"detail": {"message": "Sign in required.", "login": "/login"}})
        if stale:
            resp.delete_cookie(COOKIE, path="/")
        return resp
