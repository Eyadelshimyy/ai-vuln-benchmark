"""
Optional auth for the TraceBi web layer.

Three modes, mutually exclusive (the third, OIDC sign-in, lives in ``sso.py``):

* **Basic auth** — enabled when ``TRACEBI_AUTH_USER`` and ``TRACEBI_AUTH_PASS``
  are both set. Single shared user/password, suitable for small-team deployments.

* **Proxy header trust** — enabled when ``TRACEBI_AUTH_PROXY_HEADER`` is set.
  Designed for sitting behind Authelia / oauth2-proxy / Cloudflare Access,
  which authenticate the user upstream and forward the identity in a header
  (default ``X-Forwarded-User``). The header value is exposed at
  ``request.state.user`` for downstream handlers. Optionally set
  ``TRACEBI_AUTH_PROXY_TRUSTED_IPS`` (comma-separated) to require the
  request to originate from a known proxy address. The proxy **must replace
  any client-supplied role header** (``TRACEBI_AUTH_ROLE_HEADER``), which is
  trusted in this mode precisely because the proxy sets it. Appending is
  tolerated — the last occurrence is the one read — but a proxy that neither
  strips nor sets it hands the role back to the caller.

Both modes exempt ``/api/health`` so liveness probes work without credentials.
For multi-user identity, use the proxy mode.
"""

from __future__ import annotations

import base64
import ipaddress
import logging
import os
import secrets
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from tracebi.audit import actor as audit_actor

_logger = logging.getLogger("tracebi.auth")


_PROTECTED_PREFIXES = ("/api/", "/dashboards/", "/r/")
_EXEMPT_PATHS = ("/api/health",)


# ── Authorization ───────────────────────────────────────────────────────────
#
# Authentication answers "who is this"; until now nothing asked "may they".
# Both middlewares set request.state.user and no code read it, so any
# authenticated principal could run every report, execute every request
# script, and trigger every pipeline layer.
#
# Three roles, ordered. The split is by *side effect*, which is what a
# security review asks about and what a reader can predict without a table:
#
#   viewer   reads. Browsing models, reports, lineage, and running Explore
#            queries — analytical reads that write nothing.
#   analyst  viewer, plus executing report and request code.
#   admin    analyst, plus running pipeline layers, which write to the
#            warehouse, dev-mode reload, and keeping a selection cut,
#            which writes the published report.json.

ROLES = ("viewer", "analyst", "admin")
_ROLE_RANK = {role: i for i, role in enumerate(ROLES)}

#: Role assumed for an authenticated principal with no role of their own,
#: once any role configuration exists. Overridable per deployment.
_DEFAULT_ROLE = "viewer"


def _required_role(method: str, path: str) -> str:
    """
    Minimum role for *method* *path*.

    Unlisted writes require `analyst`, so a route added later is guarded by
    default rather than open by default.
    """
    if method in ("GET", "HEAD", "OPTIONS"):
        # A run's log is whatever its steps printed, so reading it is more than
        # reading a status.
        if path.startswith("/api/pipelines/") and path.endswith("/log"):
            return "analyst"
        return "viewer"
    if path.startswith("/api/pipelines/") or path.startswith("/api/_dev/"):
        return "admin"
    # Keeping a selection cut writes the published report.json. Foldered
    # report names contain slashes, so match the suffix, not one segment.
    if path.startswith("/api/reports/") and path.endswith("/selection/keep"):
        return "admin"
    # Explore and spec validation compute but persist nothing, so they sit
    # with the reads rather than behind an execute permission.
    if path.startswith("/api/models/") and path.endswith("/query"):
        return "viewer"
    if path.startswith("/api/spec/validate"):
        return "viewer"
    return "analyst"


def _rank(role: Optional[str]) -> int:
    return _ROLE_RANK.get(role or "", -1)


def _highest_role(values) -> Optional[str]:
    """Best known role among *values*; unknown names are ignored, not fatal."""
    known = [v for v in values if v in _ROLE_RANK]
    return max(known, key=_rank) if known else None


class _Authorizer:
    """
    Resolves a principal's role and decides whether a request is permitted.

    Role comes from a header claim forwarded by the SSO proxy (the usual
    enterprise shape — the org's IdP already knows the groups) or from an
    explicit user→role map for the Basic-auth case.

    **The role header is only read when the caller can attribute it to an
    upstream.** *trust_role_header* says so, and each middleware must decide:
    proxy mode has an upstream that sets the header, Basic auth does not, so
    under Basic a role claim in a request header is self-asserted — the
    authenticated principal would be choosing their own privileges. There is
    no default; a middleware added later has to answer the question.

    **Enforcement is opt-in, deliberately.** With no *usable* source
    configured every principal is treated as `admin`, so adding this cannot
    silently lock existing deployments out of their own pipelines — the one
    case being a Basic deployment whose only role config is the header this
    now ignores *and* which never named the role everyone should get.
    Configure a usable source and enforcement switches on for everyone, with
    unmatched principals falling back to TRACEBI_AUTH_DEFAULT_ROLE
    (`viewer`).
    """

    def __init__(self, *, trust_role_header: bool) -> None:
        self.trust_role_header = trust_role_header
        self.role_header = os.environ.get("TRACEBI_AUTH_ROLE_HEADER", "").strip()
        _raw_map = os.environ.get("TRACEBI_AUTH_ROLE_MAP", "")
        self.role_map = self._parse_map(_raw_map)
        if _raw_map.strip() and not self.role_map:
            # An operator who set the variable meant to enforce roles; a map
            # that parses to nothing (wrong separator, unknown role names)
            # would silently leave every caller admin. Always a
            # misconfiguration — say so.
            import warnings
            warnings.warn(
                f"TRACEBI_AUTH_ROLE_MAP is set but no entry parsed "
                f"({_raw_map!r}). Entries are comma-separated user:role "
                f"pairs (e.g. 'alice:admin,bob:analyst') with roles from "
                f"{ROLES}. With an empty map there is no role source, so "
                f"role enforcement is OFF and every authenticated caller "
                f"is admin.",
                stacklevel=2,
            )
        stated_default = os.environ.get("TRACEBI_AUTH_DEFAULT_ROLE", "").strip()
        #: True when the operator named the fallback role rather than
        #: inheriting it. Not a role source on its own — it only keeps
        #: enforcement on beside an untrusted role header (see `enabled`).
        self.explicit_default_role = bool(stated_default)
        self.default_role = stated_default or _DEFAULT_ROLE
        if self.default_role not in _ROLE_RANK:
            raise ValueError(
                f"TRACEBI_AUTH_DEFAULT_ROLE must be one of {ROLES}, "
                f"got {self.default_role!r}"
            )

    @property
    def enabled(self) -> bool:
        if self.role_map:
            return True
        if not self.role_header:
            return False
        # A role header this mode may not trust is not a role source. It is
        # still the operator asking for roles, so enforcement stays on when
        # they also named the role everyone gets — TRACEBI_AUTH_DEFAULT_ROLE
        # applies to every principal the (absent) map does not list. Only
        # when no role was stated anywhere would switching on mean demoting
        # a running deployment to the implicit `viewer`, out of its own
        # pipelines; there, enforcement stays off.
        return self.trust_role_header or self.explicit_default_role

    @staticmethod
    def _parse_map(raw: str) -> dict:
        """Parse ``alice:admin,bob:analyst`` into a dict, ignoring blanks."""
        import warnings
        out = {}
        for pair in raw.split(","):
            entry = pair.strip()
            if not entry:
                continue
            if ":" not in pair:
                warnings.warn(
                    f"TRACEBI_AUTH_ROLE_MAP entry {entry!r} was dropped: no ':'.",
                    stacklevel=2,
                )
                continue
            user, _, role = pair.partition(":")
            user, role = user.strip(), role.strip()
            if user and role in _ROLE_RANK:
                out[user] = role
                continue
            if role not in _ROLE_RANK:
                warnings.warn(
                    f"TRACEBI_AUTH_ROLE_MAP entry {entry!r} was dropped: "
                    f"unknown role {role!r}.",
                    stacklevel=2,
                )
                continue
            # A valid role with nobody to give it to is still a dropped
            # entry. Name it, the same as the other drops.
            warnings.warn(
                f"TRACEBI_AUTH_ROLE_MAP entry {entry!r} was dropped: "
                f"no user name.",
                stacklevel=2,
            )
        return out

    def role_for(self, request: Request, user: Optional[str]) -> str:
        if not self.enabled:
            # No authorization configured: preserve pre-existing behaviour
            # rather than breaking a running deployment.
            return "admin"
        if self.role_header and self.trust_role_header:
            # Last occurrence, not first: a proxy that *appends* its claim
            # leaves a client-supplied copy ahead of it, and reading the
            # first would hand the role back to the caller.
            values = request.headers.getlist(self.role_header)
            raw = values[-1] if values else ""
            # Group claims are conventionally comma- or space-separated.
            claimed = _highest_role(raw.replace(" ", ",").split(","))
            if claimed:
                return claimed
        if user and user in self.role_map:
            return self.role_map[user]
        return self.default_role

    def check(self, request: Request, user: Optional[str],
              role: Optional[str] = None) -> Optional[Response]:
        """None when permitted, otherwise the 403 to return.

        *role* is one the identity provider already decided (app sign-in, see
        ``sso.py``): it is enforced as given, whatever the env-configured
        sources say, and a name outside ROLES gets nothing."""
        if role is None:
            role = self.role_for(request, user)
        elif role not in _ROLE_RANK:
            role = "viewer"
        request.state.role = role
        needed = _required_role(request.method, request.url.path)
        if _rank(role) >= _rank(needed):
            return None
        return JSONResponse(
            status_code=403,
            content={
                "detail": {
                    "message": (
                        f"Role '{role}' may not {request.method} {request.url.path}. "
                        f"Requires '{needed}' or higher."
                    ),
                    "role": role,
                    "required_role": needed,
                }
            },
        )


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Single-credential HTTP Basic auth guarding the API and dashboards."""

    def __init__(self, app, username: str, password: str, realm: str = "TraceBi") -> None:
        super().__init__(app)
        self._username = username
        self._password = password
        self._realm = realm
        # No upstream proxy here, so a role header on the request is written
        # by the caller themselves.
        self._authz = _Authorizer(trust_role_header=False)
        _log_posture("basic", self._authz)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(_PROTECTED_PREFIXES):
            return await call_next(request)
        if path in _EXEMPT_PATHS:
            return await call_next(request)
        if not self._check(request.headers.get("authorization")):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": f'Basic realm="{self._realm}"'},
            )
        request.state.user = self._username
        denied = self._authz.check(request, self._username)
        if denied is not None:
            return denied
        # Carry identity down to whatever this request ends up executing, so
        # a pipeline run records who asked for it.
        with audit_actor(self._username, getattr(request.state, "role", None)):
            return await call_next(request)

    def _check(self, header: Optional[str]) -> bool:
        if not header or not header.lower().startswith("basic "):
            return False
        try:
            decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        except Exception:
            return False
        user, _, password = decoded.partition(":")
        return (
            secrets.compare_digest(user, self._username)
            and secrets.compare_digest(password, self._password)
        )


class ProxyHeaderAuthMiddleware(BaseHTTPMiddleware):
    """
    Trust an upstream proxy's identity header.

    The auth header value is exposed at ``request.state.user`` and the raw
    name at ``request.state.user_header``. When ``trusted_ips`` is set,
    requests from any other address are rejected with 401.
    """

    def __init__(
        self,
        app,
        header: str = "X-Forwarded-User",
        trusted_ips: Optional[list[str]] = None,
    ) -> None:
        super().__init__(app)
        self._header = header
        self._trusted_networks = [
            ipaddress.ip_network(ip.strip(), strict=False)
            for ip in (trusted_ips or [])
            if ip.strip()
        ]
        # The upstream proxy is the identity source, so it is also the role
        # source — it must replace any client-supplied role header.
        self._authz = _Authorizer(trust_role_header=True)
        _log_posture("proxy", self._authz)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(_PROTECTED_PREFIXES):
            return await call_next(request)
        if path in _EXEMPT_PATHS:
            return await call_next(request)

        if self._trusted_networks and not self._is_trusted_client(request):
            return Response(status_code=401, content="proxy not in trusted_ips")

        user = request.headers.get(self._header)
        if not user:
            return Response(
                status_code=401,
                content=f"missing identity header {self._header}",
            )
        request.state.user = user
        request.state.user_header = self._header
        denied = self._authz.check(request, user)
        if denied is not None:
            return denied
        with audit_actor(user, getattr(request.state, "role", None)):
            return await call_next(request)

    def _is_trusted_client(self, request: Request) -> bool:
        host = request.client.host if request.client else None
        if not host:
            return False
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            return False
        return any(addr in net for net in self._trusted_networks)


def _posture_text(mode: str, authz: _Authorizer) -> str:
    """One line an operator can read. Never includes a password or token."""
    if authz.role_map:
        source = f"role map ({len(authz.role_map)} users)"
    elif authz.role_header and authz.trust_role_header:
        source = f"role header {authz.role_header}"
    elif authz.explicit_default_role and authz.enabled:
        source = f"default role {authz.default_role}"
    else:
        source = "none"
    if mode == "oidc":
        return ("auth posture: OIDC; role source: identity provider groups "
                "(TRACEBI_OIDC_ROLE_MAP, TRACEBI_OIDC_DEFAULT_ROLE); enforcement on")
    if mode == "off":
        # install_if_configured installs no middleware in this mode, so a
        # role map or header cannot be enforced. Naming the source above
        # still shows the operator that it is set and being ignored.
        enforcement = (
            "enforcement is off: no authentication configured, "
            "every request is admin"
        )
    elif authz.enabled:
        enforcement = "enforcement on"
    else:
        enforcement = "enforcement is off: every principal is admin"
    label = {"off": "off", "basic": "Basic", "proxy": "proxy", "oidc": "OIDC"}[mode]
    return f"auth posture: {label}; role source: {source}; {enforcement}"


def _log_posture(mode: str, authz: _Authorizer) -> None:
    _logger.info(_posture_text(mode, authz))


def posture_line() -> str:
    """The startup posture line, read again without installing middleware.

    Same words ``_log_posture`` writes. No password, token, or header value.
    """
    if os.environ.get("TRACEBI_OIDC_ISSUER", "").strip():
        mode, trust = "oidc", False
    elif os.environ.get("TRACEBI_AUTH_PROXY_HEADER"):
        mode, trust = "proxy", True
    elif os.environ.get("TRACEBI_AUTH_USER") and os.environ.get("TRACEBI_AUTH_PASS"):
        mode, trust = "basic", False
    else:
        mode, trust = "off", False
    return _posture_text(mode, _Authorizer(trust_role_header=trust))


def install_if_configured(app) -> Optional[str]:
    """
    Install the appropriate auth middleware on *app* based on env vars.

    Returns the mode that was installed (``"basic"`` / ``"proxy"``) or
    ``None`` when no auth is configured.
    """
    user = os.environ.get("TRACEBI_AUTH_USER")
    pw = os.environ.get("TRACEBI_AUTH_PASS")
    proxy_header = os.environ.get("TRACEBI_AUTH_PROXY_HEADER")

    if os.environ.get("TRACEBI_OIDC_ISSUER", "").strip():
        # App sign-in (sso.py) is the one way in. Two ways to say who a person
        # is, on one app, would let the weaker one decide: refuse to start.
        clash = [name for name, on in (("TRACEBI_AUTH_USER/TRACEBI_AUTH_PASS", user and pw),
                                       ("TRACEBI_AUTH_PROXY_HEADER", proxy_header)) if on]
        if clash:
            raise ValueError(
                "TRACEBI_OIDC_ISSUER turns on the app's own sign-in, which cannot run "
                f"beside {' / '.join(clash)}. Unset one: either people sign in through "
                "the identity provider, or the app keeps its Basic / proxy login.")
        return "oidc"

    if proxy_header:
        trusted = os.environ.get("TRACEBI_AUTH_PROXY_TRUSTED_IPS", "")
        trusted_list = [ip for ip in trusted.split(",") if ip.strip()]
        if not trusted_list:
            import warnings
            warnings.warn(
                "TRACEBI_AUTH_PROXY_HEADER is set without "
                "TRACEBI_AUTH_PROXY_TRUSTED_IPS. Any client that can reach "
                f"this server directly can authenticate by sending the "
                f"'{proxy_header}' header itself. Set "
                "TRACEBI_AUTH_PROXY_TRUSTED_IPS to your proxy's address(es) "
                "unless the server is only reachable through the proxy.",
                stacklevel=2,
            )
        app.add_middleware(
            ProxyHeaderAuthMiddleware,
            header=proxy_header,
            trusted_ips=trusted_list,
        )
        return "proxy"

    if user and pw:
        role_header = os.environ.get("TRACEBI_AUTH_ROLE_HEADER", "").strip()
        if role_header:
            import warnings
            warnings.warn(
                f"TRACEBI_AUTH_ROLE_HEADER ('{role_header}') is ignored under "
                "Basic auth: there is no upstream proxy to set it, so the "
                "header would be self-asserted by the authenticated caller. "
                "Roles come from TRACEBI_AUTH_ROLE_MAP, and from "
                "TRACEBI_AUTH_DEFAULT_ROLE for principals it does not list. "
                "With neither set there is no usable role source, so no role "
                "is enforced at all and every authenticated caller is admin "
                "— including on the pipeline routes that write to the "
                "warehouse.",
                stacklevel=2,
            )
        app.add_middleware(
            BasicAuthMiddleware,
            username=user,
            password=pw,
            realm=os.environ.get("TRACEBI_AUTH_REALM", "TraceBi"),
        )
        return "basic"

    if os.environ.get("TRACEBI_AUTH_ROLE_HEADER", "").strip() or os.environ.get(
        "TRACEBI_AUTH_ROLE_MAP", ""
    ).strip():
        import warnings
        warnings.warn(
            "TRACEBI_AUTH_ROLE_HEADER / TRACEBI_AUTH_ROLE_MAP are set but no "
            "authentication mode is configured, so no middleware is installed "
            "and no role is enforced — including on the pipeline routes that "
            "write to the warehouse. Set TRACEBI_AUTH_USER/TRACEBI_AUTH_PASS "
            "or TRACEBI_AUTH_PROXY_HEADER.",
            stacklevel=2,
        )

    _log_posture("off", _Authorizer(trust_role_header=False))
    return None
