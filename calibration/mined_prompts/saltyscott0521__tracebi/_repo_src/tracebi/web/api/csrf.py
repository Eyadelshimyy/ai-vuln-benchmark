"""Same-origin CSRF guard for state-changing requests.

Basic-auth credentials are auto-attached by the browser, so a page on another
site could drive a cross-site POST to a state-changing endpoint — the code-exec
``requests`` run, or a warehouse-writing pipeline run — with the user's
credentials riding along. CORS does not stop this: a "simple" cross-site POST
is still *delivered* to the server; CORS only blocks the attacker from reading
the response.

The guard is deliberately simple. A browser always sends an ``Origin`` header
on a state-changing cross-site request, so:

* an unsafe request whose ``Origin`` is present and not allowed is refused;
* an unsafe request with **no** ``Origin`` (curl, the CLI, server-to-server) is
  not a browser CSRF, so it is allowed — this keeps non-browser clients working.

Allowed origins are the app's own (same host:port as the request) plus the
configured frontend origins, extendable with ``TRACEBI_ALLOWED_ORIGINS``.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

#: Frontend origins permitted to make state-changing requests (the same set the
#: CORS layer allows). Extend with ``TRACEBI_ALLOWED_ORIGINS`` (comma-separated).
_DEFAULT_ALLOWED = (
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",   # CRA / alternate dev port
    "http://localhost:8000",   # same-origin (prod static serving)
)


#: OAuth endpoints for the MCP gateway's connectors. They authenticate with
#: the request's own body (a PKCE code, a refresh token) and use no cookie or
#: Basic credential a browser would attach, so there is nothing for a
#: cross-site POST to ride on; a browser-based MCP client (an inspector) sends
#: an Origin here and must not be refused.
#: The built-in sign-in form posts to ``/oauth/callback``. The browser arrives
#: there from the AI app's site (a cross-origin redirect chain), so it sends
#: ``Origin: null``, which the same-origin rule would refuse. It does not need
#: the rule: the form is answered only for the browser that began the sign-in
#: (a sealed per-flow cookie) and a state checked against it, neither of which a
#: page on another site can supply.
_EXEMPT_PATHS = frozenset({"/token", "/register", "/revoke", "/oauth/callback"})


def allowed_origins() -> list[str]:
    """The configured allow-list: the defaults plus ``TRACEBI_ALLOWED_ORIGINS``."""
    extra = os.environ.get("TRACEBI_ALLOWED_ORIGINS", "")
    return list(_DEFAULT_ALLOWED) + [o.strip() for o in extra.split(",") if o.strip()]


def _origin_allowed(origin: str, request: Request) -> bool:
    if origin in allowed_origins():
        return True
    # Same-origin: the Origin's host:port equals the request's Host header.
    host = request.headers.get("host")
    return bool(host) and urlsplit(origin).netloc == host


class CSRFMiddleware(BaseHTTPMiddleware):
    """Refuse a state-changing request from a disallowed browser Origin."""

    async def dispatch(self, request: Request, call_next):
        if request.method not in _SAFE_METHODS and request.url.path not in _EXEMPT_PATHS:
            origin = request.headers.get("origin")
            if origin and not _origin_allowed(origin, request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": (
                        f"cross-origin request refused (CSRF): Origin {origin!r} "
                        "is not allowed. Set TRACEBI_ALLOWED_ORIGINS to permit an "
                        "additional frontend origin."
                    )},
                )
        return await call_next(request)
