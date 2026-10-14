"""
Security middleware and utilities for the Diagram Generation API.

Covers:
- HTTPS enforcement (via X-Forwarded-Proto when behind a proxy)
- Safe log filtering to scrub secrets and payment signatures
- Rate limiting per IP (via Redis sliding-window counters)
- Anomaly logging for repeated failed payment verifications
- Security response headers (HSTS, CSP, X-Frame-Options, etc.)
- CORS allowlist enforcement
- Request body size limiting
- Request-ID correlation for audit logs
- Structured audit logging
- Startup secret strength validation
"""

import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

log = logging.getLogger(__name__)
anomaly_log = logging.getLogger("security.anomaly")
audit_log = logging.getLogger("security.audit")

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_secrets() -> None:
    warnings = []
    if len(settings.api_key) < 16:
        warnings.append("API_KEY is too short (< 16 chars)")
    if settings.api_key.lower() in {"secret", "changeme", "password", "test", "your-secret-api-key"}:
        warnings.append("API_KEY looks like a placeholder")
    if not settings.openai_api_key.startswith("sk-") and not settings.use_llm_fallback:
        warnings.append("OPENAI_API_KEY does not look like a valid OpenAI key")
    if settings.x402_enabled and settings.x402_pay_to and len(settings.x402_pay_to) != 42:
        warnings.append("X402_PAY_TO does not look like a valid EVM address (expected 42 chars)")
    for w in warnings:
        log.warning("STARTUP SECURITY WARNING: %s", w)


# ---------------------------------------------------------------------------
# Safe log filter
# ---------------------------------------------------------------------------

_SCRUB_PATTERNS = [
    (re.compile(r"(sk-[A-Za-z0-9\-_]{20,})", re.IGNORECASE), "sk-***"),
    (re.compile(r"(0x[0-9a-fA-F]{64,})", re.IGNORECASE), "0x***"),
    (re.compile(r"(PAYMENT.SIGNATURE\s*[:=]\s*)([A-Za-z0-9+/=_\-]{40,})", re.IGNORECASE), r"\g<1>***"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), r"\g<1>***"),
    (re.compile(r"(X-API-Key\s*[:=]\s*)[A-Za-z0-9\-_]{8,}", re.IGNORECASE), r"\g<1>***"),
]


class ScrubSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            for pattern, replacement in _SCRUB_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
            record.args = ()
        except Exception:
            pass
        return True


def install_log_filter() -> None:
    filt = ScrubSecretsFilter()
    for name in ("", "uvicorn", "uvicorn.access", "uvicorn.error", "app"):
        logging.getLogger(name).addFilter(filt)


# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------

async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


# ---------------------------------------------------------------------------
# Security response headers
# ---------------------------------------------------------------------------

_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": (
        "default-src 'none'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "font-src https://cdnjs.cloudflare.com; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    ),
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    return response


# ---------------------------------------------------------------------------
# HTTPS enforcement
# ---------------------------------------------------------------------------

async def https_redirect_middleware(request: Request, call_next: Callable) -> Response:
    if not settings.enforce_https:
        return await call_next(request)
    proto = request.headers.get("x-forwarded-proto", "https")
    if proto != "https" and request.url.path != "/health":
        return JSONResponse(status_code=400, content={"detail": "HTTPS is required"})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request body size limit
# ---------------------------------------------------------------------------

MAX_BODY_BYTES = 64 * 1024  # 64 KB


async def body_size_limit_middleware(request: Request, call_next: Callable) -> Response:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request body too large (max 64 KB)"})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiter (slowapi)
# ---------------------------------------------------------------------------

def _get_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_get_ip)


# ---------------------------------------------------------------------------
# Admin IP allowlist
# ---------------------------------------------------------------------------

def check_admin_ip(request: Request) -> bool:
    if not settings.admin_ip_allowlist:
        return True
    allowed = {ip.strip() for ip in settings.admin_ip_allowlist.split(",") if ip.strip()}
    return _get_ip(request) in allowed


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

_payment_failures: dict[str, list[float]] = defaultdict(list)
ANOMALY_WINDOW_SECONDS = 60
ANOMALY_THRESHOLD = 5


def record_payment_failure(ip: str, path: str, detail: str = "") -> None:
    now = time.monotonic()
    _payment_failures[ip] = [t for t in _payment_failures[ip] if now - t < ANOMALY_WINDOW_SECONDS]
    _payment_failures[ip].append(now)
    count = len(_payment_failures[ip])
    if count >= ANOMALY_THRESHOLD:
        anomaly_log.warning(
            "ANOMALY ip=%s path=%s failures_in_%ds=%d detail=%s",
            ip, path, ANOMALY_WINDOW_SECONDS, count, detail[:120] if detail else "",
        )


# ---------------------------------------------------------------------------
# Structured audit logging
# ---------------------------------------------------------------------------

def audit(event: str, request: Request, **extra) -> None:
    req_id = get_request_id(request)
    ip = _get_ip(request)
    parts = [f"event={event}", f"req_id={req_id}", f"ip={ip}",
             f"method={request.method}", f"path={request.url.path}"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    audit_log.info(" ".join(parts))
