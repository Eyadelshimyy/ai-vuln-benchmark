"""Redact secrets from logs, evidence, prompts, and exports."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from app.domain.http import HttpExchange, HttpHeader

_REDACTED = "[REDACTED]"
_SECRET_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
    "x-access-token",
    "x-csrf-token",
    "x-session-token",
    "api-key",
    "apikey",
}
_PATTERNS = (
    re.compile(r"(?i)bearer\s+[a-z0-9._\-+=/]+"),
    re.compile(r"(?i)basic\s+[a-z0-9=+/]+"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|authorization)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(access_token|refresh_token|id_token)=([^&\s]+)"),
    re.compile(r"(?i)(?:cookie|set-cookie)\s*[:=]\s*\S+"),
)
_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9._-]{10,}\.[A-Za-z0-9._-]{10,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("github_fine_grained", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("password_assignment", re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+")),
    ("session_cookie", re.compile(r"(?i)(sessionid|connect\.sid|auth_token)=([^\s;]+)")),
    ("high_entropy_hex", re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{40,}(?![A-Fa-f0-9])")),
)


def redact_text(value: str | None) -> str:
    if not value:
        return ""
    redacted = value
    for pattern in _PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    for _label, pattern in _CREDENTIAL_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def detect_secrets(
    *parts: str,
    configured: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return labels for credential-like material. Used to *block* submission."""
    blob = "\n".join(part for part in parts if part)
    if not blob:
        return ()
    hits: list[str] = []
    lowered = blob.lower()
    for hint in (
        "authorization:",
        "bearer ",
        "hackerone_api_token",
        "bugforge_operator_token",
    ):
        if hint in lowered:
            hits.append(hint.strip(" :"))
    for label, pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(blob):
            hits.append(label)
    for secret in configured:
        if secret and len(secret) >= 8 and secret in blob:
            hits.append("configured_secret")
    return tuple(dict.fromkeys(hits))


def configured_secret_values() -> tuple[str, ...]:
    """Known local secrets that must never appear in outbound reports."""
    names = (
        "HACKERONE_API_TOKEN",
        "HACKERONE_API_USERNAME",
        "BUGFORGE_OPERATOR_TOKEN",
        "GITHUB_TOKEN",
        "AI_API_KEY",
        "SECRET_KEY",
    )
    values = [environment_credential(name) for name in names]
    return tuple(item for item in values if item and len(item) >= 8)


def redact_headers(headers: Sequence[HttpHeader]) -> tuple[HttpHeader, ...]:
    cleaned: list[HttpHeader] = []
    for header in headers:
        if header.name.lower() in _SECRET_HEADERS:
            cleaned.append(HttpHeader(name=header.name, value=_REDACTED))
        else:
            cleaned.append(HttpHeader(name=header.name, value=redact_text(header.value)))
    return tuple(cleaned)


def redact_mapping(values: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in values.items():
        if key.lower() in _SECRET_HEADERS or "token" in key.lower() or "secret" in key.lower():
            out[key] = _REDACTED
        else:
            out[key] = redact_text(value)
    return out


def redact_exchange(exchange: HttpExchange) -> HttpExchange:
    return HttpExchange(
        method=exchange.method,
        url=exchange.url,
        request_id=exchange.request_id,
        timestamp=exchange.timestamp,
        scheme=exchange.scheme,
        host=exchange.host,
        port=exchange.port,
        path=exchange.path,
        request_headers=redact_headers(exchange.request_headers),
        request_body=redact_text(exchange.request_body) or None,
        request_body_meta=exchange.request_body_meta,
        query=exchange.query,
        cookies=tuple(_REDACTED for _ in exchange.cookies) if exchange.cookies else (),
        response_status=exchange.response_status,
        response_headers=redact_headers(exchange.response_headers),
        response_body=redact_text(exchange.response_body) or None,
        response_body_meta=exchange.response_body_meta,
        elapsed_ms=exchange.elapsed_ms,
        source_tool=exchange.source_tool,
        scope_decision=exchange.scope_decision,
        metadata=exchange.metadata,
    )


def environment_credential(name: str, *, required: bool = False) -> str:
    """Read a secret from the environment. Never log the value."""
    import os

    value = os.environ.get(name, "")
    if required and not value:
        raise RuntimeError(f"Missing required credential environment variable {name}")
    return value
