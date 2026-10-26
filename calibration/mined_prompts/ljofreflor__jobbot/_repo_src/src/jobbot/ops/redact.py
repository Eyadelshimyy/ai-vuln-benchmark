"""Redact secrets and contact PII from failure payloads."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from jobbot.ops.pii_guard import redact as redact_contact

_SECRET_KV = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization|cookie|bearer)"
    r"\s*[=:]\s*([^\s\"']+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+=/]+")
_OPENAI = re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")
_CDP_WS = re.compile(r"(?i)(ws://|wss://|http://|https://)[^\s\"']+/devtools/[^\s\"']+")


def redact_text(
    text: str,
    *,
    max_len: int = 8000,
    extra_secrets: Sequence[str] = (),
) -> str:
    """Strip secrets and contact PII, then truncate."""
    if not text:
        return ""
    out = text
    for secret in extra_secrets:
        if secret:
            out = out.replace(secret, "<redacted>")
    out = _SECRET_KV.sub(r"\1=<redacted>", out)
    out = _BEARER.sub("Bearer <redacted>", out)
    out = _OPENAI.sub("sk-<redacted>", out)
    out = _CDP_WS.sub(r"\1<host>/devtools/<redacted>", out)
    out = redact_contact(out)
    if len(out) > max_len:
        out = out[: max_len - 20] + "\n…[truncated]"
    return out


def host_only_url(url: str) -> str:
    """Keep scheme+host(+path); drop query and fragment (often tokens)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return "<invalid-url>"
    if not parts.scheme and not parts.netloc:
        return redact_text(url, max_len=200)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def redact_context(
    context: dict[str, Any] | None,
    *,
    extra_secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """Copy context with URLs host-only and string values redacted."""
    if not context:
        return {}
    out: dict[str, Any] = {}
    for key, value in context.items():
        lk = str(key).lower()
        if value is None:
            out[key] = None
        elif isinstance(value, str) and ("url" in lk or lk.endswith("_uri")):
            out[key] = host_only_url(value)
        elif isinstance(value, str):
            out[key] = redact_text(value, max_len=500, extra_secrets=extra_secrets)
        elif isinstance(value, dict):
            out[key] = redact_context(value, extra_secrets=extra_secrets)
        else:
            out[key] = value
    return out
