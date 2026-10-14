"""Normalized targets and DNS/IP-aware matching.

Naive substring matching is forbidden: ``evil-example.com`` must not match
``example.com`` merely because the latter appears inside the former.
"""

from __future__ import annotations

import ipaddress
import posixpath
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import unquote, urlparse


class AssetType(StrEnum):
    URL = "url"
    WILDCARD = "wildcard"
    DOMAIN = "domain"
    IP = "ip"
    CIDR = "cidr"
    OTHER = "other"
    SOURCE_CODE = "source_code"
    EXECUTABLE = "executable"
    ANDROID_APP = "android_app"
    IOS_APP = "ios_app"
    HARDWARE = "hardware"
    OTHER_ASSET = "other_asset"
    AI_MODEL = "ai_model"
    UNSUPPORTED = "unsupported"


NETWORK_ASSET_TYPES = frozenset(
    {AssetType.URL, AssetType.WILDCARD, AssetType.DOMAIN, AssetType.IP, AssetType.CIDR}
)


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


@dataclass(frozen=True)
class NormalizedTarget:
    original: str
    scheme: str
    hostname: str
    port: int | None
    path: str
    url: str
    query: str
    ip: str | None
    cidr: str | None
    asset_type: AssetType
    program_id: str | None = None
    project_id: str | None = None

    @property
    def host_port(self) -> str:
        if self.port is None:
            return self.hostname
        return f"{self.hostname}:{self.port}"


class TargetNormalizer:
    """Parse operator-supplied targets into a comparable form."""

    def normalize(
        self,
        raw: str,
        *,
        program_id: str | None = None,
        project_id: str | None = None,
        default_scheme: str = "https",
    ) -> NormalizedTarget:
        text = (raw or "").strip()
        if not text:
            raise ValueError("Target must be a non-empty string")

        cidr = _try_cidr(text)
        if cidr is not None:
            return NormalizedTarget(
                original=raw,
                scheme="",
                hostname="",
                port=None,
                path="",
                url="",
                query="",
                ip=None,
                cidr=str(cidr),
                asset_type=AssetType.CIDR,
                program_id=program_id,
                project_id=project_id,
            )

        ip_only = _try_ip(text)
        if ip_only is not None:
            return NormalizedTarget(
                original=raw,
                scheme="",
                hostname=str(ip_only),
                port=None,
                path="",
                url="",
                query="",
                ip=str(ip_only),
                cidr=None,
                asset_type=AssetType.IP,
                program_id=program_id,
                project_id=project_id,
            )

        if "://" not in text and text.startswith("*."):
            host = _normalize_hostname(text[2:])
            return NormalizedTarget(
                original=raw,
                scheme="",
                hostname=host,
                port=None,
                path="",
                url="",
                query="",
                ip=None,
                cidr=None,
                asset_type=AssetType.WILDCARD,
                program_id=program_id,
                project_id=project_id,
            )

        parsed = urlparse(text if "://" in text else f"{default_scheme}://{text}")
        scheme = (parsed.scheme or default_scheme).lower()
        hostname = _normalize_hostname(parsed.hostname or "")
        if not hostname:
            raise ValueError(f"Cannot parse hostname from target {raw!r}")
        port = parsed.port
        default_port = _DEFAULT_PORTS.get(scheme)
        if port is None:
            port = default_port
        path = canonicalize_path(parsed.path or "/")
        query = parsed.query
        ip = str(_try_ip(hostname)) if _try_ip(hostname) else None
        netloc = _format_netloc(hostname, port, default_port)
        url = f"{scheme}://{netloc}{path}"
        if query:
            url = f"{url}?{query}"
        asset = AssetType.IP if ip else AssetType.URL
        return NormalizedTarget(
            original=raw,
            scheme=scheme,
            hostname=hostname,
            port=port,
            path=path or "/",
            url=url,
            query=query,
            ip=ip,
            cidr=None,
            asset_type=asset,
            program_id=program_id,
            project_id=project_id,
        )


def hostname_matches(host: str, patterns: tuple[str, ...]) -> bool:
    """Return True when ``host`` matches a domain/wildcard/IP pattern.

    Matching is label-based. ``evil-example.com`` does not match ``example.com``.
    """
    candidate = _normalize_hostname(host)
    if not candidate:
        return False
    for pattern in patterns:
        if _single_host_match(candidate, pattern):
            return True
    return False


def path_matches(candidate_path: str, prefix: str) -> bool:
    """Prefix match on URL path segments, not raw substrings."""
    left = canonicalize_path(candidate_path)
    right = canonicalize_path(prefix)
    if right in {"", "/"}:
        return True
    if left == right:
        return True
    return left.startswith(right.rstrip("/") + "/")


def _single_host_match(host: str, pattern: str) -> bool:
    raw = (pattern or "").strip()
    if not raw:
        return False
    if "/" in raw and not raw.startswith("*."):
        # CIDR or accidental URL — only treat slash as CIDR when it looks like one.
        cidr = _try_cidr(raw)
        if cidr is not None:
            addr = _try_ip(host)
            return addr is not None and addr in cidr
    if raw.startswith("*."):
        base = _normalize_hostname(raw[2:])
        if not base:
            return False
        if host == base:
            return True
        return host.endswith("." + base)
    exact = _normalize_hostname(raw)
    if host == exact:
        return True
    ip_host = _try_ip(host)
    ip_pat = _try_ip(exact)
    return ip_host is not None and ip_pat is not None and ip_host == ip_pat


def _normalize_hostname(host: str) -> str:
    value = (host or "").strip().lower().rstrip(".")
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    try:
        # IDN → punycode so equivalent names compare equal.
        return value.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        return value


def canonicalize_path(path: str) -> str:
    """Canonicalize a URL path for scope comparison.

    Collapses ``.`` / ``..`` segments, duplicate slashes, trailing slashes,
    and bounded percent-encoding (including encoded slashes). Equivalent
    representations must compare equal so encoding cannot bypass scope.
    """
    raw = (path or "/").replace("\\", "/")
    raw = _percent_decode_bounded(raw, rounds=3)
    if not raw.startswith("/"):
        raw = "/" + raw
    normalized = posixpath.normpath(raw)
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")
    return normalized or "/"


def _percent_decode_bounded(value: str, *, rounds: int = 3) -> str:
    current = value
    for _ in range(max(1, rounds)):
        decoded = unquote(current)
        if decoded == current:
            break
        current = decoded
    return current


def _normalize_path(path: str) -> str:
    return canonicalize_path(path)


def try_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    return _try_ip(value)


def _try_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def _try_cidr(value: str) -> ipaddress.IPv4Network | ipaddress.IPv6Network | None:
    text = value.strip()
    if "/" not in text:
        return None
    try:
        return ipaddress.ip_network(text, strict=False)
    except ValueError:
        return None


def _format_netloc(hostname: str, port: int | None, default_port: int | None) -> str:
    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is None or (default_port is not None and port == default_port):
        return host
    return f"{host}:{port}"
