"""URL normalization for shareable career-site knowledge (no personal tokens)."""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse


class PrivateRouteRejected(ValueError):
    """Raised when a URL cannot be shared as company knowledge (e.g. mailto:)."""


class ReservedDomainRejected(PrivateRouteRejected):
    """Raised for documentation/test domains: they are fixtures, never an employer."""


# RFC 2606 / RFC 6761 names reserved for documentation and testing.
_RESERVED_DOMAINS: frozenset[str] = frozenset({"example.com", "example.net", "example.org"})
_RESERVED_TLDS: frozenset[str] = frozenset({"example", "test", "invalid", "localhost"})


# Suffixes where the registrable domain needs three labels (empresa.com.ar).
_MULTI_SUFFIXES: frozenset[str] = frozenset(
    {
        "com.ar",
        "com.au",
        "com.br",
        "com.co",
        "com.mx",
        "com.pe",
        "com.uy",
        "com.ve",
        "co.uk",
        "co.il",
        "co.nz",
        "com.tr",
    }
)

# Prefixes companies stick in front of their own name for career hosts.
_CAREER_NAME_PREFIXES: tuple[str, ...] = (
    "trabajaen",
    "trabaja-en",
    "trabajacon",
    "trabaja",
    "careers",
    "career",
    "jobs",
    "empleos",
    "empleo",
    "talento",
    "vacantes",
)


def public_url(url: str) -> str:
    """Normalize to a shareable URL: no credentials, no query, no fragment.

    Query strings and fragments are dropped on purpose: they are where portals put
    session ids, referral tokens and per-application parameters.
    """
    raw = (url or "").strip()
    if not raw:
        msg = "empty URL"
        raise PrivateRouteRejected(msg)
    if raw.lower().startswith("mailto:"):
        msg = "email apply routes are personal data, not company knowledge"
        raise PrivateRouteRejected(msg)
    if not re.match(r"^https?://", raw, re.I):
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not host:
        msg = f"URL without host: {url!r}"
        raise PrivateRouteRejected(msg)
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "").rstrip("/")
    scheme = parsed.scheme.lower() if parsed.scheme in {"http", "https"} else "https"
    return urlunparse((scheme, netloc, path, "", "", ""))


def canonical_key(url: str) -> str:
    """Dedup key: scheme-insensitive host + path (already query/fragment free)."""
    normalized = public_url(url)
    parsed = urlparse(normalized)
    return f"{parsed.netloc}{parsed.path}".casefold()


def host_of(url: str) -> str:
    return urlparse(public_url(url)).netloc.split(":", 1)[0]


def display_url(url: str) -> str:
    """Host plus path, the part a reviewer reads; the scheme carries nothing."""
    return re.sub(r"^https?://", "", public_url(url), flags=re.I)


def is_reserved_host(host: str) -> bool:
    """True for example/test/invalid/localhost names, which no employer can own."""
    labels = (host or "").strip().casefold().rstrip(".").split(".")
    if not labels or not labels[-1]:
        return False
    if labels[-1] in _RESERVED_TLDS:
        return True
    return ".".join(labels[-2:]) in _RESERVED_DOMAINS


def ensure_not_reserved(url: str) -> str:
    """Normalized URL, or ReservedDomainRejected when its host is a fixture name."""
    normalized = public_url(url)
    host = host_of(normalized)
    if is_reserved_host(host):
        msg = f"{host} is a reserved example/test domain, not a company portal; not stored"
        raise ReservedDomainRejected(msg)
    return normalized


def registrable_domain(host: str) -> str:
    """Best-effort eTLD+1 without a public suffix list (good enough for dedup hints)."""
    labels = (host or "").lower().removeprefix("www.").split(".")
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in _MULTI_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def slugify(text: str) -> str:
    """Stable company id from a display name."""
    lowered = (text or "").strip().casefold()
    lowered = (
        lowered.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug[:60]


# Generic labels ATS vendors use, so they never name the employer.
_VENDOR_LABELS: frozenset[str] = frozenset(
    {
        "apply",
        "boards",
        "career",
        "career5",
        "careers",
        "job-boards",
        "jobs",
        "performancemanager",
        "www",
    }
)


def company_hint_from_url(url: str) -> str:
    """Guess a company slug from a career host (fallback only, never a fact)."""
    from jobbot.portals.detect import AtsKind, detect_ats

    host = host_of(url)
    if detect_ats(url) not in {AtsKind.UNKNOWN, AtsKind.EMAIL}:
        return _hint_from_ats_url(url, host)
    label = registrable_domain(host).split(".")[0]
    subdomain = host.removesuffix(registrable_domain(host)).strip(".")
    for candidate in (label, subdomain):
        name = candidate
        for prefix in _CAREER_NAME_PREFIXES:
            if name.startswith(prefix) and len(name) > len(prefix) + 1:
                name = name[len(prefix) :].strip("-")
        if name and name not in _CAREER_NAME_PREFIXES:
            return slugify(name)
    return slugify(label)


def _hint_from_ats_url(url: str, host: str) -> str:
    """On an ATS, the employer is the tenant subdomain or the board path segment."""
    tenant = host.split(".")[0]
    if tenant not in _VENDOR_LABELS and not re.fullmatch(r"wd\d+", tenant):
        return slugify(tenant)
    segments = [seg for seg in urlparse(public_url(url)).path.split("/") if seg]
    return slugify(segments[0]) if segments else slugify(tenant)
