"""Follow HTTP redirects to resolve short links (lnkd.in, etc.) — no stealth."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("jobbot.portals.redirect")

_MAX_REDIRECTS = 8
_USER_AGENT = "jobbot/0.1 (local; redirect resolve)"
_MAX_INTERSTITIAL_BYTES = 512_000

# LinkedIn no longer 30x-redirects a lnkd.in link that leaves the site: it answers 200 with
# "This link will take you to a page that's not on LinkedIn" and prints the destination in a
# single anchor. That anchor is LinkedIn stating the target, so reading it is not evasion.
_INTERSTITIAL_DEST_RE = re.compile(
    r"<a\b[^>]*\bdata-tracking-control-name=\"external_url_click\"[^>]*\bhref=\"([^\"]+)\"",
    re.IGNORECASE,
)


def _client(**kwargs: Any) -> httpx.Client:
    """Seam for tests: httpx handles the redirect chain and relative locations."""
    return httpx.Client(**kwargs)


def read_interstitial_destination(html: str) -> str | None:
    """External URL declared by LinkedIn's leaving-the-site page, or None."""
    match = _INTERSTITIAL_DEST_RE.search(html or "")
    if match is None:
        return None
    dest = match.group(1).strip()
    return dest if dest.startswith(("http://", "https://")) else None


def follow_redirect_url(url: str, *, timeout: float = 15.0) -> str:
    """Return final URL after redirects; original URL on failure."""
    raw = url.strip()
    if not raw:
        return url
    target = raw if raw.startswith(("http://", "https://")) else f"https://{raw}"
    try:
        with _client(
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = client.head(target)
            # Plenty of ATS hosts answer HEAD with 405/501 but redirect fine on GET.
            if response.status_code >= 400:
                response = client.get(target)
            final = str(response.url)
            # lnkd.in may answer 200 with an interstitial that names the destination.
            if final == target or _is_lnkd_in(urlparse(final).hostname or ""):
                destination = _interstitial_destination(client, final)
                if destination:
                    return destination
            return final
    except httpx.HTTPError as exc:
        logger.debug("Redirect resolve failed for %s: %s", target, exc)
        return target


def expand_url_map(urls: list[str]) -> dict[str, str]:
    """Map each URL to its destination (itself when there is nothing to resolve)."""
    out: dict[str, str] = {}
    for url in urls:
        if not url or url in out:
            continue
        host = (urlparse(url).hostname or "").lower()
        out[url] = follow_redirect_url(url) if _should_resolve(host) else url
    return out


def expand_urls(urls: list[str]) -> list[str]:
    """Resolve redirects; dedupe while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for url, resolved in expand_url_map(urls).items():
        for candidate in (url, resolved):
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def _should_resolve(host: str) -> bool:
    if not host:
        return False
    if host.endswith("lnkd.in") or host == "lnkd.in":
        return True
    return bool("linkedin.com" in host and "redir" in host)


def _is_lnkd_in(host: str) -> bool:
    return host == "lnkd.in" or host.endswith(".lnkd.in")


def _interstitial_destination(client: httpx.Client, url: str) -> str | None:
    """Destination behind a short link that answered 200 with an interstitial page."""
    host = (urlparse(url).hostname or "").lower()
    if not _is_lnkd_in(host):
        return None
    try:
        response = client.get(url)
        body = response.content[:_MAX_INTERSTITIAL_BYTES]
    except httpx.HTTPError as exc:
        logger.debug("Interstitial fetch failed for %s: %s", url, exc)
        return None
    return read_interstitial_destination(body.decode("utf-8", errors="replace"))
