"""Read public pages about hiring practice, politely and without logging in.

Same posture as company discovery: robots.txt decides what we may read, a refusal is
reported as a refusal rather than as an absence, and a page that asks us to sign in
is left alone — content behind a login was not published to us.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from jobbot.companies.oneshot import Fetcher, RobotsPolicy, RobotsVerdict, is_refusal
from jobbot.companies.urls import PrivateRouteRejected, public_url
from jobbot.ops.pii_guard import redact
from jobbot.recruiters.playbook import extract_practices
from jobbot.recruiters.sources import RecruiterSource

# Signs that the page is a door, not an article.
_LOGIN_MARKERS: tuple[str, ...] = (
    'type="password"',
    "type='password'",
    "sign in to continue",
    "please log in",
    "members only",
    "inicia sesión para",
    "solo para miembros",
    "regístrate para ver",
)


@dataclass
class DiscoveryReport:
    """What the run established, in buckets that never collapse into each other."""

    sources: list[RecruiterSource] = field(default_factory=list)
    skipped_robots: list[str] = field(default_factory=list)
    skipped_login: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    nothing_learned: list[str] = field(default_factory=list)
    requests_made: int = 0

    @property
    def unknown(self) -> int:
        """Pages we could not read; none of them proved anything either way."""
        return len(self.skipped_robots) + len(self.skipped_login) + len(self.refused)


def discover_sources(
    urls: Sequence[str],
    fetcher: Fetcher,
    *,
    respect_robots: bool = True,
    limit: int | None = None,
) -> DiscoveryReport:
    """Turn a list of public URLs into candidate sources. Writes nothing to disk."""
    report = DiscoveryReport()
    robots = RobotsPolicy(fetcher) if respect_robots else None
    for raw in list(urls)[: limit or len(urls)]:
        try:
            url = public_url(raw)
        except PrivateRouteRejected:
            continue
        if robots is not None:
            verdict = robots.verdict(url)
            if verdict is RobotsVerdict.DISALLOWED:
                report.skipped_robots.append(url)
                continue
            if verdict is RobotsVerdict.HOST_REFUSED:
                report.refused.append(url)
                continue
        result = fetcher.fetch(url)
        report.requests_made += 1
        if is_refusal(result.status) or not result.ok:
            report.refused.append(url)
            continue
        if asks_for_login(result.html):
            report.skipped_login.append(url)
            continue
        practices = extract_practices(result.html)
        if not practices:
            report.nothing_learned.append(url)
            continue
        report.sources.append(
            RecruiterSource(
                url=url,
                title=page_title(result.html),
                practices=practices,
                evidence=f"read {len(practices)} practice(s) from the public page",
            )
        )
    if robots is not None:
        report.requests_made += robots.requests
    return report


def asks_for_login(html: str) -> bool:
    """Whether the page is gated. Gated content is not public knowledge."""
    folded = (html or "").casefold()
    return any(marker in folded for marker in _LOGIN_MARKERS)


def page_title(html: str) -> str:
    """The page's own title, with any contact data taken out."""
    soup = BeautifulSoup(html or "", "html.parser")
    node = soup.title or soup.find("h1")
    if node is None:
        return ""
    return redact(node.get_text(" ", strip=True))[:180]
