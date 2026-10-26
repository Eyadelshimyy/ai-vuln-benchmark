"""ATS / recruitment portal detection and kinds."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import cast
from urllib.parse import urlparse


class AtsKind(StrEnum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    ASHBY = "ashby"
    GETONBOARD = "getonboard"
    SMARTRECRUITERS = "smartrecruiters"
    BAMBOOHR = "bamboohr"
    SUCCESSFACTORS = "successfactors"
    ORACLE = "oracle"
    TEAMTAILOR = "teamtailor"
    WORKABLE = "workable"
    RECRUITEE = "recruitee"
    BREEZY = "breezy"
    MANATAL = "manatal"
    CORNERSTONE = "cornerstone"
    TORRE = "torre"
    JOBTOME = "jobtome"
    REMOSHIFT = "remoshift"
    REMOTEYEAH = "remoteyeah"
    JOBGETHER = "jobgether"
    VACANTES = "vacantes"
    EMPLEOS_PUBLICOS = "empleos_publicos"
    INDEED = "indeed"
    LINKEDIN = "linkedin"
    EMAIL = "email"
    UNKNOWN = "unknown"


# Boards/aggregators: many companies publish there, so they never identify one employer.
JOB_BOARD_KINDS: frozenset[AtsKind] = frozenset(
    {
        AtsKind.INDEED,
        AtsKind.LINKEDIN,
        AtsKind.GETONBOARD,
        AtsKind.TORRE,
        AtsKind.JOBTOME,
        AtsKind.REMOSHIFT,
        AtsKind.REMOTEYEAH,
        AtsKind.JOBGETHER,
        AtsKind.VACANTES,
        AtsKind.EMPLEOS_PUBLICOS,
    }
)

# Aggregators that republish other employers' openings from their own ATS tenant
# (jobs.lever.co/<tenant>): the vendor is real, the tenant is still a board.
_AGGREGATOR_TENANTS: dict[str, AtsKind] = {
    "jobgether": AtsKind.JOBGETHER,
}


# Host suffix / substring → kind (checked in order)
_HOST_RULES: list[tuple[str, AtsKind]] = [
    ("greenhouse.io", AtsKind.GREENHOUSE),
    ("boards.greenhouse.io", AtsKind.GREENHOUSE),
    ("jobs.lever.co", AtsKind.LEVER),
    ("lever.co", AtsKind.LEVER),
    ("myworkdayjobs.com", AtsKind.WORKDAY),
    ("workday.com", AtsKind.WORKDAY),
    ("ashbyhq.com", AtsKind.ASHBY),
    ("jobs.ashbyhq.com", AtsKind.ASHBY),
    ("getonbrd.com", AtsKind.GETONBOARD),
    ("getonboard.com", AtsKind.GETONBOARD),
    ("smartrecruiters.com", AtsKind.SMARTRECRUITERS),
    ("bamboohr.com", AtsKind.BAMBOOHR),
    ("successfactors.com", AtsKind.SUCCESSFACTORS),
    ("successfactors.eu", AtsKind.SUCCESSFACTORS),
    ("sapsf.com", AtsKind.SUCCESSFACTORS),
    ("sapsf.eu", AtsKind.SUCCESSFACTORS),
    ("taleo.net", AtsKind.ORACLE),
    ("oraclecloud.com", AtsKind.ORACLE),
    ("teamtailor.com", AtsKind.TEAMTAILOR),
    ("workable.com", AtsKind.WORKABLE),
    ("recruitee.com", AtsKind.RECRUITEE),
    ("breezy.hr", AtsKind.BREEZY),
    ("careers-page.com", AtsKind.MANATAL),
    ("csod.com", AtsKind.CORNERSTONE),
    ("torre.ai", AtsKind.TORRE),
    ("torre.co", AtsKind.TORRE),
    ("jobtome.com", AtsKind.JOBTOME),
    ("remoshift.com", AtsKind.REMOSHIFT),
    ("remoteyeah.com", AtsKind.REMOTEYEAH),
    ("jobgether.com", AtsKind.JOBGETHER),
    ("vacantes.com", AtsKind.VACANTES),
    ("empleospublicos.cl", AtsKind.EMPLEOS_PUBLICOS),
    ("indeed.com", AtsKind.INDEED),
    ("linkedin.com", AtsKind.LINKEDIN),
]

# Embedded markers that prove an ATS behind a corporate page (technical evidence only).
_HTML_MARKERS: tuple[tuple[re.Pattern[str], AtsKind, str], ...] = (
    (
        re.compile(r"(?:boards|job-boards)\.greenhouse\.io/embed", re.I),
        AtsKind.GREENHOUSE,
        "greenhouse embed script",
    ),
    (
        re.compile(r"(?:boards|job-boards)\.greenhouse\.io/[a-z0-9_-]+", re.I),
        AtsKind.GREENHOUSE,
        "greenhouse board link",
    ),
    (re.compile(r"jobs\.lever\.co/[a-z0-9_-]+", re.I), AtsKind.LEVER, "lever board link"),
    (
        re.compile(r"[a-z0-9_-]+\.wd\d+\.myworkdayjobs\.com", re.I),
        AtsKind.WORKDAY,
        "workday tenant host",
    ),
    (
        # Board slugs may carry a dot (an employer named after its domain).
        re.compile(r"jobs\.ashbyhq\.com/[a-z0-9_-]+(?:\.[a-z0-9_-]+)*", re.I),
        AtsKind.ASHBY,
        "ashby board link",
    ),
    (
        re.compile(r"(?:careers|jobs)\.smartrecruiters\.com/[a-z0-9_-]+", re.I),
        AtsKind.SMARTRECRUITERS,
        "smartrecruiters board link",
    ),
    (re.compile(r"[a-z0-9_-]+\.teamtailor\.com", re.I), AtsKind.TEAMTAILOR, "teamtailor host"),
    (
        # Custom career domains (careers.empresa.cl) still load Teamtailor's CDN.
        re.compile(r"teamtailor(?:-cdn)?\.(?:com|io)", re.I),
        AtsKind.TEAMTAILOR,
        "teamtailor assets",
    ),
    (
        re.compile(r"(?:apply|[a-z0-9_-]+)\.workable\.com", re.I),
        AtsKind.WORKABLE,
        "workable host",
    ),
    (re.compile(r"[a-z0-9_-]+\.recruitee\.com", re.I), AtsKind.RECRUITEE, "recruitee host"),
    (re.compile(r"[a-z0-9_-]+\.breezy\.hr", re.I), AtsKind.BREEZY, "breezy host"),
    (
        re.compile(r"(?:[a-z0-9_-]+\.)?careers-page\.com(?:/[a-z0-9_-]+)?", re.I),
        AtsKind.MANATAL,
        "manatal careers page",
    ),
    (
        re.compile(r"[a-z0-9_-]*\.(?:successfactors|sapsf)\.(?:com|eu)", re.I),
        AtsKind.SUCCESSFACTORS,
        "successfactors host",
    ),
    (
        re.compile(r"[a-z0-9_-]+\.(?:taleo\.net|fa\.oraclecloud\.com)", re.I),
        AtsKind.ORACLE,
        "oracle/taleo host",
    ),
    (re.compile(r"[a-z0-9_-]+\.bamboohr\.com", re.I), AtsKind.BAMBOOHR, "bamboohr host"),
    (
        # Cornerstone OnDemand: careersite hosts and CSP/frame-src on corporate SPAs.
        re.compile(r"[a-z0-9_-]+\.csod\.com", re.I),
        AtsKind.CORNERSTONE,
        "cornerstone csod host",
    ),
)


def detect_ats(url: str) -> AtsKind:
    """Classify a recruitment URL by host (facts only; no login)."""
    raw = url.strip()
    if not raw:
        return AtsKind.UNKNOWN
    if raw.lower().startswith("mailto:"):
        return AtsKind.EMAIL
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    host = (urlparse(raw).hostname or "").lower()
    if not host:
        return AtsKind.UNKNOWN
    for needle, kind in _HOST_RULES:
        if host == needle or host.endswith("." + needle) or needle in host:
            return kind
    return AtsKind.UNKNOWN


def aggregator_board(url: str) -> AtsKind | None:
    """Board kind when an ATS tenant belongs to an aggregator, else None."""
    kind = detect_ats(url)
    if kind in {AtsKind.UNKNOWN, AtsKind.EMAIL} or kind in JOB_BOARD_KINDS:
        return None
    raw = url.strip() if re.match(r"^https?://", url.strip(), re.I) else f"https://{url.strip()}"
    parsed = urlparse(raw)
    labels = (parsed.hostname or "").lower().split(".")
    segments = [seg.casefold() for seg in parsed.path.split("/") if seg]
    for tenant in (labels[0] if labels else "", segments[0] if segments else ""):
        board = _AGGREGATOR_TENANTS.get(tenant)
        if board is not None:
            return board
    return None


def html_marker_kinds() -> tuple[AtsKind, ...]:
    """Platforms recognisable from a corporate page's own markup, in marker order."""
    return tuple(dict.fromkeys(kind for _, kind, _ in _HTML_MARKERS))


def detect_ats_in_html(html: str) -> tuple[AtsKind, str]:
    """Classify by embedded ATS markers. Returns (kind, evidence); never guesses by looks."""
    if not html:
        return AtsKind.UNKNOWN, ""
    for pattern, kind, label in _HTML_MARKERS:
        match = pattern.search(html)
        if match:
            return kind, f"html marker: {label} ({match.group(0)[:60]})"
    return AtsKind.UNKNOWN, ""


_MAX_SNIFF_BYTES = 96_000
_SNIFF_USER_AGENT = "jobbot/0.1 (local; ats sniff)"


def sniff_ats(url: str, *, timeout: float = 10.0) -> AtsKind:
    """
    Classify a vacancy URL: host rule first, then the page's own ATS markers.

    Custom career domains (``careers.empresa.cl``) look unknown by host alone but
    still load Teamtailor/Greenhouse assets — that is technical evidence, not a guess.
    """
    kind = detect_ats(url)
    if kind != AtsKind.UNKNOWN:
        return kind
    html = _fetch_html_prefix(url, timeout=timeout)
    kind, _ = detect_ats_in_html(html or "")
    return kind


def _fetch_html_prefix(url: str, *, timeout: float) -> str | None:
    import logging
    import urllib.error
    import urllib.request

    raw = (url or "").strip()
    if not raw.startswith(("http://", "https://")):
        return None
    req = urllib.request.Request(
        raw,
        headers={"User-Agent": _SNIFF_USER_AGENT, "Accept": "text/html"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — user URL
            chunk = resp.read(_MAX_SNIFF_BYTES)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logging.getLogger("jobbot.portals.detect").debug("ats sniff failed for %s: %s", raw, exc)
        return None
    try:
        return cast(str, chunk.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 — sniff must never raise
        return None


def extract_http_urls(text: str) -> list[str]:
    """Pull http(s) URLs from free text / post body."""
    pattern = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
    out: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text or ""):
        url = match.group(0).rstrip(".,;:)")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def first_external_ats_url(urls: list[str]) -> tuple[str | None, AtsKind]:
    """
    Best apply link in a post: a real ATS, else the employer's own page, else a board.

    An aggregator republishes other people's postings ("apply on the original posting"),
    so it is the last resort — an unknown host in a hiring post is usually the employer's
    own career page. Indeed is never returned.
    """
    ranked: list[tuple[str, AtsKind]] = []
    for url in urls:
        kind = detect_ats(url)
        if kind == AtsKind.LINKEDIN:
            continue
        host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
        if host.endswith("lnkd.in") or host == "lnkd.in":
            continue
        ranked.append((url, kind))
    for url, kind in ranked:
        if kind != AtsKind.UNKNOWN and kind not in JOB_BOARD_KINDS:
            return url, kind
    for url, kind in ranked:
        if kind == AtsKind.UNKNOWN:
            return url, kind
    for url, kind in ranked:
        if kind != AtsKind.INDEED:
            return url, kind
    return None, AtsKind.UNKNOWN
