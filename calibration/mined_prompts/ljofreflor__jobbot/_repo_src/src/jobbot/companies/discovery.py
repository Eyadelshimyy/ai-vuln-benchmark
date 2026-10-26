"""Classify a URL as posting / corporate career portal / ATS instance / redirect."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from jobbot.companies.models import CareerSiteType
from jobbot.companies.urls import PrivateRouteRejected, canonical_key, host_of, public_url
from jobbot.portals.detect import (
    JOB_BOARD_KINDS,
    AtsKind,
    aggregator_board,
    detect_ats,
    detect_ats_in_html,
)
from jobbot.portals.redirect import follow_redirect_url

# Hosts/paths companies use for their own career presence (public naming patterns).
_CAREER_HOST_HINTS: tuple[str, ...] = (
    "career",
    "careers",
    "carrera",
    "carreras",
    "empleo",
    "empleos",
    "jobs",
    "recruit",
    "talento",
    "talent",
    "trabaja",
    "trabajaen",
    "vacantes",
)

_CAREER_PATH_HINTS: tuple[str, ...] = (
    "careers",
    "career",
    "carreras",
    "empleos",
    "empleo",
    "jobs",
    "join-us",
    "oportunidades",
    "oportunidades-laborales",
    "talento",
    "talent",
    "trabaja-con-nosotros",
    "trabaja-en",
    "trabaja",
    "unete",
    "unete-a-nosotros",
    "vacantes",
    "work-with-us",
    "working-here",
)

# Path shapes that mean "one specific opening", not the portal itself.
_POSTING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/jobs?/\d+", re.I),
    re.compile(r"/jobs?/[a-z0-9][a-z0-9._-]{6,}", re.I),
    re.compile(r"/job/[^/]+", re.I),
    re.compile(r"/vacante(?:s)?/[^/]+", re.I),
    re.compile(r"/position(?:s)?/[^/]+", re.I),
    re.compile(r"/opening(?:s)?/[^/]+", re.I),
    re.compile(r"/oferta(?:s)?/[^/]+", re.I),
    re.compile(r"/vacancy/[^/]+", re.I),
    # A listing directory followed by a slug that carries an id is one opening.
    re.compile(r"/vacancies/[^/]*\d[^/]*", re.I),
    re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I),
)

_PATH_TENANT_HOSTS: frozenset[str] = frozenset({"apply.workable.com", "careers-page.com"})

_LOCALE_SEGMENT = re.compile(r"^[a-z]{2}(?:-[a-zA-Z]{2})?$")

# One vacancy, not the board that lists them. Singular on purpose: `/jobs/` and
# `/vacantes/` are listings and keep their directory. `/job/<id>` plus an
# optional title slug is still that one posting.
_SINGULAR_OPENING_TAIL = re.compile(
    r"/(?:job|vacante|vacancy|position|opening|oferta)/[^/]+(?:/.*)?$",
    re.I,
)

# Employment vocabulary grouped by concept (ES/EN). A page that only repeats one
# concept is usually something else answering 200 (soft 404, profile, landing).
_CAREER_VOCABULARY: dict[str, tuple[str, ...]] = {
    "careers": ("careers", "career page", "carreras profesionales"),
    "employment": ("empleo", "empleos", "employment"),
    "jobs": ("jobs", "job openings", "open positions", "ofertas de trabajo"),
    "vacancy": ("vacante", "vacantes", "vacancies"),
    "join": (
        "join us",
        "join our team",
        "trabaja con nosotros",
        "trabaja en nosotros",
        "work with us",
        "únete",
        "unete",
    ),
    "opportunities": ("oportunidades laborales", "oportunidades de carrera"),
    "apply": ("apply now", "postula", "postular", "aplica ahora"),
    "talent": ("talento", "talent community", "talent network"),
    "benefits": ("beneficios", "benefits"),
}

_CAREER_PAGE_WORDS: tuple[str, ...] = tuple(
    word for words in _CAREER_VOCABULARY.values() for word in words
)

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_HEADING_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)


@dataclass(frozen=True)
class SiteClassification:
    """What JobBot can honestly say about a URL, plus the evidence for it."""

    url: str
    requested_url: str
    domain: str
    site_type: CareerSiteType
    ats: AtsKind
    evidence: str
    redirects_to: str | None = None

    @property
    def is_company_specific(self) -> bool:
        """Boards host many employers, so they never identify one company."""
        return self.site_type in {
            CareerSiteType.COMPANY_CAREER_PORTAL,
            CareerSiteType.ATS_INSTANCE,
            CareerSiteType.JOB_POSTING,
        }


def classify_url(
    url: str,
    *,
    html: str | None = None,
    resolve: bool = False,
    resolver: Callable[[str], str] = follow_redirect_url,
) -> SiteClassification:
    """Classify a career URL. ATS stays ``unknown`` without technical evidence."""
    requested = public_url(url)
    final = requested
    redirects_to: str | None = None
    evidence_bits: list[str] = []
    if resolve:
        try:
            resolved = public_url(resolver(requested))
        except PrivateRouteRejected:
            resolved = requested
        if canonical_key(resolved) != canonical_key(requested):
            final = resolved
            redirects_to = resolved
            evidence_bits.append(f"redirect: {requested} → {resolved}")

    host_ats = detect_ats(final)
    ats = host_ats
    if host_ats != AtsKind.UNKNOWN:
        evidence_bits.append(f"host rule: {host_of(final)} matches {host_ats.value}")
    elif html:
        ats, html_evidence = detect_ats_in_html(html)
        if html_evidence:
            evidence_bits.append(html_evidence)

    site_type = _site_type_for(final, ats, host_ats=host_ats)
    if site_type == CareerSiteType.COMPANY_CAREER_PORTAL and ats == AtsKind.UNKNOWN:
        evidence_bits.append(f"career naming pattern: {_career_signal(final)}")
    return SiteClassification(
        url=final,
        requested_url=requested,
        domain=host_of(final),
        site_type=site_type,
        ats=ats,
        evidence="; ".join(bit for bit in evidence_bits if bit),
        redirects_to=redirects_to,
    )


def career_root_url(url: str, ats: AtsKind = AtsKind.UNKNOWN) -> str:
    """Collapse a single posting to the portal that lists the company's openings."""
    normalized = public_url(url)
    parsed = urlparse(normalized)
    host = parsed.netloc
    segments = [seg for seg in parsed.path.split("/") if seg]
    if ats in {AtsKind.GREENHOUSE, AtsKind.LEVER, AtsKind.ASHBY, AtsKind.SMARTRECRUITERS}:
        return f"https://{host}/{segments[0]}" if segments else f"https://{host}"
    if ats == AtsKind.WORKDAY:
        tenant_path = [seg for seg in segments if not _LOCALE_SEGMENT.match(seg)]
        return f"https://{host}/{tenant_path[0]}" if tenant_path else f"https://{host}"
    if ats in {AtsKind.WORKABLE, AtsKind.MANATAL} and host in _PATH_TENANT_HOSTS:
        # Shared vendor host: the company is the first path segment.
        return f"https://{host}/{segments[0]}" if segments else f"https://{host}"
    if ats in {
        AtsKind.SUCCESSFACTORS,
        AtsKind.ORACLE,
        AtsKind.TEAMTAILOR,
        AtsKind.WORKABLE,
        AtsKind.MANATAL,
    }:
        return f"https://{host}"
    if ats == AtsKind.RECRUITEE or ats == AtsKind.BAMBOOHR:
        return f"https://{host}"
    opening = _SINGULAR_OPENING_TAIL.search(parsed.path)
    if opening is not None:
        prefix = parsed.path[: opening.start()].rstrip("/")
        return f"https://{host}{prefix}" if prefix else f"https://{host}"
    if _is_posting_path(parsed.path) and segments:
        return (
            f"https://{host}/{'/'.join(segments[:-1])}" if len(segments) > 1 else f"https://{host}"
        )
    return normalized


def career_page_evidence(html: str, *, min_concepts: int = 2) -> str:
    """Evidence that a fetched page is really about employment, or '' when absent.

    HTTP 200 proves nothing: sites answer 200 for unrelated paths (soft 404s,
    user profiles named 'empleos'). A candidate needs employment wording in the
    title or heading plus several distinct employment concepts in the body.
    """
    if not html:
        return ""
    headline = ""
    for pattern, label in ((_TITLE_RE, "page title"), (_HEADING_RE, "page heading")):
        for match in pattern.finditer(html):
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", match.group(1))).strip()
            lowered = text.casefold()
            if any(word in lowered for word in _CAREER_PAGE_WORDS):
                headline = f"{label}: {text[:70]!r}"
                break
        if headline:
            break
    if not headline:
        return ""
    body = html.casefold()
    concepts = sorted(
        concept
        for concept, words in _CAREER_VOCABULARY.items()
        if any(word in body for word in words)
    )
    if len(concepts) < min_concepts:
        return ""
    return f"{headline}; career vocabulary: {', '.join(concepts)}"


def _site_type_for(url: str, ats: AtsKind, *, host_ats: AtsKind) -> CareerSiteType:
    """``ats`` says which technology; the host says whose page this is."""
    if ats == AtsKind.EMAIL:
        return CareerSiteType.UNKNOWN
    if host_ats in JOB_BOARD_KINDS or aggregator_board(url) is not None:
        return CareerSiteType.JOB_BOARD
    posting = _is_posting_path(urlparse(url).path)
    if host_ats != AtsKind.UNKNOWN:
        return CareerSiteType.JOB_POSTING if posting else CareerSiteType.ATS_INSTANCE
    # Company-owned host: still its career portal even when an ATS powers it.
    if _career_signal(url):
        return CareerSiteType.JOB_POSTING if posting else CareerSiteType.COMPANY_CAREER_PORTAL
    if ats != AtsKind.UNKNOWN:
        return CareerSiteType.JOB_POSTING if posting else CareerSiteType.COMPANY_CAREER_PORTAL
    return CareerSiteType.UNKNOWN


def _is_posting_path(path: str) -> bool:
    return any(pattern.search(path or "") for pattern in _POSTING_PATTERNS)


def _career_signal(url: str) -> str:
    """Public naming evidence that a URL is about employment, or '' when absent."""
    parsed = urlparse(public_url(url))
    host_labels = parsed.netloc.split(":", 1)[0].split(".")
    for label in host_labels[:-1]:
        for hint in _CAREER_HOST_HINTS:
            if label.startswith(hint):
                return f"host label {label!r}"
    root_label = host_labels[0]
    for hint in _CAREER_HOST_HINTS:
        if root_label.startswith(hint) and len(host_labels) == 2:
            return f"host label {root_label!r}"
    segments = [seg.casefold() for seg in parsed.path.split("/") if seg]
    for segment in segments:
        if segment in _CAREER_PATH_HINTS:
            return f"path segment {segment!r}"
    for segment in segments:
        for hint in _CAREER_PATH_HINTS:
            if hint in segment:
                return f"path segment {segment!r}"
    return ""
