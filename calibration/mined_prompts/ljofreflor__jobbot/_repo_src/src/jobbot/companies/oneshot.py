"""One-shot seeding of company career portals from public sources (not a crawler)."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import yaml
from pydantic import BaseModel, Field

from jobbot.companies.discovery import SiteClassification, career_page_evidence, classify_url
from jobbot.companies.models import (
    CareerSiteType,
    DiscoverySource,
    KnowledgeStatus,
    utc_now,
)
from jobbot.companies.registry import CompanyRegistry, ObserveOutcome
from jobbot.companies.urls import (
    PrivateRouteRejected,
    ReservedDomainRejected,
    canonical_key,
    public_url,
    registrable_domain,
    slugify,
)
from jobbot.portals.detect import AtsKind

logger = logging.getLogger("jobbot.companies.oneshot")

USER_AGENT = "jobbot/0.1 (local; company career portal discovery)"

# Public paths companies use for their career pages (ES/EN).
CAREER_PATHS: tuple[str, ...] = (
    "/careers",
    "/carreras",
    "/trabaja-con-nosotros",
    "/trabaja-con-nosotros/",
    "/trabaja-en-nosotros",
    "/trabaje-con-nosotros",
    "/empleos",
    "/empleo",
    "/vacantes",
    "/postula",
    "/postulaciones",
    "/trabajo",
    "/trabaja",
    "/talento",
    "/personas",
    "/unete-a-nosotros",
    "/oportunidades-laborales",
    "/jobs",
    "/join-us",
    "/work-with-us",
)

CAREER_SUBDOMAINS: tuple[str, ...] = (
    "careers",
    "empleos",
    "empleo",
    "vacantes",
    "postula",
    "trabaja",
    "trabajaen",
    "jobs",
    "talento",
    "talent",
)


class CompanySeed(BaseModel):
    """One line of the oneshot input list: public identity only."""

    name: str = Field(min_length=1)
    id: str | None = None
    country: str | None = None
    sector: str | None = None
    domain: str | None = None
    career_url_hints: list[str] = Field(default_factory=list)

    @property
    def company_id(self) -> str:
        return self.id or slugify(self.name)


class CompanyPortalCandidate(BaseModel):
    """Candidate knowledge: reviewable, never canonical truth."""

    company: str
    company_id: str
    country: str | None = None
    sector: str | None = None
    career_url: str
    site_type: CareerSiteType = CareerSiteType.UNKNOWN
    ats: AtsKind = AtsKind.UNKNOWN
    reached_from: str | None = None
    evidence: str = ""
    source: DiscoverySource = DiscoverySource.WEB_DISCOVERY
    checked_at: datetime = Field(default_factory=utc_now)
    status: KnowledgeStatus = KnowledgeStatus.CANDIDATE


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    html: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...


@dataclass
class UrllibFetcher:
    """Sequential, polite GET: one request at a time, capped body, no stealth."""

    timeout: float = 15.0
    max_bytes: int = 200_000
    delay: float = 1.0
    sleep_fn: Callable[[float], None] = time.sleep
    _first: bool = field(default=True, repr=False)

    def fetch(self, url: str) -> FetchResult:
        if not self._first and self.delay:
            self.sleep_fn(self.delay)
        self._first = False
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310
                body = resp.read(self.max_bytes)
                charset = resp.headers.get_content_charset() or "utf-8"
                return FetchResult(
                    url=resp.geturl() or url,
                    status=int(resp.status),
                    html=body.decode(charset, errors="replace"),
                )
        except urllib.error.HTTPError as exc:
            return FetchResult(url=url, status=int(exc.code))
        except OSError as exc:
            logger.debug("Fetch failed for %s: %s", url, exc)
            return FetchResult(url=url, status=0)


@dataclass
class OneshotReport:
    candidates: list[CompanyPortalCandidate] = field(default_factory=list)
    companies_seen: int = 0
    requests_made: int = 0
    companies_without_portal: list[str] = field(default_factory=list)
    companies_blocked: list[str] = field(default_factory=list)
    companies_robots_skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateGroups:
    """The run as a map: which kinds of portal, and which platforms behind them."""

    by_site_type: dict[str, list[CompanyPortalCandidate]]
    by_ats: dict[str, list[CompanyPortalCandidate]]

    def rows(self) -> list[tuple[str, str, int, int]]:
        """(axis, label, portals, companies), biggest group first, ties by label."""
        rows: list[tuple[str, str, int, int]] = []
        for axis, groups in (("site_type", self.by_site_type), ("ats", self.by_ats)):
            for label, items in groups.items():
                rows.append((axis, label, len(items), len({i.company_id for i in items})))
        return sorted(rows, key=lambda row: (-row[2], row[0], row[1]))


def group_candidates(candidates: Iterable[CompanyPortalCandidate]) -> CandidateGroups:
    """Group by what the portal is and by which platform runs it."""
    by_site_type: dict[str, list[CompanyPortalCandidate]] = {}
    by_ats: dict[str, list[CompanyPortalCandidate]] = {}
    for candidate in candidates:
        by_site_type.setdefault(str(candidate.site_type), []).append(candidate)
        by_ats.setdefault(str(candidate.ats), []).append(candidate)
    return CandidateGroups(by_site_type=by_site_type, by_ats=by_ats)


class RobotsVerdict(StrEnum):
    """Why a URL will or will not be fetched, kept apart on purpose.

    Being told not to read a page and being unable to ask are different facts, and
    only the second one says the site is refusing us.
    """

    ALLOWED = "allowed"
    DISALLOWED = "disallowed"
    HOST_REFUSED = "host_refused"


@dataclass
class RobotsPolicy:
    """What the site says we may read, asked once per host."""

    fetcher: Fetcher
    user_agent: str = USER_AGENT
    _parsers: dict[str, RobotFileParser | None] = field(default_factory=dict, repr=False)
    _refused: set[str] = field(default_factory=set, repr=False)
    requests: int = 0

    def verdict(self, url: str) -> RobotsVerdict:
        host = urlparse(url).netloc
        if not host:
            return RobotsVerdict.HOST_REFUSED
        if host not in self._parsers:
            self._parsers[host] = self._load(url, host)
        if host in self._refused:
            return RobotsVerdict.HOST_REFUSED
        parser = self._parsers[host]
        if parser is None:
            return RobotsVerdict.ALLOWED
        if parser.can_fetch(self.user_agent, url):
            return RobotsVerdict.ALLOWED
        return RobotsVerdict.DISALLOWED

    def allows(self, url: str) -> bool:
        return self.verdict(url) is RobotsVerdict.ALLOWED

    def _load(self, url: str, host: str) -> RobotFileParser | None:
        scheme = urlparse(url).scheme or "https"
        result = self.fetcher.fetch(f"{scheme}://{host}/robots.txt")
        self.requests += 1
        if is_refusal(result.status):
            # The host would not even serve its own rules; that is a refusal by it.
            self._refused.add(host)
            return None
        if not result.ok or not result.html.strip():
            return None
        parser = RobotFileParser()
        parser.parse(result.html.splitlines())
        return parser


def load_seeds(path: Path) -> list[CompanySeed]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("companies", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        msg = f"{path} must contain a list of companies"
        raise ValueError(msg)
    return [CompanySeed.model_validate(item) for item in items]


def load_search_hits(path: Path) -> dict[str, list[str]]:
    """Read already-obtained web search results: {company: [urls]}.

    JobBot does not query a search engine itself; hits collected elsewhere can be
    fed in here and are ranked below the company's official domain.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    out: dict[str, list[str]] = {}
    if isinstance(items, dict):
        for company, urls in items.items():
            out[slugify(str(company))] = [str(u) for u in (urls or [])]
        return out
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            company = slugify(str(item.get("company") or ""))
            urls = item.get("urls") or ([item["url"]] if item.get("url") else [])
            if company:
                out.setdefault(company, []).extend(str(u) for u in urls)
    return out


def candidate_urls(seed: CompanySeed) -> list[str]:
    """Known hard URLs first, then official-domain path/subdomain probes.

    ``career_url_hints`` is how a once-found portal is reused without rediscovery.
    Path probes are a fallback when only the corporate domain is known.
    """
    urls: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        try:
            normalized = public_url(raw)
        except PrivateRouteRejected:
            return
        key = canonical_key(normalized)
        if key not in seen:
            seen.add(key)
            urls.append(normalized)

    for hint in seed.career_url_hints:
        add(hint)
    domain = (seed.domain or "").strip().casefold().removeprefix("www.")
    if domain:
        for path in CAREER_PATHS:
            add(f"https://{domain}{path}")
        for sub in CAREER_SUBDOMAINS:
            add(f"https://{sub}.{domain}")
    return urls


REFUSAL_STATUSES: frozenset[int] = frozenset({0, 401, 403, 405, 406, 408, 429})


def is_refusal(status: int) -> bool:
    """True when the site would not answer us, so its silence proves nothing.

    A 404 is an answer (the host works, that path does not exist); a dropped
    connection, an access denial or a server error is not.
    """
    return status in REFUSAL_STATUSES or status >= 500


def www_variant(url: str) -> str:
    """``https://empresa.cl/careers`` → ``https://www.empresa.cl/careers`` ('' if N/A)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or host != registrable_domain(host):
        return ""
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=f"www.{netloc}"))


@dataclass
class CompanyProbe:
    """Outcome of probing one company, including why nothing was found."""

    candidates: list[CompanyPortalCandidate] = field(default_factory=list)
    requests: int = 0
    answers: int = 0
    refusals: int = 0
    robots_skips: int = 0

    @property
    def blocked(self) -> bool:
        """The company never answered us: absence of a portal is not established."""
        return self.answers == 0 and self.refusals > 0

    @property
    def robots_only(self) -> bool:
        """We never looked, because we were asked not to."""
        return self.answers == 0 and self.refusals == 0 and self.robots_skips > 0


def discover_company(
    seed: CompanySeed,
    fetcher: Fetcher,
    *,
    search_hits: Sequence[str] = (),
    max_probes: int = 10,
    max_sites: int = 3,
    robots: RobotsPolicy | None = None,
) -> CompanyProbe:
    """Probe public career URLs for one company (official domain first)."""
    official: list[tuple[str, DiscoverySource]] = [
        (url, DiscoverySource.OFFICIAL_SITE) for url in candidate_urls(seed)
    ]
    official_keys = {canonical_key(url) for url, _ in official}
    extra: list[tuple[str, DiscoverySource]] = []
    for raw in search_hits:
        try:
            normalized = public_url(raw)
        except PrivateRouteRejected:
            continue
        if canonical_key(normalized) not in official_keys:
            extra.append((normalized, DiscoverySource.WEB_DISCOVERY))
    # The official domain is probed first, but search hits are never crowded out.
    hints = [*official[:max_probes], *extra[:max_probes]]
    probe = CompanyProbe()
    keys: set[str] = set()
    prefer_www = False
    www_ruled_out = False
    for url, source in hints:
        if len(probe.candidates) >= max_sites:
            break
        target = (www_variant(url) or url) if prefer_www else url
        if robots is not None:
            verdict = robots.verdict(target)
            if verdict is RobotsVerdict.DISALLOWED:
                probe.robots_skips += 1
                continue
            if verdict is RobotsVerdict.HOST_REFUSED:
                probe.refusals += 1
                continue
        result = fetcher.fetch(target)
        probe.requests += 1
        if is_refusal(result.status) and not prefer_www and not www_ruled_out:
            # Sites that only serve the www host refuse the apex outright.
            alternative = www_variant(url)
            if alternative:
                result = fetcher.fetch(alternative)
                probe.requests += 1
                if is_refusal(result.status):
                    www_ruled_out = True
                else:
                    prefer_www = True
        if is_refusal(result.status):
            probe.refusals += 1
        else:
            probe.answers += 1
        if not result.ok:
            continue
        classification = _classify_fetch(url, result)
        if not classification.is_company_specific:
            continue
        verified = _verifiable_evidence(classification, result)
        if not verified:
            continue
        key = canonical_key(classification.url)
        if key in keys:
            continue
        keys.add(key)
        probe.candidates.append(
            CompanyPortalCandidate(
                company=seed.name,
                company_id=seed.company_id,
                country=seed.country,
                sector=seed.sector,
                career_url=classification.url,
                site_type=classification.site_type,
                ats=classification.ats,
                reached_from=(
                    classification.requested_url if classification.redirects_to else None
                ),
                evidence=verified,
                source=source,
            )
        )
    return probe


def run_oneshot(
    seeds: Sequence[CompanySeed],
    fetcher: Fetcher,
    *,
    search_hits: dict[str, list[str]] | None = None,
    limit: int | None = None,
    on_company: Callable[[CompanySeed, list[CompanyPortalCandidate]], None] | None = None,
    respect_robots: bool = True,
) -> OneshotReport:
    """Seed candidates for a company list. Never touches the canonical registry."""
    report = OneshotReport()
    # One policy for the whole run: hosts repeat across companies and paths.
    robots = RobotsPolicy(fetcher) if respect_robots else None
    for seed in list(seeds)[: limit or len(seeds)]:
        hits = (search_hits or {}).get(seed.company_id, [])
        probe = discover_company(seed, fetcher, search_hits=hits, robots=robots)
        report.companies_seen += 1
        report.requests_made += probe.requests
        report.candidates.extend(probe.candidates)
        if not probe.candidates:
            if probe.blocked:
                report.companies_blocked.append(seed.name)
            elif probe.robots_only:
                report.companies_robots_skipped.append(seed.name)
            else:
                report.companies_without_portal.append(seed.name)
        if on_company is not None:
            on_company(seed, probe.candidates)
    if robots is not None:
        report.requests_made += robots.requests
    return report


def write_candidates(candidates: Iterable[CompanyPortalCandidate], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "note": (
            "Candidate knowledge from public web discovery. Review, then "
            "`jobbot companies import` + `jobbot companies promote`."
        ),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_candidates(path: Path) -> list[CompanyPortalCandidate]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("candidates", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        msg = f"{path} must contain a list of candidates"
        raise ValueError(msg)
    return [CompanyPortalCandidate.model_validate(item) for item in items]


def import_candidates(
    registry: CompanyRegistry,
    candidates: Iterable[CompanyPortalCandidate],
    *,
    confirm: Callable[[CompanyPortalCandidate], bool] | None = None,
) -> list[ObserveOutcome]:
    """Merge candidates into the registry, still as candidate knowledge."""
    outcomes: list[ObserveOutcome] = []
    for candidate in candidates:
        if confirm is not None and not confirm(candidate):
            continue
        try:
            outcome = registry.observe(
                company=candidate.company,
                company_id=candidate.company_id,
                url=candidate.career_url,
                source=candidate.source,
                site_type=candidate.site_type,
                ats=candidate.ats,
                evidence=candidate.evidence,
                country=candidate.country,
                sector=candidate.sector,
                reached_from=candidate.reached_from,
                status=KnowledgeStatus.CANDIDATE,
                now=candidate.checked_at,
            )
        except ReservedDomainRejected as exc:
            logger.warning("Skipped %s: %s", candidate.company, exc)
            continue
        outcomes.append(outcome)
    return outcomes


def _verifiable_evidence(classification: SiteClassification, result: FetchResult) -> str:
    """Evidence good enough to propose a candidate; '' means we stay quiet."""
    bits: list[str] = []
    if classification.ats != AtsKind.UNKNOWN or classification.redirects_to:
        bits.append(classification.evidence)
    page = career_page_evidence(result.html)
    if page:
        bits.append(page)
    if not bits:
        return ""
    return "; ".join(bit for bit in bits if bit)


def _classify_fetch(url: str, result: FetchResult) -> SiteClassification:
    """Classify using the fetched URL (redirects included) plus embedded ATS markers."""
    classification = classify_url(url, html=result.html)
    try:
        final = public_url(result.url)
    except PrivateRouteRejected:
        return classification
    if canonical_key(final) == canonical_key(url):
        return classification
    followed = classify_url(final, html=result.html)
    redirect_note = f"redirect: {classification.url} → {followed.url}"
    evidence = "; ".join(bit for bit in (redirect_note, followed.evidence) if bit)
    return SiteClassification(
        url=followed.url,
        requested_url=classification.url,
        domain=followed.domain,
        site_type=followed.site_type,
        ats=followed.ats,
        evidence=evidence,
        redirects_to=followed.url,
    )
