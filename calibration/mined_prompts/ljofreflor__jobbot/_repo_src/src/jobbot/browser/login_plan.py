"""Login tour: which portals still need a human sign-in (issue #56).

Permanent sites reuse session evidence. Active and candidate company career sites
are listed when an account may be required. This module never launches a browser
and never writes the company registry. Opening the next gap is the CLI, behind
``--apply``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from jobbot.browser.sessions import (
    DEFAULT_PORTS,
    SITES,
    CdpEndpoint,
    ChromeProcess,
    JsonFetcher,
    SessionState,
    SessionStatus,
    discover_endpoints,
    fetch_local_json,
    free_debug_port,
    inspect_sessions,
    list_chrome_processes,
    profile_holders,
    session_for,
)
from jobbot.companies.models import CareerSite, CompanyRecord, KnowledgeStatus
from jobbot.companies.registry import (
    CompanyRegistry,
    active_career_sites,
    default_companies_path,
    load_companies,
)
from jobbot.companies.signup import AccountNeed, signup_target

COMPANIES_PROFILE = "companies"


class LoginBucket(StrEnum):
    """Where a row comes from. Active is the promoted set; candidate is not."""

    PERMANENT = "permanent"
    ACTIVE = "active"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class LoginRow:
    """One portal on the sign-in checklist. ``ready`` is never inferred for a company."""

    bucket: LoginBucket
    name: str
    url: str
    need: AccountNeed
    status: SessionStatus
    evidence: str
    hint: str
    profile_name: str
    tab_open: bool = False


@dataclass(frozen=True)
class LoginPlan:
    """Checklist plus the debugging port ``--apply`` should use for the next gap."""

    rows: tuple[LoginRow, ...]
    port: int

    def next_to_open(self) -> LoginRow | None:
        """First portal that is not proven and not already open in a tab.

        A company tab stays ``unknown`` (a host is not a session) but is not opened
        again. ``profile_busy`` is returned so the caller stops instead of skipping it.
        """
        for row in self.rows:
            if row.status is SessionStatus.READY or row.tab_open:
                continue
            return row
        return None


def plan_logins(
    root: Path,
    *,
    registry: CompanyRegistry | None = None,
    sessions: Sequence[SessionState] | None = None,
    endpoints: Sequence[CdpEndpoint] | None = None,
    processes: Sequence[ChromeProcess] | None = None,
    ports: Sequence[int] = DEFAULT_PORTS,
    fetch: JsonFetcher | None = None,
) -> LoginPlan:
    """Build the sign-in checklist. Does not launch Chrome or write companies."""
    procs = list(processes) if processes is not None else list_chrome_processes()
    fetcher = fetch if fetch is not None else fetch_local_json
    states = (
        list(sessions)
        if sessions is not None
        else inspect_sessions(root, ports=ports, fetch=fetcher, processes=procs)
    )
    found = (
        list(endpoints)
        if endpoints is not None
        else discover_endpoints(ports, fetch=fetcher, processes=procs, own_root=_browser_root(root))
    )
    reg = registry if registry is not None else load_companies(default_companies_path(root))
    rows = _permanent_rows(states)
    rows.extend(_company_rows(root, reg, found, procs, KnowledgeStatus.ACTIVE, LoginBucket.ACTIVE))
    rows.extend(
        _company_rows(root, reg, found, procs, KnowledgeStatus.CANDIDATE, LoginBucket.CANDIDATE)
    )
    return LoginPlan(rows=tuple(rows), port=free_debug_port(ports, found, procs))


def _permanent_rows(states: Sequence[SessionState]) -> list[LoginRow]:
    rows: list[LoginRow] = []
    for spec in SITES:
        state = session_for(states, spec.site) or SessionState(
            site=spec.site,
            status=SessionStatus.NO_ENDPOINT,
            evidence="no session evidence",
            hint=f"jobbot browser chrome-debug --site {spec.site}",
        )
        rows.append(
            LoginRow(
                bucket=LoginBucket.PERMANENT,
                name=spec.site,
                url=spec.start_url,
                need=AccountNeed.NEEDED,
                status=state.status,
                evidence=state.evidence,
                hint=state.hint or f"jobbot browser chrome-debug --site {spec.site}",
                profile_name=spec.cdp_profile,
            )
        )
    return rows


def _company_rows(
    root: Path,
    registry: CompanyRegistry,
    endpoints: Sequence[CdpEndpoint],
    processes: Sequence[ChromeProcess],
    status: KnowledgeStatus,
    bucket: LoginBucket,
) -> list[LoginRow]:
    rows: list[LoginRow] = []
    for record, site in active_career_sites(registry, include_candidates=True):
        if site.status is not status:
            continue
        target = signup_target(site)
        if target.need is AccountNeed.NOT_NEEDED:
            continue
        rows.append(_company_row(root, record, site, target.need, bucket, endpoints, processes))
    return rows


def _company_row(
    root: Path,
    record: CompanyRecord,
    site: CareerSite,
    need: AccountNeed,
    bucket: LoginBucket,
    endpoints: Sequence[CdpEndpoint],
    processes: Sequence[ChromeProcess],
) -> LoginRow:
    opened = _tab_on_host(endpoints, site.domain)
    if opened is not None:
        _endpoint, short = opened
        return _company(
            record,
            site,
            need,
            bucket,
            status=SessionStatus.UNKNOWN,
            evidence=f"tab open on {short}; session not proven",
            hint=(
                "A page on this host is not a signed-in session. "
                "Finish signing in there, then re-run jobbot browser login."
            ),
            tab_open=True,
        )
    holders = profile_holders(root / "browser-data" / COMPANIES_PROFILE, processes)
    if holders:
        pids = ", ".join(str(pid) for pid in holders)
        return _company(
            record,
            site,
            need,
            bucket,
            status=SessionStatus.PROFILE_BUSY,
            evidence=f"browser-data/companies held by Chrome pid {pids}",
            hint="close that Chrome window, or point --cdp at it",
        )
    own = [endpoint for endpoint in endpoints if not endpoint.foreign]
    if own:
        count = len(own)
        return _company(
            record,
            site,
            need,
            bucket,
            status=SessionStatus.UNKNOWN,
            evidence=f"{count} CDP endpoint(s) open, none on this host; session not proven",
            hint=(
                "You sign in (password, CAPTCHA, 2FA). "
                "jobbot browser login --apply opens this page and never types the password."
            ),
        )
    return _company(
        record,
        site,
        need,
        bucket,
        status=SessionStatus.NO_ENDPOINT,
        evidence="no debugging port answered on 127.0.0.1",
        hint=(
            "You sign in (password, CAPTCHA, 2FA). "
            "jobbot browser login --apply opens this page and never types the password."
        ),
    )


def _company(
    record: CompanyRecord,
    site: CareerSite,
    need: AccountNeed,
    bucket: LoginBucket,
    *,
    status: SessionStatus,
    evidence: str,
    hint: str,
    tab_open: bool = False,
) -> LoginRow:
    return LoginRow(
        bucket=bucket,
        name=record.name,
        url=site.url,
        need=need,
        status=status,
        evidence=evidence,
        hint=hint,
        profile_name=COMPANIES_PROFILE,
        tab_open=tab_open,
    )


def _tab_on_host(endpoints: Sequence[CdpEndpoint], domain: str) -> tuple[CdpEndpoint, str] | None:
    wanted = _hostname(domain)
    if not wanted:
        return None
    for endpoint in endpoints:
        if endpoint.foreign:
            continue
        for url in endpoint.page_urls:
            host = _hostname(url)
            if host == wanted or host.endswith("." + wanted):
                return endpoint, _short_url(url)
    return None


def _hostname(url_or_domain: str) -> str:
    text = url_or_domain.strip()
    host = urlsplit(text).hostname if "://" in text else text.split("/", 1)[0]
    return (host or "").casefold().removeprefix("www.")


def _short_url(url: str) -> str:
    without_scheme = re.sub(r"^https?://", "", url)
    return without_scheme.split("?", 1)[0][:80]


def _browser_root(root: Path) -> Path:
    from jobbot.config import load_config

    return load_config(root=root).browser_data_dir
