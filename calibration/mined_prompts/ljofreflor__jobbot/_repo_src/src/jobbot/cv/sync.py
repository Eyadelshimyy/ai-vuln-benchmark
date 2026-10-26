"""Standing presence sync: permanent profiles + active company portals (issue #43).

``cv sync`` is a dry-run plan by default: it writes nothing and opens no browser.
``cv sync --apply`` confirms one destination at a time. Destinations are the
permanent profiles plus career sites whose status is ``active``. Candidate and
rejected sites are not planned and are not opened.

When an active site needs an account and no signed-in page is evidenced, the
action is the assisted signup sheet from ``jobbot.companies.signup`` (issue #44):
no invented password, no terms accepted alone, no CAPTCHA/2FA, no create/submit.
When no account is required, or a session is evidenced, ``--apply`` may open the
career URL and fill only fields ``profile.yaml`` already answers, and may attach
the CV PDF already built. The human submits. A confirmation is not a receipt.
ATS ``unknown`` stays unknown — a hostname is not an ATS.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from jobbot.companies.models import CareerSite, CompanyRecord, KnowledgeStatus
from jobbot.companies.registry import (
    CompanyRegistry,
    active_career_sites,
    default_companies_path,
    load_companies,
)
from jobbot.companies.signup import AccountNeed, signup_target
from jobbot.config import JobbotConfig
from jobbot.cv.propagate import TargetPlan, plan_propagation
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind

# A public career URL or a login wall is not a session. These are path shapes,
# not an employer's vocabulary.
_AUTH_MARKERS: tuple[str, ...] = (
    "login",
    "sign-in",
    "signin",
    "sign_in",
    "signup",
    "sign-up",
    "sign_up",
    "register",
    "create-account",
    "createaccount",
    "checkpoint",
    "/auth",
)
_SIGNED_IN_MARKERS: tuple[str, ...] = (
    "/profile",
    "/account",
    "/candidate",
    "/dashboard",
    "/applications",
    "userhome",
    "/my-jobs",
    "/myjobs",
)


@dataclass(frozen=True)
class CompanySyncRow:
    """One active company career site in the sync plan."""

    company_id: str
    company_name: str
    url: str
    ats: AtsKind
    need: AccountNeed
    action: str
    hint: str
    status: KnowledgeStatus
    session_evidenced: bool

    @property
    def destination(self) -> str:
        return f"{self.company_id}:{self.ats.value}"


@dataclass(frozen=True)
class SyncPlan:
    """Permanent TargetPlans plus active-company rows."""

    permanent: list[TargetPlan]
    companies: list[CompanySyncRow]


def plan_sync(
    config: JobbotConfig,
    candidate: Candidate,
    *,
    section: str = "all",
    sessions: Sequence[object] | None = None,
    cdp_url: str | None = None,
    registry: CompanyRegistry | None = None,
    page_urls: Sequence[str] | None = None,
) -> SyncPlan:
    """Build the standing-presence plan: permanentes + active companies."""
    from jobbot.browser.sessions import SessionState

    session_states: Sequence[SessionState] | None
    if sessions is None:
        session_states = None
    else:
        session_states = [state for state in sessions if isinstance(state, SessionState)]
    permanent = plan_propagation(
        config,
        candidate,
        section=section,
        sessions=session_states,
        cdp_url=cdp_url,
    )
    companies = plan_active_companies(config, registry=registry, page_urls=page_urls)
    return SyncPlan(permanent=permanent, companies=companies)


def plan_active_companies(
    config: JobbotConfig,
    *,
    registry: CompanyRegistry | None = None,
    page_urls: Sequence[str] | None = None,
) -> list[CompanySyncRow]:
    """Active career sites only — candidates and rejected never appear."""
    reg = registry if registry is not None else _load_registry(config)
    urls = tuple(page_urls or ())
    rows: list[CompanySyncRow] = []
    for record, site in active_career_sites(reg, include_candidates=False):
        if site.status != KnowledgeStatus.ACTIVE:
            continue
        rows.append(_row_for(record, site, page_urls=urls))
    return rows


def rows_to_open(rows: Sequence[CompanySyncRow]) -> list[CompanySyncRow]:
    """Rows ``--apply`` may open. Signup-only, blocked, candidate, and rejected are not."""
    return [
        row
        for row in rows
        if row.status is KnowledgeStatus.ACTIVE and row.action == "update_profile"
    ]


def career_session_url(site: CareerSite, page_urls: Sequence[str]) -> str | None:
    """A signed-in page on this career host, or None.

    An open debugging port, the public career URL, and a login wall are not evidence.
    """
    domain = site.domain.casefold().removeprefix("www.")
    if not domain:
        return None
    for url in page_urls:
        host = (urlparse(url).hostname or "").casefold().removeprefix("www.")
        if host != domain and not host.endswith("." + domain):
            continue
        lowered = url.casefold()
        if any(marker in lowered for marker in _AUTH_MARKERS):
            continue
        if any(marker in lowered for marker in _SIGNED_IN_MARKERS):
            return url
    return None


def page_urls_from_cdp(cdp_url: str | None) -> tuple[str, ...]:
    """List open http(s) tabs on a local CDP endpoint. Failure yields no evidence."""
    if not cdp_url:
        return ()
    from jobbot.browser.sessions import fetch_local_json

    try:
        pages = fetch_local_json(f"{cdp_url.rstrip('/')}/json/list")
    except (OSError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(pages, list):
        return ()
    urls: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        url = page.get("url")
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
    return tuple(urls)


def _load_registry(config: JobbotConfig) -> CompanyRegistry:
    return load_companies(default_companies_path(config.root))


def _row_for(
    record: CompanyRecord,
    site: CareerSite,
    *,
    page_urls: Sequence[str],
) -> CompanySyncRow:
    # Stored ATS only. detect_ats(url) would invent a kind from the hostname.
    target = signup_target(site)
    evidenced = career_session_url(site, page_urls) is not None
    if target.need == AccountNeed.NEEDED and not evidenced:
        action = "needs_account"
        hint = (
            f"jobbot companies signup {record.name}  "
            "(assisted signup sheet; fill --apply is #44)"
        )
    elif target.need == AccountNeed.UNKNOWN and not evidenced:
        action = "blocked"
        hint = (
            f"jobbot companies signup {record.name}  "
            "(unknown whether an account is required; open and look — ats stays unknown)"
        )
    else:
        action = "update_profile"
        hint = (
            "cv sync --apply opens this URL, fills fields profile.yaml already answers, "
            "and may attach the built CV; you submit (a yes is not a receipt)"
        )
    return CompanySyncRow(
        company_id=record.id,
        company_name=record.name,
        url=site.url,
        ats=site.ats,
        need=target.need,
        action=action,
        hint=hint,
        status=site.status,
        session_evidenced=evidenced,
    )
