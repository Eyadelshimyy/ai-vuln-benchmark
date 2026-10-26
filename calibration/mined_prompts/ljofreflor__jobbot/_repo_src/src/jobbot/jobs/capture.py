"""Capture share URLs as candidates (phone-friendly; no fetch, no CAPTCHA).

From Cursor on a phone the useful move is: keep the URL as a *candidate* until
a desktop session can recon / get / promote. This module routes one URL into the
right unfinished bucket without inventing facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from jobbot.companies.discovery import classify_url
from jobbot.companies.learn import learn_from_url
from jobbot.companies.models import DiscoverySource
from jobbot.companies.registry import (
    default_companies_path,
    load_companies,
    save_companies,
)
from jobbot.companies.urls import ReservedDomainRejected, company_hint_from_url
from jobbot.config import JobbotConfig
from jobbot.jobs.inbox import inbox_path, list_parked, normalize_park_url, park_url
from jobbot.portals.detect import AtsKind

UNRECOGNIZED_FILENAME = "url-candidates.txt"


class CaptureKind(StrEnum):
    HARD_LINK = "hard_link"
    COMPANY_PORTAL = "company_portal"
    UNRECOGNIZED = "unrecognized"


@dataclass(frozen=True)
class CaptureResult:
    kind: CaptureKind
    url: str
    detail: str
    already_present: bool = False
    company_id: str | None = None


def unrecognized_path(config: JobbotConfig) -> Path:
    return config.root / "data" / UNRECOGNIZED_FILENAME


def list_unrecognized(config: JobbotConfig) -> list[str]:
    path = unrecognized_path(config)
    if not path.is_file():
        return []
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            urls.append(text)
    return urls


def _park_unrecognized(config: JobbotConfig, url: str) -> tuple[str, bool]:
    stored = url.strip()
    if not stored:
        raise ValueError("Empty URL")
    path = unrecognized_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = list_unrecognized(config)
    if stored in existing:
        return stored, True
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stored + "\n")
    return stored, False


def capture_url(
    config: JobbotConfig,
    url: str,
    *,
    company: str | None = None,
    country: str | None = None,
) -> CaptureResult:
    """Classify offline and store as a candidate in the matching unfinished bucket."""
    raw = url.strip()
    if not raw:
        raise ValueError("Empty URL")

    try:
        normalized = normalize_park_url(raw)
    except ValueError:
        # Typical: Indeed share without jk, or empty after strip.
        stored, existed = _park_unrecognized(config, raw)
        return CaptureResult(
            kind=CaptureKind.UNRECOGNIZED,
            url=stored,
            detail="URL kept unrecognized (could not canonicalize)",
            already_present=existed,
        )

    classification = classify_url(normalized, resolve=False)

    if classification.ats in {AtsKind.INDEED, AtsKind.GETONBOARD}:
        stored, existed = park_url(config, normalized)
        return CaptureResult(
            kind=CaptureKind.HARD_LINK,
            url=stored,
            detail=f"Parked hard link ({classification.ats.value}) for later jobbot get",
            already_present=existed,
        )

    if classification.is_company_specific:
        registry = load_companies(default_companies_path(config.root))
        name = (company or company_hint_from_url(normalized) or classification.domain).strip()
        try:
            outcome = learn_from_url(
                registry,
                company=name,
                url=normalized,
                source=DiscoverySource.USER_OBSERVATION,
                country=country,
                resolve=False,
                notes="captured from share URL (candidate; not yet recon/promoted)",
            )
        except ReservedDomainRejected as exc:
            return CaptureResult(kind=CaptureKind.UNRECOGNIZED, url=normalized, detail=str(exc))
        if outcome is not None and outcome.site is not None:
            save_companies(registry, default_companies_path(config.root))
            return CaptureResult(
                kind=CaptureKind.COMPANY_PORTAL,
                url=outcome.site.url,
                detail=(
                    f"Company portal candidate status={outcome.site.status.value} "
                    f"ats={outcome.site.ats.value} type={outcome.site.site_type.value}"
                ),
                already_present=not outcome.created_site and outcome.merged,
                company_id=outcome.company.id,
            )

    stored, existed = _park_unrecognized(config, normalized)
    return CaptureResult(
        kind=CaptureKind.UNRECOGNIZED,
        url=stored,
        detail=(
            f"Unrecognized URL kept as candidate "
            f"(site_type={classification.site_type.value}, ats={classification.ats.value})"
        ),
        already_present=existed,
    )


@dataclass(frozen=True)
class CaptureInventory:
    hard_links: list[str]
    company_portals: list[tuple[str, str, str]]  # company_id, url, status
    unrecognized: list[str]


def list_candidates(config: JobbotConfig) -> CaptureInventory:
    """Everything still unfinished: parked jobs, company candidates, unknown URLs."""
    from jobbot.companies.models import KnowledgeStatus

    companies = load_companies(default_companies_path(config.root))
    portals: list[tuple[str, str, str]] = []
    for company in companies.companies:
        for site in company.career_sites:
            if site.status == KnowledgeStatus.CANDIDATE:
                portals.append((company.id, site.url, site.status.value))
    return CaptureInventory(
        hard_links=list_parked(config),
        company_portals=portals,
        unrecognized=list_unrecognized(config),
    )


def capture_paths(config: JobbotConfig) -> dict[str, Path]:
    return {
        "hard_links": inbox_path(config),
        "unrecognized": unrecognized_path(config),
        "companies": default_companies_path(config.root),
    }
