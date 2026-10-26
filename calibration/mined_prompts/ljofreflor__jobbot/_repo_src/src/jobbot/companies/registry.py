"""Local registry of company ↔ career platforms (candidate → promote → shareable)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from jobbot.companies.models import (
    CareerSite,
    CareerSiteType,
    CompanyRecord,
    DiscoverySource,
    KnowledgeStatus,
    Observation,
    utc_now,
)
from jobbot.companies.urls import (
    canonical_key,
    ensure_not_reserved,
    host_of,
    registrable_domain,
    slugify,
)
from jobbot.portals.detect import AtsKind

REGISTRY_VERSION = 1


@dataclass(frozen=True)
class ObserveOutcome:
    """What an observation did to the registry (for honest CLI output)."""

    company: CompanyRecord
    site: CareerSite | None
    created_company: bool = False
    created_site: bool = False
    merged: bool = False
    conflict: str | None = None


class CompanyRegistry(BaseModel):
    version: int = REGISTRY_VERSION
    companies: list[CompanyRecord] = Field(default_factory=list)

    # ── lookups ──────────────────────────────────────────────────────────────

    def find_company(self, company: str) -> CompanyRecord | None:
        needle = slugify(company)
        for record in self.companies:
            if record.id == needle or slugify(record.name) == needle:
                return record
        return None

    def find_company_by_domain(self, domain: str) -> CompanyRecord | None:
        root = registrable_domain(domain)
        if not root:
            return None
        for record in self.companies:
            if any(registrable_domain(known) == root for known in record.domains):
                return record
        return None

    def find_site(self, url: str) -> tuple[CompanyRecord, CareerSite] | None:
        """Global URL lookup — how duplicates across companies get caught."""
        key = canonical_key(url)
        for record in self.companies:
            for site in record.career_sites:
                if site.key == key:
                    return record, site
        return None

    # ── writes ───────────────────────────────────────────────────────────────

    def upsert_company(
        self,
        *,
        name: str,
        company_id: str | None = None,
        country: str | None = None,
        sector: str | None = None,
        domains: Sequence[str] = (),
    ) -> tuple[CompanyRecord, bool]:
        record = self.find_company(company_id or name)
        created = False
        if record is None:
            record = CompanyRecord(
                id=company_id or slugify(name),
                name=name,
                country=country,
                sector=sector,
                domains=[],
            )
            self.companies.append(record)
            created = True
        if country and not record.country:
            record.country = country
        if sector and not record.sector:
            record.sector = sector
        for domain in domains:
            cleaned = domain.strip().casefold().removeprefix("www.")
            if cleaned and cleaned not in record.domains:
                record.domains.append(cleaned)
        return record, created

    def observe(
        self,
        *,
        company: str,
        url: str,
        source: DiscoverySource,
        site_type: CareerSiteType = CareerSiteType.UNKNOWN,
        ats: AtsKind = AtsKind.UNKNOWN,
        evidence: str = "",
        company_id: str | None = None,
        country: str | None = None,
        sector: str | None = None,
        domains: Sequence[str] = (),
        status: KnowledgeStatus | None = None,
        reached_from: str | None = None,
        notes: str | None = None,
        now: datetime | None = None,
    ) -> ObserveOutcome:
        """Record one sighting: dedupe, accumulate evidence, flag contradictions."""
        normalized = ensure_not_reserved(url)
        moment = now or utc_now()
        owner = self.find_site(normalized)
        # Only the company's own career hosts are corporate domains; ATS hosts are shared.
        own_host = (
            [host_of(normalized)] if site_type == CareerSiteType.COMPANY_CAREER_PORTAL else []
        )
        record, created_company = self.upsert_company(
            name=company,
            company_id=company_id,
            country=country,
            sector=sector,
            domains=[*domains, *own_host],
        )
        if owner is not None and owner[0].id != record.id:
            other, _site = owner
            return ObserveOutcome(
                company=record,
                site=None,
                created_company=created_company,
                conflict=(
                    f"{normalized} is already registered under company {other.id!r}; not duplicated"
                ),
            )

        observation = Observation(
            source=source,
            checked_at=moment,
            ats=ats,
            site_type=site_type,
            evidence=evidence,
            observed_url=normalized,
        )
        site = record.find_site(normalized)
        if site is None:
            site = CareerSite(
                url=normalized,
                domain=host_of(normalized),
                site_type=site_type,
                ats=ats,
                status=status or KnowledgeStatus.CANDIDATE,
                first_seen=moment,
                last_verified=moment if status == KnowledgeStatus.ACTIVE else None,
                reached_from=reached_from,
                observations=[observation],
                notes=notes,
            )
            record.career_sites.append(site)
            return ObserveOutcome(
                company=record,
                site=site,
                created_company=created_company,
                created_site=True,
            )

        site.observations.append(observation)
        conflict = _reconcile(site, ats=ats, site_type=site_type, status=status)
        if reached_from and not site.reached_from:
            site.reached_from = reached_from
        if notes and not site.notes:
            site.notes = notes
        if status == KnowledgeStatus.ACTIVE and site.status == KnowledgeStatus.ACTIVE:
            site.last_verified = moment
        return ObserveOutcome(
            company=record,
            site=site,
            created_company=created_company,
            merged=True,
            conflict=conflict,
        )

    def promote(
        self,
        company: str,
        *,
        url: str | None = None,
        now: datetime | None = None,
    ) -> list[CareerSite]:
        """Only path from candidate knowledge to active truth."""
        record = self.find_company(company)
        if record is None:
            return []
        moment = now or utc_now()
        targets = (
            [site for site in record.career_sites if url is None or site.key == canonical_key(url)]
            if record.career_sites
            else []
        )
        promoted: list[CareerSite] = []
        for site in targets:
            if site.status == KnowledgeStatus.REJECTED:
                continue
            site.status = KnowledgeStatus.ACTIVE
            site.last_verified = moment
            promoted.append(site)
        return promoted

    def reject(self, company: str, *, url: str) -> CareerSite | None:
        record = self.find_company(company)
        if record is None:
            return None
        site = record.find_site(url)
        if site is None:
            return None
        site.status = KnowledgeStatus.REJECTED
        return site


def _reconcile(
    site: CareerSite,
    *,
    ats: AtsKind,
    site_type: CareerSiteType,
    status: KnowledgeStatus | None,
) -> str | None:
    """Adopt new facts only when they add information; never overwrite silently."""
    conflict: str | None = None
    if ats != AtsKind.UNKNOWN:
        if site.ats == AtsKind.UNKNOWN:
            site.ats = ats
        elif site.ats != ats:
            conflict = f"ats changed: stored {site.ats.value}, observed {ats.value}"
            if conflict not in site.conflicts:
                site.conflicts.append(conflict)
            site.status = KnowledgeStatus.STALE
    if site_type != CareerSiteType.UNKNOWN and site.site_type == CareerSiteType.UNKNOWN:
        site.site_type = site_type
    if status is not None and site.status == KnowledgeStatus.CANDIDATE:
        site.status = status
    return conflict


# ── persistence ──────────────────────────────────────────────────────────────


def default_companies_path(root: Path | None = None) -> Path:
    """Companies YAML: config ``paths.companies`` when available, else ``data/companies.yaml``."""
    if root is None:
        from jobbot.config import load_config

        return load_config().companies_path
    from jobbot.config import load_config

    return load_config(root=root).companies_path


def generated_candidates_path(output_dir: Path) -> Path:
    return output_dir / "discovery" / "company_portals.generated.yaml"


def shared_export_path(output_dir: Path) -> Path:
    return output_dir / "discovery" / "companies.shared.yaml"


def load_companies(path: Path) -> CompanyRegistry:
    if not path.is_file():
        return CompanyRegistry()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, list):
        return CompanyRegistry(companies=raw)
    return CompanyRegistry.model_validate(raw)


def save_companies(registry: CompanyRegistry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.model_dump(mode="json", exclude_none=False)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def active_career_sites(
    registry: CompanyRegistry,
    *,
    include_candidates: bool = False,
) -> list[tuple[CompanyRecord, CareerSite]]:
    """Reusable knowledge for later job discovery (acceptance criterion 9)."""
    wanted = {KnowledgeStatus.ACTIVE}
    if include_candidates:
        wanted.add(KnowledgeStatus.CANDIDATE)
    out: list[tuple[CompanyRecord, CareerSite]] = []
    for record in registry.companies:
        for site in record.career_sites:
            if site.status in wanted and site.site_type != CareerSiteType.JOB_BOARD:
                out.append((record, site))
    return out


def shareable_payload(registry: CompanyRegistry) -> dict[str, object]:
    """Export only promoted, public facts — no local notes, no personal history."""
    companies: list[dict[str, object]] = []
    for record in registry.companies:
        sites = [site for site in record.career_sites if site.status == KnowledgeStatus.ACTIVE]
        if not sites:
            continue
        companies.append(
            {
                "id": record.id,
                "name": record.name,
                "country": record.country,
                "sector": record.sector,
                "domains": list(record.domains),
                "career_sites": [
                    {
                        "url": site.url,
                        "domain": site.domain,
                        "site_type": site.site_type.value,
                        "ats": site.ats.value,
                        "status": site.status.value,
                        "first_seen": site.first_seen.isoformat(),
                        "last_verified": (
                            site.last_verified.isoformat() if site.last_verified else None
                        ),
                        "reached_from": site.reached_from,
                        "confidence": site.confidence,
                        "evidence": _public_evidence(site.observations),
                    }
                    for site in sites
                ],
            }
        )
    return {"version": REGISTRY_VERSION, "companies": companies}


def _public_evidence(observations: Iterable[Observation]) -> list[dict[str, str]]:
    return [
        {
            "source": obs.source.value,
            "checked_at": obs.checked_at.date().isoformat(),
            "evidence": obs.evidence,
        }
        for obs in observations
    ]
