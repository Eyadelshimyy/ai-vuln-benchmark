"""Learn portal truth from a page you are already on (issue #45).

Oneshot classifies career URLs from the outside and often leaves ``ats=unknown``.
Component truth — ATS markers and form questions — is observed **after** the human
has entered (login / account HITL). This module never opens a browser and never
creates an account; it only reads HTML and, with ``apply=True``, writes technical
observations plus form questions (no typed answers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobbot.companies.models import (
    CareerSite,
    CareerSiteType,
    CompanyRecord,
    DiscoverySource,
    KnowledgeStatus,
)
from jobbot.companies.registry import (
    CompanyRegistry,
    default_companies_path,
    load_companies,
    save_companies,
)
from jobbot.companies.signup import AccountNeed, signup_target
from jobbot.companies.urls import ReservedDomainRejected
from jobbot.config import JobbotConfig
from jobbot.portals.detect import AtsKind, detect_ats, detect_ats_in_html
from jobbot.portals.form_learn import (
    FormKnowledge,
    default_form_knowledge_path,
    learn_form_html,
    load_form_knowledge,
    save_form_knowledge,
    upsert_form,
)


@dataclass(frozen=True)
class ReconReport:
    """What recon saw, and whether anything was persisted."""

    company_id: str
    company_name: str
    url: str
    previous_ats: AtsKind
    detected_ats: AtsKind
    evidence: str
    form: FormKnowledge
    account_need: AccountNeed
    wrote_companies: bool = False
    wrote_forms: bool = False
    conflict: str | None = None
    hint: str = ""
    field_labels: tuple[str, ...] = field(default_factory=tuple)
    refused: str | None = None

    @property
    def form_field_count(self) -> int:
        return len(self.field_labels)


class ReconError(ValueError):
    """Company or career site cannot be resolved for recon."""


def resolve_recon_site(
    registry: CompanyRegistry,
    company: str,
    *,
    site_url: str | None = None,
) -> tuple[CompanyRecord, CareerSite]:
    """Pick the career site to recon: explicit URL, else first active, else first."""
    record = registry.find_company(company)
    if record is None:
        msg = f"Unknown company {company!r} — learn or import it first"
        raise ReconError(msg)
    if not record.career_sites:
        msg = f"No career site stored for {record.name}"
        raise ReconError(msg)
    if site_url:
        site = record.find_site(site_url)
        if site is None:
            msg = f"No career site {site_url!r} under {record.id}"
            raise ReconError(msg)
        return record, site
    active = record.sites_with_status(KnowledgeStatus.ACTIVE)
    if active:
        return record, active[0]
    return record, record.career_sites[0]


def recon_from_html(
    registry: CompanyRegistry,
    record: CompanyRecord,
    site: CareerSite,
    html: str,
    *,
    apply: bool = False,
    companies_path: Path | None = None,
    forms_path: Path | None = None,
) -> ReconReport:
    """Detect ATS markers and form questions from inside HTML. Writes only if apply."""
    previous = site.ats
    detected, evidence = detect_ats_in_html(html)
    if detected == AtsKind.UNKNOWN:
        host_kind = detect_ats(site.url)
        if host_kind != AtsKind.UNKNOWN:
            detected = host_kind
            evidence = f"host rule: {site.domain}"

    form = learn_form_html(
        html,
        url=site.url,
        company=record.name,
        ats=detected if detected != AtsKind.UNKNOWN else None,
    )
    labels = tuple(field.label for field in form.fields)
    probe_site = site.model_copy(update={"ats": detected}) if detected != AtsKind.UNKNOWN else site
    need = signup_target(probe_site).need

    conflict: str | None = None
    refused: str | None = None
    wrote_companies = False
    wrote_forms = False

    if apply and detected != AtsKind.UNKNOWN:
        try:
            outcome = registry.observe(
                company=record.name,
                company_id=record.id,
                url=site.url,
                source=DiscoverySource.USER_OBSERVATION,
                site_type=CareerSiteType.ATS_INSTANCE,
                ats=detected,
                evidence=evidence or "inside recon HTML marker",
                country=record.country,
                status=site.status if site.status != KnowledgeStatus.REJECTED else None,
            )
        except ReservedDomainRejected as exc:
            refused = str(exc)
        else:
            conflict = outcome.conflict
            wrote_companies = True
            if companies_path is not None:
                save_companies(registry, companies_path)

    if apply and form.readable and form.fields:
        existing = load_form_knowledge(forms_path) if forms_path else []
        updated = upsert_form(existing, form)
        if forms_path is not None:
            save_form_knowledge(updated, forms_path)
        wrote_forms = True

    hint = _hint(detected, need, wrote_companies=wrote_companies, wrote_forms=wrote_forms)
    return ReconReport(
        company_id=record.id,
        company_name=record.name,
        url=site.url,
        previous_ats=previous,
        detected_ats=detected,
        evidence=evidence,
        form=form,
        account_need=need,
        wrote_companies=wrote_companies,
        wrote_forms=wrote_forms,
        conflict=conflict,
        hint=hint,
        field_labels=labels,
        refused=refused,
    )


def plan_recon(
    config: JobbotConfig,
    company: str,
    *,
    html: str | None = None,
    site_url: str | None = None,
    apply: bool = False,
) -> ReconReport:
    """Load registries, optionally apply, return the report (fixture/HTML path)."""
    companies_path = default_companies_path(config.root)
    forms_path = default_form_knowledge_path(config.root)
    registry = load_companies(companies_path)
    record, site = resolve_recon_site(registry, company, site_url=site_url)
    if html is None:
        need = signup_target(site).need
        return ReconReport(
            company_id=record.id,
            company_name=record.name,
            url=site.url,
            previous_ats=site.ats,
            detected_ats=site.ats,
            evidence="",
            form=FormKnowledge(
                url=site.url,
                company=record.name,
                ats=site.ats,
                readable=False,
                evidence="no HTML yet — pass --fixture or --cdp",
            ),
            account_need=need,
            hint=(
                "Dry-run preview only. Capture a page with --fixture PATH or "
                "--cdp URL (after you are signed in), then --apply to store evidence."
            ),
        )
    return recon_from_html(
        registry,
        record,
        site,
        html,
        apply=apply,
        companies_path=companies_path if apply else None,
        forms_path=forms_path if apply else None,
    )


def _hint(
    ats: AtsKind,
    need: AccountNeed,
    *,
    wrote_companies: bool,
    wrote_forms: bool,
) -> str:
    parts: list[str] = []
    if ats == AtsKind.UNKNOWN:
        parts.append(
            "ats still unknown — enter further (signup / profile) and recon again; "
            "do not guess from the hostname"
        )
    elif wrote_companies:
        parts.append(f"stored ats={ats.value} observation")
    else:
        parts.append(f"would store ats={ats.value} with --apply")
    if wrote_forms:
        parts.append("form questions saved to form_knowledge")
    elif need != AccountNeed.NOT_NEEDED:
        parts.append("next: companies signup NOMBRE (fill --apply is #44)")
    else:
        parts.append("no standing account needed — jobbot cv sync --apply (HITL fill; #43)")
    return "; ".join(parts)
