"""Assemble what a portal registration will ask — and what the profile already answers.

Creating an account is still the candidate's irreversible act: password, terms acceptance
and the final create/submit click. This module builds the **sheet** only (no HTTP, no
browser driver), which is what makes that purity checkable in tests.

Allowed automation (issue #44) lives in portal adapters behind ``companies signup --apply``:
fill fields ``profile.yaml`` already answers and attach the built CV. Never invent a
password, never accept terms alone, never solve CAPTCHA/2FA, never click create without
HITL confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jobbot.companies.models import CareerSite
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind
from jobbot.portals.field_homologation import answer_for_label
from jobbot.portals.form_learn import FieldKind, FormKnowledge
from jobbot.portals.sso import SsoProvider, provider_label


class AccountNeed(StrEnum):
    """Whether this portal makes you register before applying."""

    NEEDED = "needed"
    NOT_NEEDED = "not_needed"
    UNKNOWN = "unknown"


# What each platform requires, by how it works — not by how it looks.
_ACCOUNT_BY_ATS: dict[AtsKind, AccountNeed] = {
    AtsKind.GREENHOUSE: AccountNeed.NOT_NEEDED,
    AtsKind.LEVER: AccountNeed.NOT_NEEDED,
    AtsKind.ASHBY: AccountNeed.NOT_NEEDED,
    AtsKind.WORKDAY: AccountNeed.NEEDED,
    AtsKind.SUCCESSFACTORS: AccountNeed.NEEDED,
    AtsKind.ORACLE: AccountNeed.NEEDED,
    AtsKind.TEAMTAILOR: AccountNeed.NOT_NEEDED,
    AtsKind.TORRE: AccountNeed.NEEDED,
    AtsKind.GETONBOARD: AccountNeed.NEEDED,
    AtsKind.INDEED: AccountNeed.NEEDED,
    AtsKind.LINKEDIN: AccountNeed.NEEDED,
}

_NOTES: dict[AccountNeed, str] = {
    AccountNeed.NOT_NEEDED: (
        "No account needed: this platform takes the application directly. "
        "Open it and apply."
    ),
    AccountNeed.NEEDED: (
        "This platform keeps a candidate profile, so it will ask you to register. "
        "JobBot may fill fields the profile already answers and attach the CV "
        "(signup --apply); you type the password, accept terms, and confirm create."
    ),
    AccountNeed.UNKNOWN: (
        "Unknown whether it needs an account. Open the portal and look for "
        "'Crear cuenta' / 'Sign up'; nothing is assumed here."
    ),
}


@dataclass(frozen=True)
class SignupTarget:
    """Where to register, and how sure we are that registering is even required."""

    url: str
    need: AccountNeed
    note: str


@dataclass(frozen=True)
class SignupItem:
    """One thing the portal will ask, next to what the profile already answers."""

    label: str
    value: str
    source: str

    @property
    def yours_to_decide(self) -> bool:
        """No fact in the profile answers this, so nobody may fill it but you."""
        return not self.value


def account_need(ats: AtsKind) -> AccountNeed:
    return _ACCOUNT_BY_ATS.get(ats, AccountNeed.UNKNOWN)


def signup_target(site: CareerSite) -> SignupTarget:
    need = account_need(site.ats)
    return SignupTarget(url=site.url, need=need, note=_NOTES[need])


def signup_sheet(
    candidate: Candidate,
    *,
    form: FormKnowledge | None = None,
) -> list[SignupItem]:
    """What to have at hand, from the profile. A password is never among it.

    With a form already observed, the sheet is that form's own questions; without
    one it is the set every portal asks. Either way a question the profile cannot
    answer is handed back empty, never filled with something plausible.
    """
    if form is not None and form.readable:
        items = [_item_for(candidate, field.label) for field in form.fields]
    else:
        items = [_item_for(candidate, label) for label in _USUAL_FIELDS]
    items.extend(_sso_items(form))
    return items


def _sso_items(form: FormKnowledge | None) -> list[SignupItem]:
    """Provider buttons are HITL: named so you can click, never started for you."""
    if form is None or not form.sso_providers:
        return []
    items: list[SignupItem] = []
    for raw in form.sso_providers:
        try:
            provider = SsoProvider(raw)
        except ValueError:
            continue
        items.append(
            SignupItem(
                label=provider_label(provider),
                value="",
                source="portal SSO — you click; JobBot never starts OAuth",
            )
        )
    return items


def screening_to_prepare(form: FormKnowledge | None) -> list[str]:
    """The questions no profile can answer: yours to decide before you start."""
    if form is None or not form.readable:
        return []
    return [
        field.label
        for field in form.fields
        if field.is_screening and field.kind is not FieldKind.CHECKBOX
    ]


# The fields portals ask for before showing you a single job.
_USUAL_FIELDS: tuple[str, ...] = (
    "Full name",
    "Email",
    "Phone",
    "City",
    "Country",
    "Current role",
    "LinkedIn",
    "Resume/CV",
)


def _item_for(candidate: Candidate, label: str) -> SignupItem:
    """Map a portal label through the homologation table (#94)."""
    value, source = answer_for_label(candidate, label)
    return SignupItem(label=label, value=value, source=source)
