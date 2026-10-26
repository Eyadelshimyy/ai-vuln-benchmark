"""Fill known signup fields from profile.yaml (HITL; no irreversible actions).

The fill driver lives here so ``jobbot.companies.signup`` stays pure: no HTTP,
no Playwright. Stop before password, terms, CAPTCHA/2FA and final create/submit.
Never invent a password. Never click create/submit without explicit confirmation —
and this module never clicks create/submit at all (``submitted`` is always false).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobbot.companies.signup import AccountNeed, SignupItem, signup_sheet
from jobbot.models.candidate import Candidate
from jobbot.portals.form_learn import FieldKind, FormField, FormKnowledge, learn_form_html

logger = logging.getLogger("jobbot.ats.signup")

_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "password",
    "contraseña",
    "contrasena",
    "passcode",
    "captcha",
    "recaptcha",
    "2fa",
    "otp",
    "verification code",
    "terms",
    "términos",
    "terminos",
    "privacy",
    "privacidad",
    "consent",
    "consentimiento",
    "i agree",
    "acepto",
)

_ALWAYS_MANUAL: tuple[str, ...] = (
    "Password",
    "Terms and conditions",
    "CAPTCHA / 2FA if shown",
    "Final create/submit button",
)

_RESUME_TOKENS: tuple[str, ...] = ("resume", "cv", "curriculum", "currículum")

_FILLABLE: frozenset[FieldKind] = frozenset(
    {
        FieldKind.TEXT,
        FieldKind.EMAIL,
        FieldKind.PHONE,
        FieldKind.URL,
        FieldKind.NUMBER,
        FieldKind.DATE,
        FieldKind.LONG_TEXT,
    }
)


@dataclass(frozen=True)
class SignupFillPlan:
    """What fields will be filled and what the candidate must do themselves."""

    url: str
    need: AccountNeed
    fields_to_fill: list[str]
    needs_manual: list[str]
    can_attach_cv: bool
    cv_path: Path | None
    will_submit: bool = False


@dataclass(frozen=True)
class SignupFillResult:
    """What was actually filled. ``submitted`` is always false."""

    plan: SignupFillPlan
    filled_fields: list[str]
    attached: bool
    stopped_at: str
    submitted: bool = False


def build_fill_plan(
    candidate: Candidate,
    url: str,
    need: AccountNeed,
    form: FormKnowledge | None = None,
    cv_path: Path | None = None,
) -> SignupFillPlan:
    """What can be filled from profile.yaml and what requires HITL.

    A password field is never fillable. Terms/consent, CAPTCHA/2FA and submit
    stay manual. ``will_submit`` is always false — create/submit is not automated.
    """
    sheet = signup_sheet(candidate, form=form)
    fillable = [
        item.label
        for item in sheet
        if _is_fillable_item(item)
    ]
    manual = list(_ALWAYS_MANUAL)
    for item in sheet:
        if item.yours_to_decide or _forbidden_label(item.label):
            if item.label not in manual:
                manual.append(item.label)
        elif (
            _is_resume_label(item.label)
            and item.label not in manual
            and _attachable_pdf(cv_path) is None
        ):
            # Resume is attached separately when a PDF exists; otherwise manual.
            manual.append(item.label)

    attachable = _attachable_pdf(cv_path)
    return SignupFillPlan(
        url=url,
        need=need,
        fields_to_fill=fillable,
        needs_manual=manual,
        can_attach_cv=attachable is not None,
        cv_path=attachable,
        will_submit=False,
    )


def fill_signup_form(
    page: Any,
    candidate: Candidate,
    *,
    form: FormKnowledge | None = None,
    cv_path: Path | None = None,
    need: AccountNeed = AccountNeed.UNKNOWN,
    confirm_submit: bool = False,
) -> SignupFillResult:
    """Fill known fields from profile.yaml. Never submits.

    ``confirm_submit`` is accepted so callers can express HITL intent, but this
    driver never clicks create/submit — irreversible steps stay human.
    """
    del confirm_submit  # never used to click; kept for the HITL contract
    url = getattr(page, "url", "") or ""
    if form is None or not form.readable:
        content = page.content() if hasattr(page, "content") else ""
        form = learn_form_html(content, url=url)

    plan = build_fill_plan(candidate, url, need, form=form, cv_path=cv_path)
    filled, attached = _fill_page(page, form, candidate, plan.cv_path)

    stopped_at = "Before password, terms, CAPTCHA/2FA, and create/submit"
    if not filled and not attached:
        stopped_at = "Could not fill any fields (form may need manual completion)"

    return SignupFillResult(
        plan=plan,
        filled_fields=list(filled),
        attached=attached,
        stopped_at=stopped_at,
        submitted=False,
    )


def fill_signup_with_session(
    session: Any,
    url: str,
    candidate: Candidate,
    *,
    form: FormKnowledge | None = None,
    cv_path: Path | None = None,
    need: AccountNeed = AccountNeed.UNKNOWN,
    confirm_submit: bool = False,
) -> SignupFillResult:
    """Open signup page, fill known fields, stop before irreversible actions."""
    page = session.page
    page.goto(url, wait_until="domcontentloaded")
    try:
        page.wait_for_selector("form, input, select, textarea", timeout=5000)
    except Exception:
        logger.debug("No form elements detected immediately at %s", url)
    return fill_signup_form(
        page,
        candidate,
        form=form,
        cv_path=cv_path,
        need=need,
        confirm_submit=confirm_submit,
    )


def _is_fillable_item(item: SignupItem) -> bool:
    if item.yours_to_decide or not item.value:
        return False
    return not (_forbidden_label(item.label) or _is_resume_label(item.label))


def _fill_page(
    page: Any,
    form: FormKnowledge,
    candidate: Candidate,
    cv_path: Path | None,
) -> tuple[tuple[str, ...], bool]:
    if not form.readable:
        return (), False
    answers = {
        item.label.casefold(): item.value
        for item in signup_sheet(candidate, form=form)
        if item.value and _is_fillable_item(item)
    }
    filled: list[str] = []
    for field in form.fields:
        if field.kind is FieldKind.FILE or _blocked_field(field):
            continue
        if _is_resume_label(field.label):
            continue
        value = answers.get(field.label.casefold(), "")
        selector = _selector(field)
        if not value or selector is None:
            continue
        page.fill(selector, value)
        filled.append(field.label)
    return tuple(filled), _attach_cv(page, form, cv_path)


def _attach_cv(page: Any, form: FormKnowledge, cv_path: Path | None) -> bool:
    path = _attachable_pdf(cv_path)
    if path is None:
        return False
    files = [
        field
        for field in form.fields
        if field.kind is FieldKind.FILE and not _forbidden_label(field.label)
    ]
    if not files:
        return False
    selector = _selector(files[0])
    if selector is None:
        return False
    locator = page.locator(selector)
    target = locator.first if hasattr(locator, "first") else locator
    target.set_input_files(str(path))
    return True


def _attachable_pdf(path: Path | None) -> Path | None:
    if path is None or not path.is_file() or path.suffix.casefold() != ".pdf":
        return None
    if not path.read_bytes()[:5].startswith(b"%PDF"):
        return None
    return path


def _blocked_field(field: FormField) -> bool:
    if _forbidden_label(field.label) or _forbidden_label(field.name):
        return True
    return field.kind not in _FILLABLE and field.kind is not FieldKind.FILE


def _forbidden_label(label: str) -> bool:
    folded = label.casefold()
    return any(token in folded for token in _FORBIDDEN_TOKENS)


def _is_resume_label(label: str) -> bool:
    folded = label.casefold()
    return any(token in folded for token in _RESUME_TOKENS)


def _selector(field: FormField) -> str | None:
    """CSS for Playwright. Workday often has no name= — only automation-id / id (#92)."""
    name = field.name
    if not name or any(char in name for char in "\"'\\[]"):
        return None
    if field.kind is FieldKind.LONG_TEXT:
        return f'textarea[name="{name}"], textarea#{name}'
    return f'[name="{name}"], [data-automation-id="{name}"], #{name}'
