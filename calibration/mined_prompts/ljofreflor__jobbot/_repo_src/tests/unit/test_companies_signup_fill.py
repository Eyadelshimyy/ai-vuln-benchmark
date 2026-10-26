"""Assisted signup fill: HITL, stops before irreversible actions (#44)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jobbot.adapters.ats.signup_fill import (
    build_fill_plan,
    fill_signup_form,
)
from jobbot.companies.models import CareerSite, CareerSiteType, KnowledgeStatus
from jobbot.companies.signup import AccountNeed, signup_target
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind
from jobbot.portals.form_learn import FieldKind, FormField, FormKnowledge, learn_form_html
from tests.fixtures.profile import sample_profile_dict

FORM_FIXTURE = Path("tests/fixtures/forms/greenhouse_apply.html")

_SIGNUP_FORM = """
<html><body>
<form>
  <label for="full_name">Full name</label>
  <input id="full_name" name="full_name" type="text">
  <label for="email">Email</label>
  <input id="email" name="email" type="email">
  <label for="phone">Phone</label>
  <input id="phone" name="phone" type="tel">
  <label for="password">Password</label>
  <input id="password" name="password" type="password">
  <label for="terms">I agree to terms and conditions</label>
  <input id="terms" name="terms" type="checkbox">
  <label for="resume">Resume/CV</label>
  <input id="resume" name="resume" type="file" accept=".pdf">
  <button type="submit">Create account</button>
</form>
</body></html>
"""


def _candidate() -> Candidate:
    return Candidate.model_validate(sample_profile_dict())


def _site(url: str, ats: AtsKind) -> CareerSite:
    return CareerSite(
        url=url,
        domain=url.split("//", 1)[-1].split("/", 1)[0],
        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
        ats=ats,
        status=KnowledgeStatus.ACTIVE,
    )


def _pdf(tmp_path: Path) -> Path:
    path = tmp_path / "output" / "base" / "cv.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + b"x" * 40)
    return path


class _Locator:
    def __init__(self, page: FakePage, selector: str) -> None:
        self._page = page
        self._selector = selector

    @property
    def first(self) -> _Locator:
        return self

    def set_input_files(self, files: str) -> None:
        self._page.uploaded.append((self._selector, files))


class FakePage:
    def __init__(self, html: str = _SIGNUP_FORM, url: str = "https://example.com/signup") -> None:
        self.html = html
        self.url = url
        self.visited: list[str] = []
        self.filled: dict[str, str] = {}
        self.uploaded: list[tuple[str, str]] = []
        self.clicked: list[str] = []
        self.checked: list[str] = []

    def goto(self, url: str, **_: Any) -> None:
        self.visited.append(url)

    def content(self) -> str:
        return self.html

    def fill(self, selector: str, value: str) -> None:
        self.filled[selector] = value

    def click(self, selector: str, **_: Any) -> None:
        self.clicked.append(selector)

    def check(self, selector: str) -> None:
        self.checked.append(selector)

    def locator(self, selector: str) -> _Locator:
        return _Locator(self, selector)

    def wait_for_selector(self, *_a: Any, **_k: Any) -> None:
        return None


def test_fill_plan_maps_profile_fields_to_fillable() -> None:
    """Fixture form with name/email/phone → fill plan maps to profile fields."""
    form = FormKnowledge(
        url="https://example.com/signup",
        readable=True,
        fields=[
            FormField(name="email", label="Email", kind=FieldKind.EMAIL, required=True),
            FormField(name="full_name", label="Full name", kind=FieldKind.TEXT, required=True),
            FormField(name="phone", label="Phone", kind=FieldKind.PHONE, required=False),
        ],
    )

    plan = build_fill_plan(
        _candidate(),
        "https://example.com/signup",
        AccountNeed.NEEDED,
        form=form,
    )

    assert "Email" in plan.fields_to_fill
    assert "Full name" in plan.fields_to_fill
    assert "Phone" in plan.needs_manual
    assert plan.will_submit is False


def test_password_is_never_fillable() -> None:
    """A password field must be absent from the fill plan."""
    form = FormKnowledge(
        url="https://example.com/signup",
        readable=True,
        fields=[
            FormField(name="email", label="Email", kind=FieldKind.EMAIL, required=True),
            FormField(name="password", label="Password", kind=FieldKind.TEXT, required=True),
        ],
    )

    plan = build_fill_plan(
        _candidate(),
        "https://example.com/signup",
        AccountNeed.NEEDED,
        form=form,
    )

    assert "Password" in plan.needs_manual
    assert "Password" not in plan.fields_to_fill
    assert all("password" not in label.casefold() for label in plan.fields_to_fill)


def test_apply_without_confirm_does_not_submit() -> None:
    """--apply without confirm does not submit (driver never clicks create)."""
    page = FakePage()
    result = fill_signup_form(
        page,
        _candidate(),
        form=None,
        confirm_submit=False,
    )

    assert result.submitted is False
    assert result.plan.will_submit is False
    assert page.clicked == []
    assert page.checked == []
    manual_text = " ".join(result.plan.needs_manual).casefold()
    assert "submit" in manual_text or "create" in manual_text


def test_confirm_true_still_does_not_auto_submit() -> None:
    """Even with HITL confirm, create/submit stays human — never auto-clicked."""
    page = FakePage()
    result = fill_signup_form(
        page,
        _candidate(),
        confirm_submit=True,
    )

    assert result.submitted is False
    assert page.clicked == []
    blob = " ".join(f"{key}={value}" for key, value in page.filled.items()).casefold()
    assert "ana@example.com" in blob
    assert "ana ejemplo" in blob
    assert "password" not in blob
    assert "terms" not in blob


def test_fills_known_fields_and_may_attach_cv(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    page = FakePage()
    result = fill_signup_form(page, _candidate(), cv_path=pdf)

    assert result.attached is True
    assert page.uploaded
    assert page.uploaded[0][1] == str(pdf)
    assert "Email" in result.filled_fields or any(
        "email" in label.casefold() for label in result.filled_fields
    )
    assert result.submitted is False


def test_learned_form_fixture_maps_without_password(project_root: Path) -> None:
    html = (project_root / FORM_FIXTURE).read_text(encoding="utf-8")
    form = learn_form_html(html, url="https://boards.greenhouse.io/acme/jobs/4001")
    plan = build_fill_plan(
        _candidate(),
        form.url,
        AccountNeed.NOT_NEEDED,
        form=form,
    )

    assert form.fields
    assert any(field.kind == FieldKind.FILE for field in form.fields)
    assert all("password" not in label.casefold() for label in plan.fields_to_fill)


def test_cv_attachment_requires_real_pdf(tmp_path: Path) -> None:
    plan_none = build_fill_plan(
        _candidate(),
        "https://example.com/signup",
        AccountNeed.NEEDED,
        cv_path=None,
    )
    assert not plan_none.can_attach_cv

    fake = tmp_path / "missing.pdf"
    plan_fake = build_fill_plan(
        _candidate(),
        "https://example.com/signup",
        AccountNeed.NEEDED,
        cv_path=fake,
    )
    assert not plan_fake.can_attach_cv

    not_pdf = tmp_path / "cv.pdf"
    not_pdf.write_text("not a pdf", encoding="utf-8")
    plan_bad = build_fill_plan(
        _candidate(),
        "https://example.com/signup",
        AccountNeed.NEEDED,
        cv_path=not_pdf,
    )
    assert not plan_bad.can_attach_cv

    good = _pdf(tmp_path)
    plan_ok = build_fill_plan(
        _candidate(),
        "https://example.com/signup",
        AccountNeed.NEEDED,
        cv_path=good,
    )
    assert plan_ok.can_attach_cv
    assert plan_ok.cv_path == good


def test_terms_and_consent_never_filled() -> None:
    form = FormKnowledge(
        url="https://example.com/signup",
        readable=True,
        fields=[
            FormField(name="email", label="Email", kind=FieldKind.EMAIL, required=True),
            FormField(
                name="terms",
                label="I agree to terms and conditions",
                kind=FieldKind.CHECKBOX,
                required=True,
            ),
        ],
    )
    plan = build_fill_plan(
        _candidate(),
        "https://example.com/signup",
        AccountNeed.NEEDED,
        form=form,
    )

    assert "Email" in plan.fields_to_fill
    assert any("terms" in manual.casefold() for manual in plan.needs_manual)
    assert all("agree" not in label.casefold() for label in plan.fields_to_fill)


def test_greenhouse_lever_ashby_stay_not_needed() -> None:
    for ats in (AtsKind.GREENHOUSE, AtsKind.LEVER, AtsKind.ASHBY):
        target = signup_target(_site("https://example.com/careers", ats))
        plan = build_fill_plan(
            _candidate(),
            target.url,
            target.need,
            form=None,
        )
        assert plan.need is AccountNeed.NOT_NEEDED
        assert target.need is AccountNeed.NOT_NEEDED


def test_workday_create_account_fill_uses_automation_id(tmp_path: Path) -> None:
    """Workday email has no name= — only data-automation-id / id (#92)."""
    html = Path("tests/fixtures/forms/workday_create_account.html").read_text(
        encoding="utf-8"
    )
    page = FakePage(
        html=html,
        url=(
            "https://example.wd5.myworkdayjobs.com/en-US/External_Career/"
            "job/Remote/Role_1/apply/applyManually"
        ),
    )
    result = fill_signup_form(page, _candidate(), cv_path=_pdf(tmp_path))

    assert result.submitted is False
    assert "Email Address" in result.filled_fields
    assert not any("password" in f.casefold() for f in result.filled_fields)
    assert any(
        "data-automation-id" in sel or sel.startswith("#") for sel in page.filled
    )
    assert page.clicked == []


def test_bare_name_does_not_fill_middle_or_family_name_fields() -> None:
    """Substring 'name' must not overwrite Workday's split name fields."""
    from jobbot.companies.signup import signup_sheet

    form = FormKnowledge(
        url="https://example.wd5.myworkdayjobs.com/x",
        readable=True,
        fields=[
            FormField(name="middle", label="Middle Name", kind=FieldKind.TEXT),
            FormField(name="father", label="Father's Family Name", kind=FieldKind.TEXT),
            FormField(name="email", label="Email Address", kind=FieldKind.EMAIL),
            FormField(name="city", label="City", kind=FieldKind.TEXT),
        ],
    )
    sheet = {item.label: item for item in signup_sheet(_candidate(), form=form)}

    assert sheet["Middle Name"].value == ""
    assert sheet["Father's Family Name"].value == ""
    assert sheet["Email Address"].value == str(_candidate().personal.email)
    assert sheet["City"].value == (_candidate().personal.city or "")
