"""Active company portals: fill known fields, never treat a yes as a receipt (#43)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jobbot.companies.models import (
    CareerSite,
    CareerSiteType,
    CompanyRecord,
    KnowledgeStatus,
)
from jobbot.companies.registry import CompanyRegistry
from jobbot.companies.signup import AccountNeed
from jobbot.config import JobbotConfig, PathsConfig
from jobbot.cv.company_apply import (
    company_apply_intent,
    observe_company_presence,
    perform_company_apply,
)
from jobbot.cv.status import PresenceState, build_cv_status
from jobbot.cv.sync import CompanySyncRow
from jobbot.models.candidate import Candidate
from jobbot.portals.detect import AtsKind
from tests.fixtures.profile import sample_profile_dict

_FORM = """
<html><body>
<form>
  <label for="full_name">Full name</label>
  <input id="full_name" name="full_name" type="text">
  <label for="email">Email</label>
  <input id="email" name="email" type="email">
  <label for="password">Password</label>
  <input id="password" name="password" type="password">
  <label for="terms">I accept the terms</label>
  <input id="terms" name="terms" type="checkbox">
  <label for="resume">Resume/CV</label>
  <input id="resume" name="resume" type="file" accept=".pdf">
  <button type="submit">Create account</button>
</form>
</body></html>
"""

_CAREERS = "https://boards.greenhouse.io/acme"


def _candidate() -> Candidate:
    return Candidate.model_validate(sample_profile_dict())


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


def _pdf(tmp_path: Path) -> Path:
    path = tmp_path / "output" / "base" / "cv.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + b"x" * 40)
    return path


def _fill_row() -> CompanySyncRow:
    return CompanySyncRow(
        company_id="acme",
        company_name="Acme",
        url=_CAREERS,
        ats=AtsKind.GREENHOUSE,
        need=AccountNeed.NOT_NEEDED,
        action="update_profile",
        hint="fill",
        status=KnowledgeStatus.ACTIVE,
        session_evidenced=False,
    )


def _signup_row() -> CompanySyncRow:
    return CompanySyncRow(
        company_id="betterfly",
        company_name="Betterfly",
        url="https://betterfly.wd3.myworkdayjobs.com/careers",
        ats=AtsKind.WORKDAY,
        need=AccountNeed.NEEDED,
        action="needs_account",
        hint="jobbot companies signup Betterfly",
        status=KnowledgeStatus.ACTIVE,
        session_evidenced=False,
    )


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
    def __init__(self, html: str = _FORM) -> None:
        self.html = html
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


def test_signup_intent_points_at_the_sheet_and_opens_nothing() -> None:
    intent = company_apply_intent(_signup_row(), _candidate(), cv_path=None)
    assert intent.mode == "signup_sheet"
    assert any(not item.value for item in intent.sheet)
    labels = " ".join(item.label.casefold() for item in intent.sheet)
    assert "password" not in labels
    page = FakePage()
    result = perform_company_apply(
        intent,
        _candidate(),
        page=page,
        output_dir=Path("/tmp/unused-jobbot-company-apply"),
    )
    assert result.opened is False
    assert result.submitted is False
    assert result.receipt_path is None
    assert page.visited == []
    assert page.filled == {}
    assert page.clicked == []


def test_fill_known_fields_and_attach_pdf_without_submit_or_receipt(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    intent = company_apply_intent(_fill_row(), _candidate(), cv_path=pdf)
    assert intent.mode == "fill"
    page = FakePage()
    result = perform_company_apply(
        intent,
        _candidate(),
        page=page,
        output_dir=tmp_path / "output",
    )
    assert page.visited == [_CAREERS]
    assert result.submitted is False
    assert page.clicked == []
    assert page.checked == []
    blob = " ".join(f"{key}={value}" for key, value in page.filled.items()).casefold()
    assert "ana@example.com" in blob
    assert "ana ejemplo" in blob
    assert "password" not in blob
    assert "terms" not in blob
    assert page.uploaded
    assert page.uploaded[0][1] == str(pdf)
    assert result.receipt_path is None
    assert not (tmp_path / "output").joinpath("cv").exists() or result.receipt_path is None


def test_yes_without_page_evidence_does_not_write_a_receipt(tmp_path: Path) -> None:
    """The caller already confirmed. That yes is not a portal receipt."""
    pdf = _pdf(tmp_path)
    intent = company_apply_intent(_fill_row(), _candidate(), cv_path=pdf)
    page = FakePage()
    output = tmp_path / "output"
    result = perform_company_apply(intent, _candidate(), page=page, output_dir=output)
    assert result.receipt_path is None
    assert list(output.rglob("*receipt*")) == []
    assert observe_company_presence(
        page.content(),
        _candidate(),
        company_id="acme",
        url=_CAREERS,
        pdf_path=pdf,
    ) is None


def test_visible_identity_on_the_page_is_a_receipt(tmp_path: Path) -> None:
    html = "<html><body><h1>Ana Ejemplo</h1><p>Candidate home</p></body></html>"
    receipt = observe_company_presence(
        html,
        _candidate(),
        company_id="acme",
        url=_CAREERS,
    )
    assert receipt is not None
    assert "Ana Ejemplo" in receipt.evidence
    assert receipt.company_id == "acme"
    page = FakePage(html)
    intent = company_apply_intent(_fill_row(), _candidate(), cv_path=None)
    result = perform_company_apply(
        intent,
        _candidate(),
        page=page,
        output_dir=tmp_path / "output",
    )
    assert result.receipt_path is not None
    assert result.receipt_path.is_file()
    assert result.submitted is False


def test_status_unknown_without_receipt_and_present_only_from_the_page(
    tmp_path: Path, project_root: Path
) -> None:
    registry = CompanyRegistry(
        companies=[
            CompanyRecord(
                id="acme",
                name="Acme",
                career_sites=[
                    CareerSite(
                        url=_CAREERS,
                        domain="boards.greenhouse.io",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.GREENHOUSE,
                        status=KnowledgeStatus.ACTIVE,
                    )
                ],
            ),
            CompanyRecord(
                id="buk",
                name="Buk",
                career_sites=[
                    CareerSite(
                        url="https://buk.cl/trabaja-con-nosotros",
                        domain="buk.cl",
                        site_type=CareerSiteType.COMPANY_CAREER_PORTAL,
                        ats=AtsKind.UNKNOWN,
                        status=KnowledgeStatus.CANDIDATE,
                    )
                ],
            ),
            CompanyRecord(
                id="rejected-co",
                name="Rejected Co",
                career_sites=[
                    CareerSite(
                        url="https://boards.greenhouse.io/rejected",
                        domain="boards.greenhouse.io",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.GREENHOUSE,
                        status=KnowledgeStatus.REJECTED,
                    )
                ],
            ),
        ]
    )
    config = _config(tmp_path, project_root)
    report = build_cv_status(config, _candidate(), registry=registry)
    company_rows = [row for row in report.rows if row.artifact == _CAREERS]
    assert len(company_rows) == 1
    assert company_rows[0].state is PresenceState.UNKNOWN
    assert company_rows[0].state.value != "ready"
    assert "receipt" in company_rows[0].evidence.casefold()
    assert all(row.artifact != "https://buk.cl/trabaja-con-nosotros" for row in report.rows)
    assert all("rejected" not in row.artifact for row in report.rows)

    html = "<html><body><p>Signed in as Ana Ejemplo</p></body></html>"
    receipt = observe_company_presence(
        html, _candidate(), company_id="acme", url=_CAREERS
    )
    assert receipt is not None
    from jobbot.cv.company_apply import write_company_receipt

    write_company_receipt(config.output_dir, receipt)
    again = build_cv_status(config, _candidate(), registry=registry)
    present = next(row for row in again.rows if row.artifact == _CAREERS)
    assert present.state is PresenceState.PRESENT
    assert present.state.value != "ready"
    assert "Ana Ejemplo" in present.evidence
