"""Gmail compose with a real attachment (HITL: JobBot never sends)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jobbot.adapters.ats.email_apply import build_email_draft
from jobbot.adapters.gmail import selectors
from jobbot.adapters.gmail.compose import (
    GmailAuthRequired,
    GmailComposeError,
    attachment_chip_selector,
    is_auth_url,
    prepare_gmail_draft,
)
from jobbot.models.candidate import Candidate, PersonalInfo
from jobbot.models.job import JobPosting


class FakeChooser:
    def __init__(self) -> None:
        self.files: list[str] = []

    def set_files(self, files: str) -> None:
        self.files.append(files)


class FakeChooserContext:
    def __init__(self, holder: FakeChooser) -> None:
        self.value = holder

    def __enter__(self) -> FakeChooserContext:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeLocator:
    def __init__(self, page: FakePage, count: int) -> None:
        self._page = page
        self._count = count

    def count(self) -> int:
        return self._count

    @property
    def first(self) -> FakeLocator:
        return self

    def set_input_files(self, files: str) -> None:
        self._page.uploaded.append(files)


class FakePage:
    """Minimal Playwright Page stand-in recording what the adapter did."""

    def __init__(self, *, url: str = "https://mail.google.com/mail/?view=cm", file_inputs: int = 1):
        self.url = url
        self._file_inputs = file_inputs
        self.visited: list[str] = []
        self.waited: list[str] = []
        self.clicked: list[str] = []
        self.uploaded: list[str] = []
        self.missing_selectors: set[str] = set()
        self.evaluated: list[str] = []
        self.chooser = FakeChooser()

    def evaluate(self, script: str, *_args: object) -> object:
        """No attachment URL exposed → upload stays unverified, not broken."""
        self.evaluated.append(script)
        return None

    def goto(self, url: str, **_: Any) -> None:
        self.visited.append(url)

    def wait_for_selector(self, selector: str, **_: Any) -> None:
        if selector in self.missing_selectors:
            msg = f"timeout waiting for {selector}"
            raise TimeoutError(msg)
        self.waited.append(selector)

    def locator(self, _selector: str) -> FakeLocator:
        return FakeLocator(self, self._file_inputs)

    def click(self, selector: str, **_: Any) -> None:
        self.clicked.append(selector)

    def expect_file_chooser(self, **_: Any) -> FakeChooserContext:
        return FakeChooserContext(self.chooser)


def _draft(cv: Path | None):
    candidate = Candidate(
        personal=PersonalInfo(name="Ana Ejemplo", headline="Data Scientist"),
    )
    job = JobPosting(
        id="J0005",
        title="Senior Data Scientist",
        company="Empresa Acequia",
        description="Enviar CV a seleccion@empresa.cl",
        ats_url="mailto:seleccion@empresa.cl",
        ats_kind="email",
    )
    return build_email_draft(candidate, job, cv_path=cv)


def _cv(tmp_path: Path) -> Path:
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.4 fake")
    return cv


def test_prepare_gmail_draft_uploads_cv_via_file_input(tmp_path: Path) -> None:
    """Regression: the CV must be attached, not just named in the body."""
    cv = _cv(tmp_path)
    page = FakePage(file_inputs=1)
    url = prepare_gmail_draft(page, _draft(cv))

    assert page.visited == [url]
    assert page.uploaded == [str(cv)]
    assert attachment_chip_selector("cv.pdf") in page.waited


def test_prepare_gmail_draft_waits_for_standalone_compose(tmp_path: Path) -> None:
    """Regression: `fs=1` compose is full-window, never a role="dialog"."""
    page = FakePage(file_inputs=1)
    prepare_gmail_draft(page, _draft(_cv(tmp_path)))

    assert selectors.COMPOSE_READY in page.waited
    assert not any('role="dialog"' in waited for waited in page.waited)


def test_prepare_gmail_draft_never_clicks_send(tmp_path: Path) -> None:
    page = FakePage(file_inputs=1)
    prepare_gmail_draft(page, _draft(_cv(tmp_path)))
    assert selectors.SEND_BUTTON not in page.clicked


def test_prepare_gmail_draft_falls_back_to_file_chooser(tmp_path: Path) -> None:
    cv = _cv(tmp_path)
    page = FakePage(file_inputs=0)
    prepare_gmail_draft(page, _draft(cv))

    assert page.uploaded == []
    assert page.clicked == [selectors.ATTACH_BUTTON]
    assert page.chooser.files == [str(cv)]


def test_prepare_gmail_draft_requires_existing_cv(tmp_path: Path) -> None:
    page = FakePage()
    with pytest.raises(GmailComposeError, match="not found"):
        prepare_gmail_draft(page, _draft(tmp_path / "missing.pdf"))
    assert page.visited == []


def test_prepare_gmail_draft_detects_login(tmp_path: Path) -> None:
    page = FakePage(url="https://accounts.google.com/signin/v2/identifier")
    with pytest.raises(GmailAuthRequired):
        prepare_gmail_draft(page, _draft(_cv(tmp_path)))


def test_prepare_gmail_draft_errors_when_attachment_not_confirmed(tmp_path: Path) -> None:
    cv = _cv(tmp_path)
    page = FakePage(file_inputs=1)
    page.missing_selectors.add(attachment_chip_selector("cv.pdf"))
    with pytest.raises(GmailComposeError, match="cv.pdf"):
        prepare_gmail_draft(page, _draft(cv))


def test_is_auth_url() -> None:
    assert is_auth_url("https://accounts.google.com/ServiceLogin")
    assert not is_auth_url("https://mail.google.com/mail/u/0/#inbox")


class _VerifyPage(FakePage):
    """Fake page whose evaluate() returns queued values (url, then fetch info)."""

    def __init__(self, results: list[object]) -> None:
        super().__init__()
        self._results = list(results)

    def evaluate(self, script: str, *_args: object) -> object:
        self.evaluated.append(script)
        return self._results.pop(0) if self._results else None


def test_verify_attachment_upload_accepts_matching_bytes(tmp_path: Path) -> None:
    from jobbot.adapters.gmail.compose import verify_attachment_upload

    cv = _cv(tmp_path)
    page = _VerifyPage(
        [
            "https://mail.google.com/mail/u/0?view=att&attid=0.1&disp=safe",
            {"status": 200, "size": cv.stat().st_size, "head": "%PDF-1.7", "tail": "%%EOF\n"},
        ]
    )
    assert verify_attachment_upload(page, cv) is True


def test_verify_attachment_upload_rejects_truncated_upload(tmp_path: Path) -> None:
    """Regression: an interrupted run left a draft whose attachment was incomplete."""
    from jobbot.adapters.gmail.compose import verify_attachment_upload

    cv = _cv(tmp_path)
    page = _VerifyPage(
        [
            "https://mail.google.com/mail/u/0?view=att&attid=0.1&disp=safe",
            {"status": 200, "size": 12, "head": "%PDF-1.7", "tail": "trunc"},
        ]
    )
    with pytest.raises(GmailComposeError, match="12 bytes"):
        verify_attachment_upload(page, cv)


def test_verify_attachment_upload_rejects_non_pdf_bytes(tmp_path: Path) -> None:
    from jobbot.adapters.gmail.compose import verify_attachment_upload

    cv = _cv(tmp_path)
    page = _VerifyPage(
        [
            "https://mail.google.com/mail/u/0?view=att&attid=0.1&disp=safe",
            {"status": 200, "size": cv.stat().st_size, "head": "<!DOCTYPE", "tail": "html>"},
        ]
    )
    with pytest.raises(GmailComposeError, match="not a PDF"):
        verify_attachment_upload(page, cv)


def test_verify_attachment_upload_unverified_without_url(tmp_path: Path) -> None:
    from jobbot.adapters.gmail.compose import verify_attachment_upload

    page = _VerifyPage([None])
    assert verify_attachment_upload(page, _cv(tmp_path)) is False


def test_discard_compose_clicks_discard_not_send(tmp_path: Path) -> None:
    """Failed runs must not leave half-uploaded drafts behind to be sent later."""
    from jobbot.adapters.gmail.compose import discard_compose

    page = FakePage()
    assert discard_compose(page) is True
    assert page.clicked == [selectors.DISCARD_BUTTON]
    assert selectors.SEND_BUTTON not in page.clicked
