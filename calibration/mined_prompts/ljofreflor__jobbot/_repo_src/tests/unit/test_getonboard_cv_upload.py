"""Get on Board CV upload: validate locally, then attach — never the other way around."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jobbot.adapters.getonboard.cv_upload import (
    DEFAULT_CHECKBOX,
    FILE_INPUT,
    LABEL_INPUT,
    MAX_BYTES,
    RESUMES_NEW_URL,
    RESUMES_URL,
    SUBMIT_BUTTON,
    GobCvUploadError,
    resolve_base_cv,
    upload_cv,
    validate_cv_for_upload,
)


def _pdf(tmp_path: Path, name: str = "cv.pdf", *, size: int = 1200) -> Path:
    path = tmp_path / name
    # Minimal PDF header + padding so size checks are realistic.
    path.write_bytes(b"%PDF-1.4\n" + b"%" + b"x" * max(0, size - 10))
    return path


class FakeLocator:
    def __init__(self, page: FakePage, selector: str) -> None:
        self._page = page
        self._selector = selector

    def count(self) -> int:
        return 1 if self._selector == FILE_INPUT else 0

    @property
    def first(self) -> FakeLocator:
        return self

    def set_input_files(self, files: str) -> None:
        self._page.uploaded.append(files)

    def all_inner_texts(self) -> list[str]:
        return list(self._page.labels)


class FakePage:
    def __init__(self) -> None:
        self.visited: list[str] = []
        self.uploaded: list[str] = []
        self.filled: dict[str, str] = {}
        self.checked: list[str] = []
        self.clicked: list[str] = []
        self.labels: list[str] = ["otro cv"]
        self.url = RESUMES_NEW_URL
        self._after_submit_html = ""

    def goto(self, url: str, **_: Any) -> None:
        self.visited.append(url)
        self.url = url

    def wait_for_timeout(self, _timeout: float) -> None:
        return None

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self, selector)

    def fill(self, selector: str, value: str) -> None:
        self.filled[selector] = value

    def check(self, selector: str) -> None:
        self.checked.append(selector)

    def click(self, selector: str, **_: Any) -> None:
        self.clicked.append(selector)
        if selector == SUBMIT_BUTTON:
            label = self.filled.get(LABEL_INPUT, "")
            self.labels.insert(0, label)
            # Default resume: no "Establecer como predeterminado" next to it.
            self._after_submit_html = (
                f'<div class="resume-item">{label} Subido hoy</div>'
                f'<div class="resume-item">otro cv Establecer como predeterminado</div>'
            )
            self.url = RESUMES_URL

    def content(self) -> str:
        return self._after_submit_html or "<html></html>"


def test_a_real_pdf_under_the_limit_passes(tmp_path: Path) -> None:
    path = _pdf(tmp_path)
    check = validate_cv_for_upload(path)

    assert check.ok
    assert check.size_bytes == path.stat().st_size
    assert len(check.sha256) == 64
    assert check.reasons == ()


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    check = validate_cv_for_upload(tmp_path / "nope.pdf")

    assert not check.ok
    assert "not found" in check.reasons[0]


def test_non_pdf_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cv.docx"
    path.write_bytes(b"%PDF-1.4\nfake")

    check = validate_cv_for_upload(path)

    assert not check.ok
    assert any(".pdf" in reason for reason in check.reasons)


def test_wrong_magic_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(b"PK\x03\x04not-a-pdf")

    check = validate_cv_for_upload(path)

    assert not check.ok
    assert any("%PDF" in reason for reason in check.reasons)


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    path = _pdf(tmp_path, size=MAX_BYTES + 50)

    check = validate_cv_for_upload(path)

    assert not check.ok
    assert any(str(MAX_BYTES) in reason for reason in check.reasons)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(b"")

    check = validate_cv_for_upload(path)

    assert not check.ok


def test_upload_never_sends_a_file_that_failed_validation(tmp_path: Path) -> None:
    page = FakePage()
    bad = tmp_path / "cv.pdf"
    bad.write_bytes(b"not-pdf")

    with pytest.raises(GobCvUploadError, match="before upload"):
        upload_cv(page, bad, label="CV base")

    assert page.uploaded == []
    assert page.visited == []


def test_upload_attaches_only_after_validation_and_submits(tmp_path: Path) -> None:
    path = _pdf(tmp_path)
    page = FakePage()
    check = validate_cv_for_upload(path)

    result = upload_cv(page, path, label="CV base", make_default=True, check=check)

    assert page.visited[0] == RESUMES_NEW_URL
    assert page.uploaded == [str(path)]
    assert page.filled[LABEL_INPUT] == "CV base"
    assert DEFAULT_CHECKBOX in page.checked
    assert SUBMIT_BUTTON in page.clicked
    assert result.listed
    assert result.is_default
    assert result.label == "CV base"


def test_upload_can_skip_marking_default(tmp_path: Path) -> None:
    path = _pdf(tmp_path)
    page = FakePage()

    upload_cv(page, path, label="CV experimental", make_default=False)

    assert DEFAULT_CHECKBOX not in page.checked


def test_resolve_base_cv_prefers_output_base(tmp_path: Path) -> None:
    preferred = tmp_path / "base" / "cv.pdf"
    preferred.parent.mkdir()
    preferred.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "cv.pdf").write_bytes(b"%PDF-1.4\nold")

    assert resolve_base_cv(tmp_path) == preferred


def test_summary_lines_include_hash_for_the_confirm_prompt(tmp_path: Path) -> None:
    check = validate_cv_for_upload(_pdf(tmp_path))
    lines = "\n".join(check.summary_lines())

    assert check.sha256 in lines
    assert str(check.path) in lines


def test_upload_receipt_round_trip(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from jobbot.adapters.getonboard.cv_upload import (
        CvUploadResult,
        load_upload_receipt,
        write_upload_receipt,
    )

    path = _pdf(tmp_path)
    check = validate_cv_for_upload(path)
    write_upload_receipt(
        tmp_path,
        check,
        CvUploadResult(label="CV base", path=path, listed=True, is_default=True),
        uploaded_at=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
    )
    receipt = load_upload_receipt(tmp_path)
    assert receipt is not None
    assert receipt.sha256 == check.sha256
    assert receipt.label == "CV base"
    assert receipt.is_default is True
