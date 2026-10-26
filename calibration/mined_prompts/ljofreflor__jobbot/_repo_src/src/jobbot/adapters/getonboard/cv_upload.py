"""Upload the local CV PDF to Get on Board 'Tus CVs', after validating it.

Validation is local and cheap: existence, PDF magic bytes, size ≤ 5 MB, hash for
the confirmation prompt. The browser step only runs after that passes, and after
the user confirms (or ``--yes``). JobBot never invents a CV: it uploads the file
``cv build`` already produced.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

logger = logging.getLogger("jobbot.getonboard.cv_upload")

RESUMES_URL = "https://www.getonbrd.com/resumes"
RESUMES_NEW_URL = "https://www.getonbrd.com/resumes/new"
FILE_INPUT = "#resume_file"
LABEL_INPUT = "#resume_label"
DEFAULT_CHECKBOX = "#resume_is_default"
SUBMIT_BUTTON = 'input[type="submit"][value="Subir"]'
RECEIPT_NAME = "cv_upload_receipt.yaml"

# Documented by Get on Board on the upload page / sync sheet.
MAX_BYTES = 5 * 1024 * 1024
PDF_MAGIC = b"%PDF"


class GobCvUploadError(RuntimeError):
    """The CV could not be validated or the portal did not accept it."""


@dataclass(frozen=True)
class CvUploadCheck:
    """What we know about the local file before touching the portal."""

    path: Path
    ok: bool
    size_bytes: int = 0
    sha256: str = ""
    reasons: tuple[str, ...] = ()

    def summary_lines(self) -> list[str]:
        lines = [
            f"path:   {self.path}",
            f"size:   {self.size_bytes} bytes ({self.size_bytes / 1024:.1f} KiB)",
            f"sha256: {self.sha256 or '—'}",
            f"limit:  {MAX_BYTES} bytes (Get on Board)",
        ]
        if self.reasons:
            lines.append("blockers:")
            lines.extend(f"  • {reason}" for reason in self.reasons)
        return lines


@dataclass(frozen=True)
class CvUploadResult:
    """What the portal shows after a successful upload."""

    label: str
    path: Path
    listed: bool
    is_default: bool
    resumes_url: str = RESUMES_URL


@dataclass(frozen=True)
class CvUploadReceipt:
    """Local evidence that an upload completed (never invent remote state)."""

    sha256: str
    label: str
    path: str
    is_default: bool
    listed: bool
    uploaded_at: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "label": self.label,
            "path": self.path,
            "is_default": self.is_default,
            "listed": self.listed,
            "uploaded_at": (self.uploaded_at.isoformat() if self.uploaded_at is not None else None),
        }


def upload_receipt_path(output_dir: Path) -> Path:
    return output_dir / "getonboard" / RECEIPT_NAME


def write_upload_receipt(
    output_dir: Path,
    check: CvUploadCheck,
    result: CvUploadResult,
    *,
    uploaded_at: datetime | None = None,
) -> Path:
    """Persist hash/label after a successful portal upload so ``status`` can compare."""
    when = uploaded_at or datetime.now(UTC)
    receipt = CvUploadReceipt(
        sha256=check.sha256,
        label=result.label,
        path=str(result.path),
        is_default=result.is_default,
        listed=result.listed,
        uploaded_at=when,
    )
    path = upload_receipt_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(receipt.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_upload_receipt(output_dir: Path) -> CvUploadReceipt | None:
    """Read the last upload receipt, or None when JobBot has never recorded one."""
    path = upload_receipt_path(output_dir)
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    sha = str(raw.get("sha256") or "").strip()
    if not sha:
        return None
    when_raw = raw.get("uploaded_at")
    when: datetime | None = None
    if isinstance(when_raw, datetime):
        when = when_raw
    elif isinstance(when_raw, str) and when_raw.strip():
        when = datetime.fromisoformat(when_raw)
    return CvUploadReceipt(
        sha256=sha,
        label=str(raw.get("label") or ""),
        path=str(raw.get("path") or ""),
        is_default=bool(raw.get("is_default")),
        listed=bool(raw.get("listed", True)),
        uploaded_at=when,
    )


class PageLike(Protocol):
    def goto(self, url: str, **kwargs: Any) -> Any: ...
    def wait_for_timeout(self, timeout: float) -> None: ...
    def locator(self, selector: str) -> Any: ...
    def fill(self, selector: str, value: str) -> None: ...
    def check(self, selector: str) -> None: ...
    def click(self, selector: str, **kwargs: Any) -> None: ...
    def content(self) -> str: ...
    @property
    def url(self) -> str: ...


def validate_cv_for_upload(path: Path) -> CvUploadCheck:
    """Inspect the local PDF. Pure: no network, no browser."""
    reasons: list[str] = []
    if not path.exists():
        return CvUploadCheck(path=path, ok=False, reasons=("file not found",))
    if not path.is_file():
        return CvUploadCheck(path=path, ok=False, reasons=("not a regular file",))

    size = path.stat().st_size
    if size <= 0:
        reasons.append("file is empty")
    if size > MAX_BYTES:
        reasons.append(f"file is {size} bytes; Get on Board allows at most {MAX_BYTES}")

    suffix = path.suffix.casefold()
    if suffix != ".pdf":
        reasons.append(f"extension is {suffix or '(none)'}; only .pdf is uploaded")

    head = path.read_bytes()[:8]
    if not head.startswith(PDF_MAGIC):
        reasons.append(f"does not start with %PDF (got {head!r})")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return CvUploadCheck(
        path=path,
        ok=not reasons,
        size_bytes=size,
        sha256=digest,
        reasons=tuple(reasons),
    )


def resolve_base_cv(output_dir: Path) -> Path:
    """The PDF ``cv build`` / ``cv propagate`` writes."""
    preferred = output_dir / "base" / "cv.pdf"
    if preferred.is_file():
        return preferred
    fallback = output_dir / "cv.pdf"
    return preferred if not fallback.is_file() else fallback


def upload_cv(
    page: PageLike,
    path: Path,
    *,
    label: str = "CV base",
    make_default: bool = True,
    check: CvUploadCheck | None = None,
) -> CvUploadResult:
    """Attach a validated PDF and submit the Get on Board resume form.

    Raises if the local check fails: the file is never sent when validation fails.
    """
    verified = check or validate_cv_for_upload(path)
    if not verified.ok:
        msg = "CV rejected before upload: " + "; ".join(verified.reasons)
        raise GobCvUploadError(msg)

    page.goto(RESUMES_NEW_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(1000)
    file_input = page.locator(FILE_INPUT)
    if hasattr(file_input, "count") and file_input.count() == 0:
        msg = f"upload form not found on {RESUMES_NEW_URL}"
        raise GobCvUploadError(msg)
    target = file_input.first if hasattr(file_input, "first") else file_input
    target.set_input_files(str(path))
    page.fill(LABEL_INPUT, label)
    if make_default:
        page.check(DEFAULT_CHECKBOX)
    page.click(SUBMIT_BUTTON)
    page.wait_for_timeout(2000)

    # Land on the list (or stay on new with an error); confirm the label is there.
    if "/resumes/new" in (page.url or ""):
        page.goto(RESUMES_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
    body = page.content()
    listed = label.casefold() in body.casefold() or path.name.casefold() in body.casefold()
    if not listed:
        msg = f"uploaded {path.name} but '{label}' did not appear on {RESUMES_URL}"
        raise GobCvUploadError(msg)
    # When a resume is already the default, GoB stops offering "Establecer como
    # predeterminado" next to it. That absence is the evidence we use.
    actions = _item_actions(body, label)
    is_default = make_default and "establecer como predeterminado" not in actions
    return CvUploadResult(
        label=label,
        path=path,
        listed=listed,
        is_default=is_default,
    )


def _item_actions(html: str, label: str) -> str:
    """The resume card that mentions ``label``, for default evidence.

    Looking at a window that spills into the next card would falsely see
    "Establecer como predeterminado" belonging to a neighbour.
    """
    folded = html.casefold()
    needle = label.casefold()
    for chunk in folded.split("resume-item"):
        if needle in chunk:
            return chunk
    start = folded.find(needle)
    return "" if start < 0 else folded[start : start + 200]


def list_resume_labels(page: PageLike) -> list[str]:
    """Labels currently on Tus CVs (best-effort from the list page)."""
    page.goto(RESUMES_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(800)
    # Buttons that rename a resume carry the current label as their text.
    return (
        page.locator(".resume-label").all_inner_texts()
        if hasattr(page.locator(".resume-label"), "all_inner_texts")
        else []
    )
