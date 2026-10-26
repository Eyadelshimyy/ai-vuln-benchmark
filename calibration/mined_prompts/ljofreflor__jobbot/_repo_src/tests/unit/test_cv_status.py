"""Permanent CV presence report for `jobbot status` (evidence only)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jobbot.adapters.diff_engine import save_snapshot
from jobbot.adapters.getonboard.cv_upload import (
    CvUploadCheck,
    CvUploadResult,
    write_upload_receipt,
)
from jobbot.config import JobbotConfig, PathsConfig
from jobbot.cv.status import PresenceState, build_cv_status
from jobbot.models.candidate import Candidate
from jobbot.models.external_profile import ExternalProfile
from tests.fixtures.profile import sample_profile_dict


def _candidate() -> Candidate:
    return Candidate.model_validate(sample_profile_dict())


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


def _pdf(root: Path, *, payload: bytes = b"%PDF-1.4\n" + b"x" * 200) -> Path:
    path = root / "output" / "base" / "cv.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _row(report: object, destination: str, *, artifact: str | None = None):
    rows = [r for r in report.rows if r.destination == destination]  # type: ignore[attr-defined]
    if artifact is not None:
        rows = [r for r in rows if r.artifact == artifact]
    assert len(rows) == 1, f"expected one row for {destination!r}/{artifact!r}, got {rows}"
    return rows[0]


def test_local_missing_when_no_pdf(tmp_path: Path, project_root: Path) -> None:
    report = build_cv_status(_config(tmp_path, project_root), _candidate())
    local = _row(report, "local")
    assert local.state is PresenceState.MISSING
    assert local.hint is not None and "cv build" in local.hint


def test_local_present_includes_stable_sha256(tmp_path: Path, project_root: Path) -> None:
    payload = b"%PDF-1.4\n" + b"y" * 300
    path = _pdf(tmp_path, payload=payload)
    report = build_cv_status(_config(tmp_path, project_root), _candidate())
    local = _row(report, "local")
    assert local.state is PresenceState.PRESENT
    assert path.name in local.artifact or "cv.pdf" in local.artifact
    digest = hashlib.sha256(payload).hexdigest()
    assert digest[:12] in local.evidence or digest in local.evidence


def test_gob_cv_unknown_without_receipt(tmp_path: Path, project_root: Path) -> None:
    _pdf(tmp_path)
    report = build_cv_status(_config(tmp_path, project_root), _candidate())
    gob = _row(report, "getonboard", artifact="Tus CVs")
    assert gob.state is PresenceState.UNKNOWN
    assert "receipt" in gob.evidence.casefold() or "no evidence" in gob.evidence.casefold()


def test_gob_cv_in_sync_when_receipt_matches_pdf(tmp_path: Path, project_root: Path) -> None:
    path = _pdf(tmp_path)
    config = _config(tmp_path, project_root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    check = CvUploadCheck(
        path=path,
        ok=True,
        size_bytes=path.stat().st_size,
        sha256=digest,
    )
    write_upload_receipt(
        config.output_dir,
        check,
        CvUploadResult(label="CV base", path=path, listed=True, is_default=True),
        uploaded_at=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
    )
    report = build_cv_status(config, _candidate())
    gob = _row(report, "getonboard", artifact="Tus CVs")
    assert gob.state is PresenceState.IN_SYNC
    assert "CV base" in gob.evidence


def test_gob_cv_stale_when_receipt_hash_differs(tmp_path: Path, project_root: Path) -> None:
    path = _pdf(tmp_path)
    config = _config(tmp_path, project_root)
    check = CvUploadCheck(
        path=path,
        ok=True,
        size_bytes=10,
        sha256="0" * 64,
    )
    write_upload_receipt(
        config.output_dir,
        check,
        CvUploadResult(label="CV base", path=path, listed=True, is_default=True),
    )
    report = build_cv_status(config, _candidate())
    gob = _row(report, "getonboard", artifact="Tus CVs")
    assert gob.state is PresenceState.STALE


def test_indeed_blocked_without_snapshot(tmp_path: Path, project_root: Path) -> None:
    report = build_cv_status(_config(tmp_path, project_root), _candidate())
    indeed = _row(report, "indeed")
    assert indeed.state is PresenceState.BLOCKED
    assert indeed.hint == "jobbot indeed pull"


def test_indeed_behind_when_sync_ops_pending(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    save_snapshot(
        ExternalProfile(source="indeed", headline="Old headline", summary="Old summary"),
        config.output_dir,
    )
    report = build_cv_status(config, _candidate())
    indeed = _row(report, "indeed")
    assert indeed.state is PresenceState.BEHIND


def test_json_keys_are_stable(tmp_path: Path, project_root: Path) -> None:
    report = build_cv_status(_config(tmp_path, project_root), _candidate())
    payload = report.to_dict()
    assert "rows" in payload
    row = payload["rows"][0]
    assert set(row) >= {"destination", "state", "artifact", "evidence", "hint"}


def test_build_cv_status_writes_nothing(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    build_cv_status(config, _candidate())
    after = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def test_status_cli_prints_table(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _pdf(tmp_path)
    from jobbot.cli import run_cli
    from jobbot.exit_codes import SUCCESS

    assert run_cli(["status"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert "local" in out.casefold()
    assert "getonboard" in out.casefold()


def test_cv_status_is_an_alias(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    from jobbot.cli import run_cli
    from jobbot.exit_codes import SUCCESS

    assert run_cli(["cv", "status", "--json"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert '"destination"' in out
