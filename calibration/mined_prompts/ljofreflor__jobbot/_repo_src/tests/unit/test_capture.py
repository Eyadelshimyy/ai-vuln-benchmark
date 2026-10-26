"""Capture share URLs as unfinished candidates (phone / Cursor mobile)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE
from jobbot.jobs.capture import CaptureKind, capture_url, list_candidates
from jobbot.jobs.inbox import list_parked


def _config(tmp_path: Path) -> JobbotConfig:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=Path("templates"), output=Path("output")),
    )


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_capture_indeed_hard_link(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = capture_url(
        config,
        "https://cl.indeed.com/viewjob?jk=05b1c90d67b4bd86&from=appshareios",
    )
    assert result.kind == CaptureKind.HARD_LINK
    assert result.url == "https://cl.indeed.com/viewjob?jk=05b1c90d67b4bd86"
    assert list_parked(config) == [result.url]


def test_capture_company_portal_as_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = capture_url(
        config,
        "https://www.empleo.ubimia.com/",
        company="Ubimia",
        country="CL",
    )
    assert result.kind == CaptureKind.COMPANY_PORTAL
    assert result.company_id == "ubimia"
    inv = list_candidates(config)
    assert any(cid == "ubimia" for cid, _, _ in inv.company_portals)
    assert inv.hard_links == []


def test_capture_unknown_stays_unrecognized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = capture_url(config, "https://totally-unknown.example/path")
    assert result.kind == CaptureKind.UNRECOGNIZED
    inv = list_candidates(config)
    assert inv.unrecognized == ["https://totally-unknown.example/path"]


def test_cli_capture_and_list(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert (
        run_cli(
            ["capture", "https://www.empleo.ubimia.com/", "--company", "Ubimia", "--country", "CL"],
            standalone_mode=False,
        )
        == SUCCESS
    )
    out = capsys.readouterr().out
    assert "Company portal candidate" in out
    assert (
        run_cli(
            ["capture", "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=appshareios"],
            standalone_mode=False,
        )
        == SUCCESS
    )
    assert run_cli(["capture", "--list"], standalone_mode=False) == SUCCESS
    listed = capsys.readouterr().out
    assert "8a5fab1a7c476a97" in listed
    assert "ubimia" in listed.casefold() or "empleo.ubimia" in listed


def test_cli_capture_requires_url_or_list(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert run_cli(["capture"], standalone_mode=False) == VALIDATION_FAILURE


def test_capture_refuses_a_reserved_example_career_portal(tmp_path: Path) -> None:
    from jobbot.companies.registry import default_companies_path, load_companies

    config = _config(tmp_path)
    result = capture_url(config, "https://careers.example.com/trabaja-con-nosotros")
    assert result.kind == CaptureKind.UNRECOGNIZED
    assert "reserved" in result.detail
    assert load_companies(default_companies_path(tmp_path)).companies == []
