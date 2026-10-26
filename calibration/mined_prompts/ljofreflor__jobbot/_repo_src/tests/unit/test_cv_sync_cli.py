"""`jobbot cv sync` dry-run is inert and lists active companies only."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jobbot.exit_codes import SUCCESS


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def _write_companies(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "companies": [
            {
                "id": "betterfly",
                "name": "Betterfly",
                "career_sites": [
                    {
                        "url": "https://betterfly.wd3.myworkdayjobs.com/careers",
                        "domain": "betterfly.wd3.myworkdayjobs.com",
                        "site_type": "ats_instance",
                        "ats": "workday",
                        "status": "active",
                    }
                ],
            },
            {
                "id": "buk",
                "name": "Buk",
                "career_sites": [
                    {
                        "url": "https://buk.cl/trabaja-con-nosotros",
                        "domain": "buk.cl",
                        "site_type": "company_career_portal",
                        "ats": "unknown",
                        "status": "candidate",
                    }
                ],
            },
        ],
    }
    (tmp_path / "data" / "companies.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True),
        encoding="utf-8",
    )


def test_cv_sync_dry_run_lists_active_only_and_opens_no_browser(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    _write_companies(tmp_path)
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )

    assert run_cli(["cv", "sync"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out

    assert opened == []
    assert "Betterfly" in out
    assert "needs_account" in out
    assert "Buk" not in out
    assert "permanent" in out.casefold() or "CV sync" in out
    assert "Dry-run" in out


def test_apply_signup_row_shows_the_sheet_and_opens_no_browser(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    _write_companies(tmp_path)
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )
    monkeypatch.setattr("jobbot.cli._apply_permanent_plans", lambda *args, **kwargs: None)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("browser session started")

    monkeypatch.setattr("jobbot.browser.session.BrowserSession", _boom)

    assert run_cli(["cv", "sync", "--apply", "--yes"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert opened == []
    assert "Betterfly" in out
    assert "password" in out.casefold() or "does not create" in out.casefold()
    assert not list((tmp_path / "output").rglob("*receipt*"))
