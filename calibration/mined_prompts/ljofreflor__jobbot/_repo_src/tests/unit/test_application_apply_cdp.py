"""`application apply --apply --cdp`: fill the external ATS in your Chrome, never submit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jobbot.exit_codes import SUCCESS

ATS_URL = "https://boards.greenhouse.io/acme/jobs/1"
CDP = "http://127.0.0.1:9222"


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
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
    monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)
    monkeypatch.setattr("jobbot.cli._fetch_public_html", lambda _url: None)
    monkeypatch.setattr(
        "jobbot.jobs.closure.fetch_posting_text",
        lambda _url, **_k: "<html>still open</html>",
    )
    import jobbot.cli as cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    stored = JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "PENDING",
                "source": "manual",
                "title": "Un cargo",
                "company": "Acme",
                "url": ATS_URL,
                "ats_url": ATS_URL,
                "ats_kind": "greenhouse",
                "description": "Una descripción.",
            }
        )
    )
    session.commit()
    cv = config.output_dir / "jobs" / stored.id / "application" / "cv.pdf"
    cv.parent.mkdir(parents=True, exist_ok=True)
    cv.write_bytes(b"%PDF-1.4\n" + b"x" * 40)
    return stored, cv


def test_cdp_fills_in_your_chrome_instead_of_the_default_browser(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored, cv = _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.adapters.ats.apply_fill import ApplyFillResult
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )
    calls: list[dict[str, Any]] = []

    def fake_fill(cdp_url: str, url: str, candidate: Any, **kwargs: Any) -> ApplyFillResult:
        calls.append({"cdp": cdp_url, "url": url, **kwargs})
        return ApplyFillResult(
            url=url,
            readable=True,
            filled=("First Name", "Email"),
            attached=True,
            left_for_human=("Expected salary", "Final submit / apply button"),
        )

    monkeypatch.setattr("jobbot.adapters.ats.apply_fill.open_and_fill_over_cdp", fake_fill)

    code = run_cli(
        ["application", "apply", stored.id, "--apply", "--yes", "--cdp", CDP],
        standalone_mode=False,
    )

    assert code == SUCCESS
    assert opened == [], "with --cdp the default browser is not used"
    assert calls == [{"cdp": CDP, "url": ATS_URL, "cv_path": cv}]
    out = capsys.readouterr().out
    assert "First Name" in out and "Expected salary" in out
    assert cv.name in out
    assert "you submit" in out.casefold()


def test_cdp_failure_falls_back_to_the_default_browser(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored, _cv = _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.adapters.ats.apply_fill import ApplyFillError
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )

    def broken(*_a: Any, **_k: Any) -> None:
        raise ApplyFillError("Could not attach to Chrome at http://127.0.0.1:9222: refused")

    monkeypatch.setattr("jobbot.adapters.ats.apply_fill.open_and_fill_over_cdp", broken)

    code = run_cli(
        ["application", "apply", stored.id, "--apply", "--yes", "--cdp", CDP],
        standalone_mode=False,
    )

    assert code == SUCCESS
    assert opened == [ATS_URL]
    err = capsys.readouterr().err
    assert "Could not attach" in err


def test_without_cdp_the_default_browser_is_used_as_before(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored, _cv = _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )

    def must_not_run(*_a: Any, **_k: Any) -> None:
        raise AssertionError("no CDP fill without --cdp")

    monkeypatch.setattr("jobbot.adapters.ats.apply_fill.open_and_fill_over_cdp", must_not_run)

    code = run_cli(["application", "apply", stored.id, "--apply", "--yes"], standalone_mode=False)

    assert code == SUCCESS
    assert opened == [ATS_URL]
