"""`browser login` is a dry-run checklist; --apply opens one portal (issue #56)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.browser.sessions import ChromeProcess, SessionState, SessionStatus
from jobbot.companies.models import (
    CareerSite,
    CareerSiteType,
    CompanyRecord,
    KnowledgeStatus,
)
from jobbot.companies.registry import CompanyRegistry, save_companies
from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS
from jobbot.portals.detect import AtsKind


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def _registry() -> CompanyRegistry:
    return CompanyRegistry(
        companies=[
            CompanyRecord(
                id="acme",
                name="Acme",
                career_sites=[
                    CareerSite(
                        url="https://acme.wd3.myworkdayjobs.com/careers",
                        domain="acme.wd3.myworkdayjobs.com",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.WORKDAY,
                        status=KnowledgeStatus.ACTIVE,
                    ),
                    CareerSite(
                        url="https://boards.greenhouse.io/acme",
                        domain="boards.greenhouse.io",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.GREENHOUSE,
                        status=KnowledgeStatus.ACTIVE,
                    ),
                ],
            ),
            CompanyRecord(
                id="northwind",
                name="Northwind",
                career_sites=[
                    CareerSite(
                        url="https://northwind.fa.oraclecloud.com/hcmUI/CandidateExperience",
                        domain="northwind.fa.oraclecloud.com",
                        site_type=CareerSiteType.ATS_INSTANCE,
                        ats=AtsKind.ORACLE,
                        status=KnowledgeStatus.CANDIDATE,
                    )
                ],
            ),
        ]
    )


def _sessions(linkedin: SessionStatus = SessionStatus.NEEDS_LOGIN) -> list[SessionState]:
    return [
        SessionState(site="indeed", status=SessionStatus.READY, evidence="indeed ready"),
        SessionState(site="linkedin", status=linkedin, evidence="linkedin wall"),
        SessionState(site="gmail", status=SessionStatus.READY, evidence="gmail ready"),
        SessionState(site="getonboard", status=SessionStatus.READY, evidence="getonboard ready"),
    ]


def _silence_chrome(monkeypatch: pytest.MonkeyPatch, sessions: list[SessionState]) -> None:
    monkeypatch.setattr("jobbot.browser.login_plan.list_chrome_processes", lambda: [])
    monkeypatch.setattr("jobbot.browser.sessions.list_chrome_processes", lambda: [])
    monkeypatch.setattr(
        "jobbot.browser.login_plan.inspect_sessions",
        lambda *_args, **_kwargs: sessions,
    )
    monkeypatch.setattr("jobbot.browser.login_plan.discover_endpoints", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "jobbot.browser.cdp.find_chrome_executable",
        lambda: Path("/usr/bin/google-chrome"),
    )


def test_dry_run_lists_gaps_and_does_not_launch(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jobbot.cli import run_cli
    from jobbot.companies.registry import load_companies

    _workspace(tmp_path, project_root, monkeypatch)
    save_companies(_registry(), tmp_path / "data" / "companies.yaml")
    before = (tmp_path / "data" / "companies.yaml").read_text(encoding="utf-8")
    _silence_chrome(monkeypatch, _sessions())
    launched: list[object] = []
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: launched.append((args, kwargs)))

    code = run_cli(["browser", "login"], standalone_mode=False)
    out = capsys.readouterr().out

    assert code == SUCCESS
    assert launched == []
    assert "Acme" in out
    assert "Northwind" in out
    assert "needs_login" in out
    assert "boards.greenhouse.io" not in out
    assert "Dry-run" in out or "dry-run" in out.casefold()
    assert (tmp_path / "data" / "companies.yaml").read_text(encoding="utf-8") == before
    stored = load_companies(tmp_path / "data" / "companies.yaml")
    northwind = next(company for company in stored.companies if company.id == "northwind")
    assert northwind.career_sites[0].status is KnowledgeStatus.CANDIDATE


def test_apply_opens_only_the_first_gap(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jobbot.cli import run_cli

    _workspace(tmp_path, project_root, monkeypatch)
    save_companies(_registry(), tmp_path / "data" / "companies.yaml")
    _silence_chrome(monkeypatch, _sessions(linkedin=SessionStatus.READY))
    launched: list[list[str]] = []

    def capture(argv: list[str], **_kwargs: object) -> object:
        launched.append(argv)
        return object()

    monkeypatch.setattr("subprocess.Popen", capture)

    code = run_cli(["browser", "login", "--apply"], standalone_mode=False)

    assert code == SUCCESS
    assert len(launched) == 1
    argv = " ".join(launched[0])
    assert "https://acme.wd3.myworkdayjobs.com/careers" in argv
    assert "oraclecloud.com" not in argv
    assert "browser-data/companies" in argv


def test_apply_does_not_launch_when_the_companies_profile_is_busy(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jobbot.cli import run_cli

    _workspace(tmp_path, project_root, monkeypatch)
    save_companies(_registry(), tmp_path / "data" / "companies.yaml")
    _silence_chrome(monkeypatch, _sessions(linkedin=SessionStatus.READY))
    profile = tmp_path / "browser-data" / "companies"
    profile.mkdir(parents=True)
    busy = [ChromeProcess(pid=4242, profile_dir=profile)]
    monkeypatch.setattr("jobbot.browser.login_plan.list_chrome_processes", lambda: busy)
    launched: list[object] = []
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: launched.append(args))

    code = run_cli(["browser", "login", "--apply"], standalone_mode=False)

    assert code == GENERIC_FAILURE
    assert launched == []
