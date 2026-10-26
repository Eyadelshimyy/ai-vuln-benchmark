"""Applying on a portal that keeps a candidate profile needs a signed-in session."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.applications.login_gate import login_warning
from jobbot.browser.sessions import SessionState, SessionStatus
from jobbot.exit_codes import SUCCESS
from jobbot.portals.detect import AtsKind

MUST_SIGN_IN = "no puedes postular si no inicias sesión"


def _state(site: str, status: SessionStatus, evidence: str = "e") -> SessionState:
    return SessionState(site=site, status=status, evidence=evidence)


def test_portal_without_session_check_warns_to_sign_in() -> None:
    warning = login_warning(AtsKind.TORRE, [])
    assert warning is not None
    assert MUST_SIGN_IN in warning
    assert "Torre" in warning


def test_login_wall_evidence_warns_to_sign_in() -> None:
    state = _state("getonboard", SessionStatus.NEEDS_LOGIN, "open tab sits on the login wall")
    warning = login_warning(AtsKind.GETONBOARD, [state])
    assert warning is not None
    assert MUST_SIGN_IN in warning
    assert "login wall" in warning


def test_unproven_session_is_not_taken_as_signed_in() -> None:
    state = _state("indeed", SessionStatus.UNKNOWN, "tab proves nothing about the session")
    warning = login_warning(AtsKind.INDEED, [state])
    assert warning is not None
    assert MUST_SIGN_IN in warning


def test_evidenced_session_needs_no_warning() -> None:
    state = _state("getonboard", SessionStatus.READY, "signed-in page open")
    assert login_warning(AtsKind.GETONBOARD, [state]) is None


def test_platform_that_takes_applications_directly_needs_no_warning() -> None:
    assert login_warning(AtsKind.GREENHOUSE, []) is None


def test_unknown_platform_is_not_claimed_to_need_an_account() -> None:
    assert login_warning(AtsKind.UNKNOWN, []) is None


def test_apply_dry_run_tells_you_to_sign_in_on_torre(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: a Torre apply opened the page without saying a session is required."""
    (tmp_path / "data").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    import jobbot.cli as cli
    from jobbot.cli import run_cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    monkeypatch.setattr(
        "jobbot.jobs.closure.fetch_posting_text",
        lambda _url, **_k: "<html>still open</html>",
    )
    monkeypatch.setattr("jobbot.browser.sessions.inspect_sessions", lambda *_a, **_k: [])
    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    url = "https://torre.ai/post/abc123-example-co-senior-analyst"
    stored = JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "PENDING",
                "source": "torre",
                "title": "Senior Analyst",
                "company": "Example Co",
                "url": url,
                "ats_url": url,
                "ats_kind": "torre",
                "description": "Una descripción.",
            }
        )
    )

    code = run_cli(["application", "apply", stored.id], standalone_mode=False)

    assert code == SUCCESS
    out = " ".join(capsys.readouterr().out.split())
    assert MUST_SIGN_IN in out
