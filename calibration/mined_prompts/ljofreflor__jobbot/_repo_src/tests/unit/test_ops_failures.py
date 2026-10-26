"""Local ops failure observability."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.exit_codes import (
    GENERIC_FAILURE,
    SUCCESS,
    UI_CHANGED,
    USER_CANCEL,
    VALIDATION_FAILURE,
)
from jobbot.ops.failures import (
    capture_cli_failure,
    failure_fingerprint,
    get_failure,
    list_failures,
    mark_status,
    normalize_message,
    record_failure,
    should_record_cli_failure,
)
from jobbot.ops.loop import LoopTickResult, run_loop, run_loop_tick
from jobbot.ops.redact import host_only_url, redact_context, redact_text


def _config(tmp_path: Path) -> JobbotConfig:
    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(
            profile=Path("data/profile.yaml"),
            database=Path("data/jobbot.sqlite"),
            output=Path("output"),
        ),
    )


def _session(tmp_path: Path) -> tuple[Session, JobbotConfig]:
    config = _config(tmp_path)
    engine = make_engine(config.database_path)
    return make_session_factory(engine)(), config


def test_fingerprint_stable_across_numeric_noise() -> None:
    a = failure_fingerprint(
        component="getonboard",
        error_class="RuntimeError",
        message="job J0016 failed after 12 retries",
    )
    b = failure_fingerprint(
        component="getonboard",
        error_class="RuntimeError",
        message="job J0099 failed after 3 retries",
    )
    assert a == b
    assert len(a) == 16


def test_normalize_command_stabilizes_job_urls() -> None:
    """Hard-link argv must drop tracking so the failure-work re-run stays clean."""
    from jobbot.ops.failures import normalize_command

    cmd = normalize_command(
        [
            "jobbot",
            "get",
            "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=email&tk=xyz",
        ]
    )
    assert cmd == "jobbot get https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"

    linkedin = normalize_command(
        ["jobbot", "get", "https://www.linkedin.com/jobs/view/123?trk=flagship"]
    )
    assert "trk=" not in linkedin
    assert "linkedin.com/jobs/view/123" in linkedin


def test_normalize_message_collapses_noise() -> None:
    assert "j0016" not in normalize_message("see J0016 path /tmp/x")
    assert "<id>" in normalize_message("see J0016")


def test_redact_strips_secrets_and_url_query() -> None:
    text = redact_text("OPENAI_API_KEY=sk-abc1234567890 token=sekrit Bearer abc.def")
    assert "sekrit" not in text
    assert "sk-abc" not in text or "redacted" in text.lower()
    assert "Bearer <redacted>" in text or "redacted" in text.lower()
    assert host_only_url("https://example.com/apply?token=secret#x") == (
        "https://example.com/apply"
    )
    ctx = redact_context({"ats_url": "https://boards.greenhouse.io/x?key=1", "note": "ok"})
    assert "key=" not in str(ctx["ats_url"])
    assert ctx["note"] == "ok"


def test_record_failure_strips_vault_passwords(tmp_path: Path) -> None:
    from jobbot.ops.failures import record_failure
    from jobbot.vault import init_vault, put_entry, vault_path

    session, config = _session(tmp_path)
    secret = "unique-ops-vault-secret"
    profile = config.profile_path
    profile.parent.mkdir(parents=True, exist_ok=True)
    if not profile.is_file():
        profile.write_text("name: Test\n", encoding="utf-8")
    init_vault(vault_path(profile))
    put_entry(vault_path(profile), "indeed", secret)
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=GENERIC_FAILURE,
            argv=["jobbot", "browser", "login"],
            error_class="RuntimeError",
            message=f"typed {secret} into the form",
            tb=f"RuntimeError: {secret}",
        )
        assert secret not in rec.message
        assert secret not in rec.traceback
        mirror = (config.output_dir / "ops" / "failures" / f"{rec.id}.json").read_text(
            encoding="utf-8"
        )
        assert secret not in mirror
    finally:
        session.close()


def test_redact_strips_contact_pii_not_only_secrets() -> None:
    """Failures leave the machine via ops failure issue; contact data must go too."""
    sample = (
        "Failed for ana.ejemplo@gmail.com phone +56 9 1234 5678 "
        "RUT 12.345.678-9 path /Users/someone/jobbot/data/profile.yaml "
        "token=sekrit"
    )
    text = redact_text(sample)
    assert "ana.ejemplo@gmail.com" not in text
    assert "+56 9 1234 5678" not in text
    assert "12.345.678-9" not in text
    assert "/Users/someone/" not in text
    assert "sekrit" not in text


def test_issue_body_does_not_leak_contact_pii(tmp_path: Path) -> None:
    from jobbot.ops.failures import issue_body, record_failure

    session, config = _session(tmp_path)
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=GENERIC_FAILURE,
            argv=["jobbot", "indeed", "sync"],
            error_class="RuntimeError",
            message=(
                "sync failed for ana.candidato@gmail.com at /Users/candidato/jobbot "
                "tel +56 9 8765 4321"
            ),
            context={"note": "RUT 11.222.333-4 also in context"},
            tb=(
                'File "/Users/candidato/jobbot/src/jobbot/cli.py", line 1\n'
                "RuntimeError: ana.candidato@gmail.com"
            ),
        )
        body = issue_body(rec)
    finally:
        session.close()

    assert "ana.candidato@gmail.com" not in body
    assert "/Users/candidato/" not in body
    assert "+56 9 8765 4321" not in body
    assert "11.222.333-4" not in body


def test_record_and_list_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    engine = make_engine(config.database_path)
    session = make_session_factory(engine)()
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=UI_CHANGED,
            argv=["jobbot", "linkedin", "sweep", "hiring"],
            error_class="UIChanged",
            message="selector not found",
            context={"url": "https://www.linkedin.com/feed?session=abc"},
        )
        assert rec.id == "F0001"
        assert rec.component == "linkedin"
        assert rec.fingerprint
        mirror = config.output_dir / "ops" / "failures" / "F0001.json"
        assert mirror.is_file()
        listed = list_failures(session, status="new")
        assert len(listed) == 1
        assert get_failure(session, "F0001") is not None
        updated = mark_status(session, "F0001", "fixed")
        assert updated is not None
        assert updated.status == "fixed"
        assert list_failures(session, status="new") == []
    finally:
        session.close()


def test_should_not_record_success_or_ops() -> None:
    assert not should_record_cli_failure(["jobbot", "profile", "validate"], SUCCESS)
    assert not should_record_cli_failure(["jobbot", "ops", "failures"], GENERIC_FAILURE)
    assert should_record_cli_failure(["jobbot", "probe-exit", "5"], UI_CHANGED)
    assert not should_record_cli_failure(["jobbot", "jobs", "add"], USER_CANCEL)


def test_should_not_record_help_invocations() -> None:
    """Exploring CLI help is not an ops failure — even when the command is wrong."""
    assert not should_record_cli_failure(["jobbot", "cv", "build", "--help"], GENERIC_FAILURE)
    assert not should_record_cli_failure(["jobbot", "cv", "build", "-h"], GENERIC_FAILURE)
    assert not should_record_cli_failure(
        ["jobbot", "ops failure show", "--help"], GENERIC_FAILURE
    )


def test_capture_cli_failure_ui_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    config = _config(tmp_path)
    rec = capture_cli_failure(
        UI_CHANGED,
        argv=["jobbot", "probe-exit", "5"],
        config=config,
    )
    assert rec is not None
    assert rec.id == "F0001"
    assert rec.exit_code == UI_CHANGED


def test_run_cli_records_ui_changed_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-first regression: nonzero CLI exits must land in ops_failures."""
    from jobbot.cli import run_cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    code = run_cli(["probe-exit", str(UI_CHANGED)], standalone_mode=False)
    assert code == UI_CHANGED
    engine = make_engine(tmp_path / "data" / "jobbot.sqlite")
    session = make_session_factory(engine)()
    try:
        rows = list_failures(session, status="new")
        assert len(rows) == 1
        assert rows[0].exit_code == UI_CHANGED
        assert rows[0].id == "F0001"
    finally:
        session.close()


def test_loop_tick_records_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    calls = {"n": 0}

    def boom(_cfg: JobbotConfig) -> None:
        calls["n"] += 1
        raise RuntimeError("boom for observability")

    monkeypatch.setitem(
        __import__("jobbot.ops.loop", fromlist=["_RUNNERS"])._RUNNERS,
        "getonboard-prepare",
        boom,
    )
    tick = run_loop_tick("getonboard-prepare", config=config)
    assert not tick.ok
    assert tick.failure is not None
    assert tick.failure.id == "F0001"

    sleeps: list[float] = []
    ticks: list[bool] = []

    def on_tick(result: LoopTickResult) -> None:
        ticks.append(result.ok)

    code = run_loop(
        "getonboard-prepare",
        interval_sec=0.01,
        fail_fast=False,
        max_ticks=3,
        config=config,
        sleep_fn=sleeps.append,
        on_tick=on_tick,
    )
    assert code == GENERIC_FAILURE
    assert calls["n"] >= 3
    assert ticks == [False, False, False]
    assert len(sleeps) == 2  # sleep between ticks, not after last


def test_runtime_context_describes_the_run() -> None:
    """Issues built from crashes need to say whether a human could answer prompts."""
    from jobbot.ops.failures import runtime_context

    ctx = runtime_context()
    assert set(ctx) >= {"stdin_tty", "version", "platform"}
    assert isinstance(ctx["stdin_tty"], bool)


def test_abort_is_recorded_as_abort_with_a_message(tmp_path: Path) -> None:
    """Regression: prompt aborts landed as 'Exit:' with no message, no context."""
    from jobbot.ops.failures import ABORT_MESSAGE, record_failure, runtime_context

    session, config = _session(tmp_path)
    record = record_failure(
        session,
        config=config,
        exit_code=1,
        argv=["jobbot", "linkedin", "sweep", "hiring data scientist"],
        error_class="Abort",
        message=ABORT_MESSAGE,
        context=runtime_context(),
        write_mirror=False,
    )

    assert record.error_class == "Abort"
    assert "prompt" in record.message
    assert "stdin_tty" in record.context


def test_capture_cli_failure_passes_class_message_and_context(tmp_path: Path) -> None:
    from jobbot.ops.failures import capture_cli_failure

    _session(tmp_path)
    config = _config(tmp_path)
    record = capture_cli_failure(
        1,
        argv=["jobbot", "linkedin", "sweep"],
        error_class="Abort",
        message="aborted at a confirmation prompt",
        context={"stdin_tty": False},
        config=config,
    )
    assert record is not None
    assert record.error_class == "Abort"
    assert record.context["stdin_tty"] is False


def test_issue_body_includes_runtime_context(tmp_path: Path) -> None:
    from jobbot.ops.failures import issue_body, record_failure

    session, config = _session(tmp_path)
    record = record_failure(
        session,
        config=config,
        exit_code=1,
        argv=["jobbot", "linkedin", "sweep"],
        error_class="Abort",
        message="aborted at a confirmation prompt",
        context={"stdin_tty": False, "version": "0.1.0", "platform": "darwin"},
        write_mirror=False,
    )
    body = issue_body(record)
    assert "stdin_tty" in body
    assert "Regression unit test" in body

def test_cv_propagate_invalid_target_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for F0082 / issue #47: unknown propagation target must not record ops_failure."""
    from jobbot.cli import run_cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "profile.yaml").write_text(
        """
name: Test User
email: test@example.com
phone: "+1234567890"
location: Test City
summary: Test summary
experience: []
education: []
skills: []
"""
    )
    code = run_cli(["cv", "propagate", "--targets", "companies"], standalone_mode=False)
    assert code == VALIDATION_FAILURE
    engine = make_engine(tmp_path / "data" / "jobbot.sqlite")
    session = make_session_factory(engine)()
    try:
        rows = list_failures(session, status="new")
        assert len(rows) == 0, "Invalid target should not record an ops_failure"
    finally:
        session.close()

def test_should_not_record_validation_failure() -> None:
    """Regression: validation errors (bad input) should not be recorded as ops failures."""
    assert not should_record_cli_failure(
        ["jobbot", "cv", "propagate", "--targets", "companies"], VALIDATION_FAILURE
    )
