"""Maintainer bash lane from a stored failure (issue #46)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.config import JobbotConfig, PathsConfig
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS
from jobbot.ops.failure_work import (
    branch_name,
    open_branch_commands,
    work_script,
)
from jobbot.ops.failures import record_failure


def _session(tmp_path: Path):
    config = JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(
            profile=Path("data/profile.yaml"),
            database=Path("data/jobbot.sqlite"),
            output=Path("output"),
        ),
    )
    engine = make_engine(config.database_path)
    return make_session_factory(engine)(), config


def test_branch_name_stable_from_id_and_fingerprint(tmp_path: Path) -> None:
    session, config = _session(tmp_path)
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=GENERIC_FAILURE,
            argv=["jobbot", "cv", "sync"],
            error_class="RuntimeError",
            message="boom",
        )
        name = branch_name(rec)
        assert name.startswith(f"fix/{rec.id.casefold()}-")
        assert len(name.split("-")[-1]) == 8
        assert branch_name(rec) == name
    finally:
        session.close()


def test_work_script_is_bash_and_lists_the_lane(tmp_path: Path) -> None:
    session, config = _session(tmp_path)
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=GENERIC_FAILURE,
            argv=["jobbot", "getonboard", "sync", "--apply"],
            error_class="TimeoutError",
            message="portal hung",
        )
        script = work_script(rec)
    finally:
        session.close()

    assert script.startswith("#!/usr/bin/env bash")
    assert "git fetch --all" in script
    assert f"git checkout -b {branch_name(rec)}" in script or branch_name(rec) in script
    assert "gh pr create --fill" in script
    assert f"ops failure issue {rec.id}" in script
    assert f"ops failure triage {rec.id} --status fixed" in script
    assert "uv run jobbot getonboard sync --apply" in script
    assert "set -euo pipefail" in script


def test_work_script_skips_issue_create_when_linked(tmp_path: Path) -> None:
    session, config = _session(tmp_path)
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=GENERIC_FAILURE,
            argv=["jobbot", "indeed", "sync"],
            error_class="RuntimeError",
            message="x",
        )
        script = work_script(rec, issue_url="https://github.com/ljofreflor/jobbot/issues/99")
    finally:
        session.close()

    assert "ops failure issue" not in script
    assert "issues/99" in script


def test_open_branch_commands_are_git_only(tmp_path: Path) -> None:
    session, config = _session(tmp_path)
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=GENERIC_FAILURE,
            argv=["jobbot", "cv", "sync"],
            error_class="RuntimeError",
            message="x",
        )
        steps = open_branch_commands(rec)
    finally:
        session.close()

    assert steps[0] == ["git", "fetch", "--all"]
    assert steps[-1] == ["git", "checkout", "-b", branch_name(rec)]
    flat = " ".join(" ".join(s) for s in steps)
    assert "commit" not in flat
    assert "push" not in flat
    assert "gh " not in flat


def test_cli_work_dry_run_prints_script(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    session, config = _session(tmp_path)
    try:
        rec = record_failure(
            session,
            config=config,
            exit_code=GENERIC_FAILURE,
            argv=["jobbot", "cv", "sync"],
            error_class="RuntimeError",
            message="sync exploded",
        )
        fid = rec.id
    finally:
        session.close()

    # Point the CLI session at this DB via cwd config defaults.
    from jobbot.cli import run_cli

    assert run_cli(["ops", "failure", "work", fid], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert "git fetch --all" in out
    assert "Dry-run" in out
    assert branch_name(rec) in out or fid.casefold() in out.casefold()
