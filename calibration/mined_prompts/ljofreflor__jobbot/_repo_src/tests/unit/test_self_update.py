"""jobbot update — uv tool reinstall / Docker hint."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from jobbot import self_update
from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS, UPDATE_AVAILABLE, VALIDATION_FAILURE


def test_update_in_docker_returns_pull_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_update, "_in_docker", lambda: True)
    result = self_update.update_jobbot()
    assert result.ok is False
    assert "docker pull" in result.message.casefold()


def test_update_without_uv_points_at_install_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_update, "_in_docker", lambda: False)
    monkeypatch.setattr(self_update.shutil, "which", lambda _name: None)
    result = self_update.update_jobbot()
    assert result.ok is False
    assert "install.sh" in result.message


def _fake_which(name: str) -> str | None:
    return "/usr/bin/uv" if name == "uv" else None


def test_update_runs_uv_tool_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(self_update, "_in_docker", lambda: False)
    monkeypatch.setattr(self_update.shutil, "which", _fake_which)

    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd: list[str], **_kwargs: object) -> _Completed:
        calls.append(list(cmd))
        return _Completed()

    monkeypatch.setattr(self_update.subprocess, "run", fake_run)
    monkeypatch.setattr(self_update, "_read_version", lambda: "0.1.0")

    result = self_update.update_jobbot(ref="main")
    assert result.ok is True
    assert calls
    assert calls[0][:4] == ["/usr/bin/uv", "tool", "install", "--force"]
    assert calls[0][-1].startswith("git+https://github.com/ljofreflor/jobbot@main")


def test_update_from_wheel_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(self_update, "_in_docker", lambda: False)
    monkeypatch.setattr(self_update.shutil, "which", _fake_which)
    wheel = tmp_path / "jobbot-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fake")

    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        self_update.subprocess,
        "run",
        lambda cmd, **_k: calls.append(list(cmd)) or _Completed(),
    )
    monkeypatch.setattr(self_update, "_read_version", lambda: "0.1.0")

    result = self_update.update_jobbot(source=f"wheel:{wheel}")
    assert result.ok is True
    assert calls[0][-1] == str(wheel)


# --check --------------------------------------------------------------------------

OLD = "1" * 40
NEW = "2" * 40


def _direct_url(commit: str | None = OLD, *, editable: bool = False) -> str:
    if editable:
        return json.dumps({"url": "file:///src/jobbot", "dir_info": {"editable": True}})
    return json.dumps(
        {
            "url": "https://github.com/ljofreflor/jobbot",
            "vcs_info": {"vcs": "git", "requested_revision": "main", "commit_id": commit},
        }
    )


class _LsRemote:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **_kwargs: object) -> _LsRemote:
        self.calls.append(list(cmd))
        return self


def test_parse_direct_url_reads_commit_ref_and_editable() -> None:
    info = self_update.parse_direct_url(_direct_url(OLD))
    assert info.commit == OLD and info.requested_ref == "main" and not info.editable
    assert self_update.parse_direct_url(_direct_url(editable=True)).editable
    assert self_update.parse_direct_url(None) == self_update.InstallInfo()
    assert self_update.parse_direct_url("{not json") == self_update.InstallInfo()


def test_parse_ls_remote_prefers_branch_then_peeled_tag() -> None:
    out = f"{OLD}\trefs/tags/v1\n{NEW}\trefs/tags/v1^{{}}\n{'3' * 40}\trefs/heads/main\n"
    assert self_update.parse_ls_remote(out, "main") == "3" * 40
    assert self_update.parse_ls_remote(out, "v1") == NEW
    assert self_update.parse_ls_remote("", "main") is None


def test_check_up_to_date_uses_ls_remote_only() -> None:
    runner = _LsRemote(f"{OLD}\trefs/heads/main\n")
    result = self_update.check_for_update(
        ref="main", runner=runner, git="/usr/bin/git", read=lambda: _direct_url(OLD)
    )
    assert result.status == "up_to_date"
    assert runner.calls == [
        ["/usr/bin/git", "ls-remote", "https://github.com/ljofreflor/jobbot", "main"]
    ]


def test_check_reports_available_with_short_hashes() -> None:
    result = self_update.check_for_update(
        ref="main",
        runner=_LsRemote(f"{NEW}\trefs/heads/main\n"),
        git="/usr/bin/git",
        read=lambda: _direct_url(OLD),
    )
    assert result.status == "available"
    assert "1111111 → 2222222" in result.message


def test_check_offline_is_an_error_not_a_crash() -> None:
    result = self_update.check_for_update(
        ref="main",
        runner=_LsRemote("", returncode=128, stderr="Could not resolve host"),
        git="/usr/bin/git",
        read=lambda: _direct_url(OLD),
    )
    assert result.status == "error"
    assert "Could not resolve host" in result.message


def test_check_timeout_is_an_error() -> None:
    def slow(cmd: list[str], **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd, 10)

    result = self_update.check_for_update(
        ref="main", runner=slow, git="/usr/bin/git", read=lambda: _direct_url(OLD)
    )
    assert result.status == "error" and "timed out" in result.message


def test_check_unknown_and_editable_installs() -> None:
    runner = _LsRemote(f"{NEW}\trefs/heads/main\n")
    unknown = self_update.check_for_update(
        ref="main", runner=runner, git="/usr/bin/git", read=lambda: None
    )
    editable = self_update.check_for_update(
        ref="main", runner=runner, git="/usr/bin/git", read=lambda: _direct_url(editable=True)
    )
    assert unknown.status == "unknown"
    assert editable.status == "editable" and "git pull" in editable.message


def test_check_full_sha_ref_needs_no_network() -> None:
    def no_network(cmd: list[str], **_kwargs: object) -> object:
        raise AssertionError("ls-remote must not run for a full SHA")

    result = self_update.check_for_update(
        ref=NEW, runner=no_network, git="/usr/bin/git", read=lambda: _direct_url(NEW)
    )
    assert result.status == "up_to_date"


# CLI ------------------------------------------------------------------------------


@pytest.fixture
def cli_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(self_update, "_in_docker", lambda: False)
    return tmp_path


def _cli(args: list[str]) -> int:
    from jobbot.cli import run_cli

    return run_cli(["update", *args], standalone_mode=False)


@pytest.mark.parametrize(
    ("status", "code"),
    [("up_to_date", SUCCESS), ("available", UPDATE_AVAILABLE), ("error", GENERIC_FAILURE)],
)
def test_cli_check_exit_codes(
    cli_home: Path, monkeypatch: pytest.MonkeyPatch, status: str, code: int
) -> None:
    monkeypatch.setattr(
        self_update,
        "check_for_update",
        lambda **_k: self_update.CheckResult(status=status, message=status),
    )
    assert _cli(["--check"]) == code
    assert not (cli_home / "output").exists(), "expected outcomes are not ops failures"


@pytest.mark.parametrize(
    "args",
    [
        ["--schedule", "daily", "--check"],
        ["--schedule", "daily", "--ref", "dev"],
        ["--check", "--scheduled"],
    ],
)
def test_cli_rejects_exclusive_options(cli_home: Path, args: list[str]) -> None:
    assert _cli(args) == VALIDATION_FAILURE


def test_cli_schedule_in_docker_touches_nothing(
    cli_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot import self_update_schedule

    monkeypatch.setattr(self_update, "_in_docker", lambda: True)
    monkeypatch.setattr(
        self_update_schedule.Scheduler,
        "install",
        lambda *_a: pytest.fail("must not schedule inside Docker"),
    )
    assert _cli(["--schedule", "daily"]) == GENERIC_FAILURE
    assert list(cli_home.iterdir()) == []


def test_cli_schedule_daily_installs_with_absolute_binary(
    cli_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot import self_update_schedule

    installed: list[self_update_schedule.ScheduleSpec] = []

    def fake_install(
        _self: self_update_schedule.Scheduler, spec: self_update_schedule.ScheduleSpec
    ) -> self_update_schedule.ScheduleResult:
        installed.append(spec)
        return self_update_schedule.ScheduleResult(ok=True, message="scheduled")

    binary = cli_home / "bin" / "jobbot"
    monkeypatch.setattr(self_update, "read_direct_url", lambda: _direct_url(OLD))
    monkeypatch.setattr(self_update_schedule, "resolve_binary", lambda: binary)
    monkeypatch.setattr(self_update_schedule.Scheduler, "install", fake_install)

    assert _cli(["--schedule", "daily"]) == SUCCESS
    assert installed and installed[0].binary == binary
    assert str(binary.parent) in installed[0].path_env.split(":")


def test_cli_schedule_refuses_editable_install(
    cli_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot import self_update_schedule

    monkeypatch.setattr(self_update, "read_direct_url", lambda: _direct_url(editable=True))
    monkeypatch.setattr(
        self_update_schedule.Scheduler,
        "install",
        lambda *_a: pytest.fail("must not schedule an editable checkout"),
    )
    assert _cli(["--schedule", "daily"]) == VALIDATION_FAILURE


def test_cli_scheduled_run_honours_opt_out(
    cli_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("JOBBOT_AUTO_UPDATE", "0")
    monkeypatch.setattr(
        self_update, "check_for_update", lambda **_k: pytest.fail("opt-out must not check")
    )
    assert _cli(["--scheduled"]) == SUCCESS
    assert "Auto-update disabled" in capsys.readouterr().out


def test_latest_commit_without_git_or_unknown_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_update.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="git not found"):
        self_update.latest_commit("https://example.invalid/repo", "main")
    with pytest.raises(RuntimeError, match="not found on"):
        self_update.latest_commit(
            "https://example.invalid/repo", "nope", runner=_LsRemote(""), git="/usr/bin/git"
        )
