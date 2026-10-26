"""jobbot update --schedule — launchd plist / crontab line, opt-out and the daily run."""

from __future__ import annotations

import plistlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobbot import self_update_schedule as sched
from jobbot.self_update import CheckResult, UpdateResult


@dataclass
class _Completed:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class FakeRunner:
    """Records argv; ``crontab`` keeps an in-memory table like the real one."""

    crontab: str | None = None
    launchctl_rc: dict[str, int] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, cmd: list[str], **kwargs: Any) -> _Completed:
        self.calls.append(list(cmd))
        if cmd[0] == "crontab" and cmd[1] == "-l":
            if self.crontab is None:
                return _Completed(returncode=1, stderr="no crontab for user\n")
            return _Completed(stdout=self.crontab)
        if cmd[0] == "crontab" and cmd[1] == "-":
            self.crontab = kwargs["input"]
            return _Completed()
        if cmd[0] == "launchctl":
            return _Completed(returncode=self.launchctl_rc.get(cmd[1], 0))
        raise AssertionError(f"unexpected command {cmd}")


def _spec(tmp_path: Path, **overrides: Any) -> sched.ScheduleSpec:
    values: dict[str, Any] = {
        "binary": tmp_path / "bin" / "jobbot",
        "log_path": tmp_path / "logs" / "update.log",
        "path_env": "/opt/tools/bin:/usr/bin:/bin",
        "working_dir": tmp_path,
    }
    values.update(overrides)
    return sched.ScheduleSpec(**values)


# pure builders ------------------------------------------------------------------


def test_plist_runs_absolute_binary_daily_without_run_at_load(tmp_path: Path) -> None:
    data = plistlib.loads(sched.build_launchd_plist(_spec(tmp_path)))

    assert data["Label"] == "com.ljofreflor.jobbot.update"
    assert data["ProgramArguments"] == [str(tmp_path / "bin" / "jobbot"), "update", "--scheduled"]
    assert data["StartCalendarInterval"] == {"Hour": 4, "Minute": 30}
    assert data["RunAtLoad"] is False
    assert data["EnvironmentVariables"] == {"PATH": "/opt/tools/bin:/usr/bin:/bin"}
    assert data["StandardOutPath"] == str(tmp_path / "logs" / "update.log")
    assert data["StandardErrorPath"] == data["StandardOutPath"]
    assert data["WorkingDirectory"] == str(tmp_path)


def test_plist_is_deterministic(tmp_path: Path) -> None:
    assert sched.build_launchd_plist(_spec(tmp_path)) == sched.build_launchd_plist(_spec(tmp_path))


def test_cron_line_is_daily_marked_and_quotes_paths(tmp_path: Path) -> None:
    spec = _spec(tmp_path, binary=tmp_path / "my tools" / "jobbot", path_env="/a%b:/usr/bin")
    line = sched.build_cron_line(spec)

    assert line.startswith("30 4 * * * ")
    assert line.endswith(" # jobbot:auto-update")
    assert f"'{tmp_path / 'my tools' / 'jobbot'}' update --scheduled" in line
    assert f">> {tmp_path / 'logs' / 'update.log'} 2>&1" in line
    assert "PATH=/a\\%b:/usr/bin" in line


def test_merge_crontab_appends_replaces_and_removes() -> None:
    other = "0 9 * * 1 /usr/bin/backup # mine\n"
    first = sched.merge_crontab(other, "30 4 * * * jobbot update # jobbot:auto-update")
    again = sched.merge_crontab(first, "30 4 * * * jobbot update # jobbot:auto-update")

    assert first == again
    assert again.count("# jobbot:auto-update") == 1
    assert again.startswith(other)
    assert sched.merge_crontab(again, None) == other
    assert sched.merge_crontab("", None) == ""


def test_log_path_per_platform(tmp_path: Path) -> None:
    assert sched.log_path("darwin", tmp_path, {}) == (
        tmp_path / "Library" / "Logs" / "jobbot" / "update.log"
    )
    assert sched.log_path("linux", tmp_path, {}) == (
        tmp_path / ".local" / "state" / "jobbot" / "update.log"
    )
    state = tmp_path / "state"
    assert sched.log_path("linux", tmp_path, {"XDG_STATE_HOME": str(state)}) == (
        state / "jobbot" / "update.log"
    )


def test_scheduled_path_env_keeps_order_and_dedupes() -> None:
    path = sched.scheduled_path_env(["/opt/uv/bin", None, "/usr/bin"], "linux")
    assert path == "/opt/uv/bin:/usr/bin:/usr/local/bin:/bin"


def test_backend_for_platform() -> None:
    assert sched.backend_for("darwin") == "launchd"
    assert sched.backend_for("linux") == "cron"
    assert sched.backend_for("win32") is None


def test_resolve_binary_prefers_path_and_keeps_symlink(tmp_path: Path) -> None:
    shim = tmp_path / "bin" / "jobbot"
    assert sched.resolve_binary(lambda _n: str(shim), argv0="python") == shim
    assert sched.resolve_binary(lambda _n: None, argv0="python") is None


# opt-out ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {"JOBBOT_AUTO_UPDATE": "0"},
        {"JOBBOT_AUTO_UPDATE": "false"},
        {"JOBBOT_AUTO_UPDATE": " OFF "},
        {"JOBBOT_NO_AUTO_UPDATE": "1"},
    ],
)
def test_opt_out_from_env(env: dict[str, str]) -> None:
    assert sched.auto_update_opt_out(env, None) is not None


def test_opt_out_from_user_config(tmp_path: Path) -> None:
    config = sched.user_config_path(tmp_path)
    config.parent.mkdir(parents=True)
    config.write_text("[update]\nauto = false\n", encoding="utf-8")

    assert "auto = false" in (sched.auto_update_opt_out({}, config) or "")


def test_no_opt_out_by_default_or_with_broken_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    assert sched.auto_update_opt_out({"JOBBOT_AUTO_UPDATE": "1"}, config) is None
    config.write_text("not = [toml", encoding="utf-8")
    assert sched.auto_update_opt_out({}, config) is None


# launchd ------------------------------------------------------------------------


def _scheduler(tmp_path: Path, platform: str, runner: FakeRunner) -> sched.Scheduler:
    return sched.Scheduler(platform=platform, home=tmp_path, env={}, runner=runner, uid=501)


def test_launchd_install_writes_plist_and_bootstraps(tmp_path: Path) -> None:
    runner = FakeRunner()
    scheduler = _scheduler(tmp_path, "darwin", runner)
    binary = tmp_path / "bin" / "jobbot"

    result = scheduler.install(scheduler.spec(binary, extra_path=["/opt/uv/bin"]))

    plist = tmp_path / "Library" / "LaunchAgents" / "com.ljofreflor.jobbot.update.plist"
    assert result.ok, result.message
    assert plistlib.loads(plist.read_bytes())["ProgramArguments"][0] == str(binary)
    assert (tmp_path / "Library" / "Logs" / "jobbot").is_dir()
    assert runner.calls == [
        ["launchctl", "bootout", "gui/501/com.ljofreflor.jobbot.update"],
        ["launchctl", "bootstrap", "gui/501", str(plist)],
    ]


def test_launchd_install_twice_leaves_one_identical_plist(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, "darwin", FakeRunner())
    spec = scheduler.spec(tmp_path / "bin" / "jobbot")
    scheduler.install(spec)
    agents = tmp_path / "Library" / "LaunchAgents"
    first = (agents / "com.ljofreflor.jobbot.update.plist").read_bytes()

    assert scheduler.install(spec).ok
    assert [p.name for p in agents.iterdir()] == ["com.ljofreflor.jobbot.update.plist"]
    assert (agents / "com.ljofreflor.jobbot.update.plist").read_bytes() == first


def test_launchd_bootstrap_failure_is_reported(tmp_path: Path) -> None:
    runner = FakeRunner(launchctl_rc={"bootstrap": 5})
    scheduler = _scheduler(tmp_path, "darwin", runner)

    result = scheduler.install(scheduler.spec(tmp_path / "bin" / "jobbot"))

    assert result.ok is False
    assert "bootstrap" in result.message


def test_launchd_remove_and_status(tmp_path: Path) -> None:
    runner = FakeRunner()
    scheduler = _scheduler(tmp_path, "darwin", runner)
    scheduler.install(scheduler.spec(tmp_path / "bin" / "jobbot"))
    scheduler.log_path.write_text("2026-10-09T04:30:00 Up to date with main: abc1234\n")

    status = scheduler.status()
    assert status.installed and status.loaded
    assert status.when == "daily 04:30"
    assert status.binary == str(tmp_path / "bin" / "jobbot")
    assert status.last_log_line is not None and "Up to date" in status.last_log_line

    assert scheduler.remove().ok
    assert not sched.plist_path(tmp_path).exists()
    assert scheduler.status().installed is False
    assert "No daily update" in scheduler.remove().message


def test_status_does_not_write(tmp_path: Path) -> None:
    runner = FakeRunner()
    _scheduler(tmp_path, "darwin", runner).status()
    _scheduler(tmp_path, "linux", runner).status()

    assert list(tmp_path.iterdir()) == []
    assert all(call[:2] in (["crontab", "-l"], ["launchctl", "print"]) for call in runner.calls)


# cron ---------------------------------------------------------------------------


def test_cron_install_then_remove_restores_other_entries(tmp_path: Path) -> None:
    original = "MAILTO=''\n0 9 * * 1 /usr/bin/backup\n"
    runner = FakeRunner(crontab=original)
    scheduler = _scheduler(tmp_path, "linux", runner)
    spec = scheduler.spec(tmp_path / "bin" / "jobbot")

    assert scheduler.install(spec).ok
    assert scheduler.install(spec).ok
    assert runner.crontab is not None
    assert runner.crontab.count(sched.CRON_MARKER) == 1
    assert runner.crontab.startswith(original)

    status = scheduler.status()
    assert status.installed and status.when == "daily 04:30"
    assert status.binary == str(tmp_path / "bin" / "jobbot")

    assert scheduler.remove().ok
    assert runner.crontab == original


def test_cron_install_without_existing_crontab(tmp_path: Path) -> None:
    runner = FakeRunner(crontab=None)
    scheduler = _scheduler(tmp_path, "linux", runner)

    assert scheduler.install(scheduler.spec(tmp_path / "bin" / "jobbot")).ok
    assert runner.crontab is not None
    assert runner.crontab.splitlines()[0].endswith(sched.CRON_MARKER)
    assert (tmp_path / ".local" / "state" / "jobbot").is_dir()


def test_cron_remove_without_entry_writes_nothing(tmp_path: Path) -> None:
    runner = FakeRunner(crontab="0 9 * * 1 /usr/bin/backup\n")
    result = _scheduler(tmp_path, "linux", runner).remove()

    assert result.ok
    assert ["crontab", "-"] not in runner.calls


def test_unsupported_platform(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, "win32", FakeRunner())
    assert scheduler.install(scheduler.spec(tmp_path / "jobbot")).ok is False
    assert scheduler.remove().ok is False
    assert scheduler.status().backend is None


# the daily run ------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 10, 9, 4, 30, tzinfo=UTC)


def _never_check() -> CheckResult:
    raise AssertionError("must not check")


def _never_update() -> UpdateResult:
    raise AssertionError("must not update")


def test_scheduled_run_skips_in_docker() -> None:
    run = sched.run_scheduled_update(
        env={},
        config_path=None,
        in_docker=lambda: True,
        check=_never_check,
        update=_never_update,
        now=_now,
    )
    assert run.exit_code == 0
    assert "Docker" in run.lines[0]


def test_scheduled_run_respects_opt_out() -> None:
    run = sched.run_scheduled_update(
        env={"JOBBOT_AUTO_UPDATE": "0"},
        config_path=None,
        in_docker=lambda: False,
        check=_never_check,
        update=_never_update,
        now=_now,
    )
    assert run.exit_code == 0
    assert run.lines == [
        "2026-10-09T04:30:00+00:00 Auto-update disabled (JOBBOT_AUTO_UPDATE=0); skipped."
    ]


def test_scheduled_run_up_to_date_does_not_reinstall() -> None:
    run = sched.run_scheduled_update(
        env={},
        config_path=None,
        in_docker=lambda: False,
        check=lambda: CheckResult(status="up_to_date", message="Up to date with main: abc1234"),
        update=_never_update,
        now=_now,
    )
    assert run.exit_code == 0
    assert run.lines[-1].endswith("Up to date with main: abc1234")


def test_scheduled_run_updates_when_behind() -> None:
    calls: list[str] = []

    def update() -> UpdateResult:
        calls.append("uv")
        return UpdateResult(ok=True, message="Updated")

    run = sched.run_scheduled_update(
        env={},
        config_path=None,
        in_docker=lambda: False,
        check=lambda: CheckResult(
            status="available",
            message="Update available",
            installed="a" * 40,
            latest="b" * 40,
        ),
        update=update,
        now=_now,
    )
    assert calls == ["uv"]
    assert run.exit_code == 0
    assert run.lines[-1].endswith("Updated: aaaaaaa → bbbbbbb")


def test_scheduled_run_reports_check_and_update_failures() -> None:
    offline = sched.run_scheduled_update(
        env={},
        config_path=None,
        in_docker=lambda: False,
        check=lambda: CheckResult(status="error", message="Could not reach repo"),
        update=_never_update,
        now=_now,
    )
    failed = sched.run_scheduled_update(
        env={},
        config_path=None,
        in_docker=lambda: False,
        check=lambda: CheckResult(status="unknown", message="Installed commit unknown"),
        update=lambda: UpdateResult(ok=False, message="uv tool install failed"),
        now=_now,
    )
    assert offline.exit_code == 1 and "Check failed" in offline.lines[-1]
    assert failed.exit_code == 1 and "Update failed" in failed.lines[-1]


def test_scheduled_run_leaves_editable_install_alone() -> None:
    run = sched.run_scheduled_update(
        env={},
        config_path=None,
        in_docker=lambda: False,
        check=lambda: CheckResult(status="editable", message="Editable install"),
        update=_never_update,
        now=_now,
    )
    assert run.exit_code == 0


def test_status_lines_cover_installed_and_unsupported(tmp_path: Path) -> None:
    status = sched.ScheduleStatus(
        backend="launchd",
        installed=True,
        loaded=False,
        entry="agent.plist",
        binary="/opt/tools/bin/jobbot",
        when="daily 04:30",
        log_path=tmp_path / "update.log",
        last_log_line="ok",
        opt_out="JOBBOT_AUTO_UPDATE=0",
    )
    text = "\n".join(status.lines())

    assert "Loaded: no" in text
    assert "Runs: /opt/tools/bin/jobbot update --scheduled" in text
    assert "Opt-out: JOBBOT_AUTO_UPDATE=0" in text
    assert "not supported" in sched.ScheduleStatus(backend=None, installed=False).lines()[0]
