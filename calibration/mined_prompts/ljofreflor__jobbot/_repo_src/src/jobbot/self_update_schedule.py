"""Daily ``jobbot update`` via the user's scheduler (launchd on macOS, crontab on Linux)."""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from jobbot.self_update import (
    CheckResult,
    Runner,
    UpdateResult,
    check_for_update,
    short_commit,
    update_jobbot,
)

LAUNCHD_LABEL = "com.ljofreflor.jobbot.update"
CRON_MARKER = "# jobbot:auto-update"
DEFAULT_HOUR = 4
DEFAULT_MINUTE = 30
SCHEDULED_ARGS = ("update", "--scheduled")

_OFF_VALUES = frozenset({"0", "false", "no", "off"})
_ON_VALUES = frozenset({"1", "true", "yes", "on"})
_SYSTEM_PATH = {
    "darwin": ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"),
    "linux": ("/usr/local/bin", "/usr/bin", "/bin"),
}

Backend = Literal["launchd", "cron"]


class ScheduleAction(StrEnum):
    daily = "daily"
    off = "off"
    status = "status"


@dataclass(frozen=True)
class ScheduleSpec:
    binary: Path
    log_path: Path
    path_env: str
    working_dir: Path | None = None
    hour: int = DEFAULT_HOUR
    minute: int = DEFAULT_MINUTE


@dataclass(frozen=True)
class ScheduleResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class ScheduleStatus:
    backend: Backend | None
    installed: bool
    loaded: bool | None = None
    entry: str | None = None
    binary: str | None = None
    when: str | None = None
    log_path: Path | None = None
    last_log_line: str | None = None
    opt_out: str | None = None

    def lines(self) -> list[str]:
        if self.backend is None:
            return ["Scheduling is not supported on this platform (macOS launchd / Linux crontab)."]
        out = [f"Scheduler: {self.backend}"]
        out.append(f"Installed: {'yes' if self.installed else 'no'}")
        if self.loaded is not None:
            out.append(f"Loaded: {'yes' if self.loaded else 'no'}")
        if self.entry:
            out.append(f"Entry: {self.entry}")
        if self.binary:
            out.append(f"Runs: {self.binary} {' '.join(SCHEDULED_ARGS)}")
        if self.when:
            out.append(f"When: {self.when}")
        if self.log_path is not None:
            out.append(f"Log: {self.log_path}")
        if self.last_log_line:
            out.append(f"Last log line: {self.last_log_line}")
        out.append(f"Opt-out: {self.opt_out or 'not set'}")
        return out


def backend_for(platform: str) -> Backend | None:
    if platform == "darwin":
        return "launchd"
    if platform.startswith("linux"):
        return "cron"
    return None


def plist_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def log_path(platform: str, home: Path, env: Mapping[str, str]) -> Path:
    if platform == "darwin":
        return home / "Library" / "Logs" / "jobbot" / "update.log"
    state = env.get("XDG_STATE_HOME", "").strip()
    base = Path(state) if state else home / ".local" / "state"
    return base / "jobbot" / "update.log"


def scheduled_path_env(dirs: Iterable[str | Path | None], platform: str) -> str:
    """PATH for the job: launchd / cron do not inherit the shell's."""
    ordered: list[str] = []
    for entry in (*dirs, *_SYSTEM_PATH.get(platform, _SYSTEM_PATH["linux"])):
        if entry is None:
            continue
        text = str(entry)
        if text and text not in ordered:
            ordered.append(text)
    return ":".join(ordered)


def build_launchd_plist(spec: ScheduleSpec) -> bytes:
    payload: dict[str, Any] = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [str(spec.binary), *SCHEDULED_ARGS],
        "StartCalendarInterval": {"Hour": spec.hour, "Minute": spec.minute},
        "RunAtLoad": False,
        "EnvironmentVariables": {"PATH": spec.path_env},
        "StandardOutPath": str(spec.log_path),
        "StandardErrorPath": str(spec.log_path),
        "ProcessType": "Background",
    }
    if spec.working_dir is not None:
        payload["WorkingDirectory"] = str(spec.working_dir)
    return plistlib.dumps(payload, sort_keys=True)


def build_cron_line(spec: ScheduleSpec) -> str:
    # cron turns an unescaped % into a newline.
    def q(value: str) -> str:
        return shlex.quote(value).replace("%", "\\%")

    command = " ".join(
        [
            f"PATH={q(spec.path_env)}",
            q(str(spec.binary)),
            *SCHEDULED_ARGS,
            f">> {q(str(spec.log_path))} 2>&1",
        ]
    )
    return f"{spec.minute} {spec.hour} * * * {command} {CRON_MARKER}"


def merge_crontab(existing: str, line: str | None) -> str:
    """Drop every line carrying the marker, then append ``line`` (``None`` removes)."""
    kept = [row for row in existing.splitlines() if CRON_MARKER not in row]
    while kept and not kept[-1].strip():
        kept.pop()
    if line is not None:
        kept.append(line)
    return "\n".join(kept) + "\n" if kept else ""


def auto_update_opt_out(env: Mapping[str, str], config_path: Path | None) -> str | None:
    """Why the scheduled run must not update, or ``None`` when it may."""
    flag = env.get("JOBBOT_AUTO_UPDATE", "").strip().casefold()
    if flag in _OFF_VALUES:
        return f"JOBBOT_AUTO_UPDATE={env['JOBBOT_AUTO_UPDATE'].strip()}"
    legacy = env.get("JOBBOT_NO_AUTO_UPDATE", "").strip().casefold()
    if legacy in _ON_VALUES:
        return f"JOBBOT_NO_AUTO_UPDATE={env['JOBBOT_NO_AUTO_UPDATE'].strip()}"
    if config_path is not None and config_path.is_file():
        try:
            with config_path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            return None
        section = data.get("update")
        if isinstance(section, dict) and section.get("auto") is False:
            return f"[update] auto = false in {config_path}"
    return None


def user_config_path(home: Path) -> Path:
    return home / ".config" / "jobbot" / "config.toml"


def resolve_binary(
    which: Callable[[str], str | None] = shutil.which,
    argv0: str = sys.argv[0],
) -> Path | None:
    """Absolute path of the ``jobbot`` on PATH (the uv-tool shim; symlinks kept)."""
    found = which("jobbot")
    if found:
        return Path(os.path.abspath(found))
    candidate = Path(argv0)
    if candidate.name == "jobbot" and candidate.is_file():
        return Path(os.path.abspath(candidate))
    return None


def _tail(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return None


@dataclass
class Scheduler:
    """Install / remove / inspect the daily job. ``runner`` is ``subprocess.run``-like."""

    platform: str = sys.platform
    home: Path = field(default_factory=Path.home)
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    runner: Runner = subprocess.run
    uid: int = field(default_factory=lambda: int(getattr(os, "getuid", lambda: 0)()))

    @property
    def backend(self) -> Backend | None:
        return backend_for(self.platform)

    @property
    def log_path(self) -> Path:
        return log_path(self.platform, self.home, self.env)

    def spec(self, binary: Path, extra_path: Iterable[str | Path | None] = ()) -> ScheduleSpec:
        path_env = scheduled_path_env([*extra_path, binary.parent], self.platform)
        return ScheduleSpec(
            binary=binary, log_path=self.log_path, path_env=path_env, working_dir=self.home
        )

    def install(self, spec: ScheduleSpec) -> ScheduleResult:
        if self.backend == "launchd":
            return self._install_launchd(spec)
        if self.backend == "cron":
            return self._install_cron(spec)
        return _unsupported()

    def remove(self) -> ScheduleResult:
        if self.backend == "launchd":
            return self._remove_launchd()
        if self.backend == "cron":
            return self._remove_cron()
        return _unsupported()

    def status(self) -> ScheduleStatus:
        opt_out = auto_update_opt_out(self.env, user_config_path(self.home))
        if self.backend == "launchd":
            return self._status_launchd(opt_out)
        if self.backend == "cron":
            return self._status_cron(opt_out)
        return ScheduleStatus(backend=None, installed=False, opt_out=opt_out)

    # launchd -----------------------------------------------------------------

    @property
    def _domain(self) -> str:
        return f"gui/{self.uid}"

    def _launchctl(self, *args: str) -> Any:
        return self.runner(  # noqa: S603 — fixed argv, no shell
            ["launchctl", *args], check=False, capture_output=True, text=True
        )

    def _install_launchd(self, spec: ScheduleSpec) -> ScheduleResult:
        target = plist_path(self.home)
        try:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(build_launchd_plist(spec))
        except OSError as exc:
            return ScheduleResult(ok=False, message=f"Could not write {target}: {exc}")
        try:
            self._launchctl("bootout", f"{self._domain}/{LAUNCHD_LABEL}")
            loaded = self._launchctl("bootstrap", self._domain, str(target))
        except OSError as exc:
            return ScheduleResult(ok=False, message=f"Failed to run launchctl: {exc}")
        if loaded.returncode != 0:
            detail = (loaded.stderr or loaded.stdout or "").strip()
            return ScheduleResult(ok=False, message=f"launchctl bootstrap failed: {detail}")
        return ScheduleResult(
            ok=True,
            message=(
                f"Daily update scheduled at {spec.hour:02d}:{spec.minute:02d} "
                f"(launchd {LAUNCHD_LABEL}).\nPlist: {target}\nLog: {spec.log_path}"
            ),
        )

    def _remove_launchd(self) -> ScheduleResult:
        target = plist_path(self.home)
        try:
            self._launchctl("bootout", f"{self._domain}/{LAUNCHD_LABEL}")
        except OSError as exc:
            return ScheduleResult(ok=False, message=f"Failed to run launchctl: {exc}")
        if not target.is_file():
            return ScheduleResult(ok=True, message="No daily update was scheduled.")
        try:
            target.unlink()
        except OSError as exc:
            return ScheduleResult(ok=False, message=f"Could not remove {target}: {exc}")
        return ScheduleResult(ok=True, message=f"Daily update removed ({target}).")

    def _status_launchd(self, opt_out: str | None) -> ScheduleStatus:
        target = plist_path(self.home)
        if not target.is_file():
            return ScheduleStatus(
                backend="launchd", installed=False, log_path=self.log_path, opt_out=opt_out
            )
        binary: str | None = None
        when: str | None = None
        try:
            data = plistlib.loads(target.read_bytes())
            args = data.get("ProgramArguments") or []
            binary = str(args[0]) if args else None
            interval = data.get("StartCalendarInterval") or {}
            when = f"daily {int(interval['Hour']):02d}:{int(interval['Minute']):02d}"
        except (OSError, ValueError, KeyError, TypeError, plistlib.InvalidFileException):
            pass
        try:
            loaded = self._launchctl("print", f"{self._domain}/{LAUNCHD_LABEL}").returncode == 0
        except OSError:
            loaded = False
        return ScheduleStatus(
            backend="launchd",
            installed=True,
            loaded=loaded,
            entry=str(target),
            binary=binary,
            when=when,
            log_path=self.log_path,
            last_log_line=_tail(self.log_path),
            opt_out=opt_out,
        )

    # cron --------------------------------------------------------------------

    def _read_crontab(self) -> str:
        completed = self.runner(  # noqa: S603 — fixed argv, no shell
            ["crontab", "-l"], check=False, capture_output=True, text=True
        )
        if completed.returncode == 0:
            return completed.stdout or ""
        detail = (completed.stderr or "").casefold()
        if "no crontab" in detail or not detail.strip():
            return ""
        raise RuntimeError(f"crontab -l failed: {(completed.stderr or '').strip()}")

    def _write_crontab(self, content: str) -> None:
        completed = self.runner(  # noqa: S603 — fixed argv, no shell
            ["crontab", "-"], input=content, check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"crontab - failed: {detail}")

    def _install_cron(self, spec: ScheduleSpec) -> ScheduleResult:
        try:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ScheduleResult(
                ok=False, message=f"Could not create {spec.log_path.parent}: {exc}"
            )
        try:
            current = self._read_crontab()
            self._write_crontab(merge_crontab(current, build_cron_line(spec)))
        except (OSError, RuntimeError) as exc:
            return ScheduleResult(ok=False, message=str(exc))
        return ScheduleResult(
            ok=True,
            message=(
                f"Daily update scheduled at {spec.hour:02d}:{spec.minute:02d} "
                f"(crontab, marker {CRON_MARKER!r}).\nLog: {spec.log_path}"
            ),
        )

    def _remove_cron(self) -> ScheduleResult:
        try:
            current = self._read_crontab()
            if CRON_MARKER not in current:
                return ScheduleResult(ok=True, message="No daily update was scheduled.")
            self._write_crontab(merge_crontab(current, None))
        except (OSError, RuntimeError) as exc:
            return ScheduleResult(ok=False, message=str(exc))
        return ScheduleResult(ok=True, message="Daily update removed from crontab.")

    def _status_cron(self, opt_out: str | None) -> ScheduleStatus:
        try:
            current = self._read_crontab()
        except (OSError, RuntimeError):
            current = ""
        entry = next((row for row in current.splitlines() if CRON_MARKER in row), None)
        when: str | None = None
        binary: str | None = None
        if entry is not None:
            fields = entry.split()
            if len(fields) > 5 and fields[0].isdigit() and fields[1].isdigit():
                when = f"daily {int(fields[1]):02d}:{int(fields[0]):02d}"
            try:
                tokens = shlex.split(entry.split(CRON_MARKER)[0])
                index = tokens.index(SCHEDULED_ARGS[0])
                binary = tokens[index - 1]
            except (ValueError, IndexError):
                binary = None
        return ScheduleStatus(
            backend="cron",
            installed=entry is not None,
            entry=entry,
            binary=binary,
            when=when,
            log_path=self.log_path,
            last_log_line=_tail(self.log_path),
            opt_out=opt_out,
        )


def _unsupported() -> ScheduleResult:
    return ScheduleResult(
        ok=False,
        message="Scheduling is not supported on this platform (macOS launchd / Linux crontab).",
    )


@dataclass(frozen=True)
class ScheduledRun:
    exit_code: int
    lines: list[str]


def run_scheduled_update(
    *,
    env: Mapping[str, str],
    config_path: Path | None,
    in_docker: Callable[[], bool],
    check: Callable[[], CheckResult] = check_for_update,
    update: Callable[[], UpdateResult] = update_jobbot,
    now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> ScheduledRun:
    """What the daily job does: skip (Docker / opt-out / up to date) or update."""
    lines: list[str] = []

    def log(text: str) -> None:
        lines.append(f"{now().isoformat(timespec='seconds')} {text}")

    if in_docker():
        log("Docker: skipped (update the image on the host).")
        return ScheduledRun(0, lines)
    reason = auto_update_opt_out(env, config_path)
    if reason is not None:
        log(f"Auto-update disabled ({reason}); skipped.")
        return ScheduledRun(0, lines)
    result = check()
    if result.status == "error":
        log(f"Check failed: {result.message}")
        return ScheduledRun(1, lines)
    if result.status in ("up_to_date", "editable"):
        log(result.message)
        return ScheduledRun(0, lines)
    log(result.message)
    updated = update()
    if not updated.ok:
        log(f"Update failed: {updated.message}")
        return ScheduledRun(1, lines)
    log(f"Updated: {short_commit(result.installed)} → {short_commit(result.latest)}")
    return ScheduledRun(0, lines)
