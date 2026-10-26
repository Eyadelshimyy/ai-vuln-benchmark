"""Reinstall / upgrade the ``jobbot`` executable on PATH."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

DEFAULT_REPO = "https://github.com/ljofreflor/jobbot"
DEFAULT_REF = "main"
LS_REMOTE_TIMEOUT_S = 10.0

Runner = Callable[..., Any]
"""``subprocess.run``-compatible callable (injected by tests)."""

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class UpdateResult:
    ok: bool
    message: str
    version_line: str | None = None


CheckStatus = Literal["up_to_date", "available", "unknown", "editable", "error"]


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    message: str
    installed: str | None = None
    latest: str | None = None


@dataclass(frozen=True)
class InstallInfo:
    """What ``direct_url.json`` says about the installed distribution."""

    commit: str | None = None
    requested_ref: str | None = None
    editable: bool = False


def _in_docker() -> bool:
    if Path("/.dockerenv").is_file():
        return True
    cgroup = Path("/proc/1/cgroup")
    if cgroup.is_file():
        text = cgroup.read_text(encoding="utf-8", errors="replace")
        return "docker" in text or "containerd" in text
    return False


def update_jobbot(
    *,
    ref: str | None = None,
    repo: str | None = None,
    source: str | None = None,
) -> UpdateResult:
    """
    Upgrade the uv-tool install of JobBot (same path as ``scripts/install.sh``).

    ``JOBBOT_REF`` / ``JOBBOT_REPO`` / ``JOBBOT_SOURCE`` env vars apply when args
    are omitted. Inside Docker, returns a pull hint instead of mutating the image.
    """
    if _in_docker():
        return UpdateResult(ok=False, message=docker_hint())

    uv = shutil.which("uv")
    if uv is None:
        return UpdateResult(
            ok=False,
            message=(
                "uv not found on PATH. Re-run the installer:\n"
                "  curl -fsSL https://raw.githubusercontent.com/ljofreflor/jobbot/"
                "main/scripts/install.sh | bash"
            ),
        )

    if source is not None:
        resolved_source = source.strip()
    else:
        resolved_source = os.environ.get("JOBBOT_SOURCE", "").strip()
    resolved_repo = (repo or os.environ.get("JOBBOT_REPO") or DEFAULT_REPO).strip()
    resolved_ref = (ref or os.environ.get("JOBBOT_REF") or DEFAULT_REF).strip()

    if resolved_source:
        if not resolved_source.startswith("wheel:"):
            return UpdateResult(
                ok=False,
                message=(f"Unknown JOBBOT_SOURCE={resolved_source!r} (use wheel:/path/to.whl)"),
            )
        target = resolved_source.removeprefix("wheel:")
        if not Path(target).is_file():
            return UpdateResult(ok=False, message=f"wheel not found: {target}")
        pkg = target
        label = f"wheel {target}"
    else:
        pkg = f"git+{resolved_repo}@{resolved_ref}"
        label = pkg

    cmd = [uv, "tool", "install", "--force", "--python", "3.12", pkg]
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return UpdateResult(ok=False, message=f"Failed to run uv: {exc}")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return UpdateResult(
            ok=False,
            message=f"uv tool install failed for {label}\n{detail}",
        )

    version_line = _read_version()
    return UpdateResult(
        ok=True,
        message=f"Updated JobBot from {label}",
        version_line=version_line,
    )


def _read_version() -> str | None:
    jobbot = shutil.which("jobbot")
    if jobbot is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [jobbot, "version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip() or None


def docker_hint() -> str:
    return (
        "Running inside Docker — update the image on the host:\n"
        "  docker pull ghcr.io/ljofreflor/jobbot:latest\n"
        "  # or: docker compose pull && docker compose build"
    )


def parse_direct_url(text: str | None) -> InstallInfo:
    """Read commit / ref / editable from a PEP 610 ``direct_url.json`` payload."""
    if not text:
        return InstallInfo()
    try:
        data = json.loads(text)
    except ValueError:
        return InstallInfo()
    if not isinstance(data, dict):
        return InstallInfo()
    vcs = data.get("vcs_info")
    dir_info = data.get("dir_info")
    commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
    requested = vcs.get("requested_revision") if isinstance(vcs, dict) else None
    editable = bool(dir_info.get("editable")) if isinstance(dir_info, dict) else False
    return InstallInfo(
        commit=commit if isinstance(commit, str) and commit else None,
        requested_ref=requested if isinstance(requested, str) and requested else None,
        editable=editable,
    )


def read_direct_url() -> str | None:
    try:
        return metadata.distribution("jobbot").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None


def installed_info(read: Callable[[], str | None] | None = None) -> InstallInfo:
    return parse_direct_url((read or read_direct_url)())


def parse_ls_remote(output: str, ref: str) -> str | None:
    """Commit a ref points to in ``git ls-remote`` output (peeled tag wins)."""
    rows: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2 and _FULL_SHA.match(parts[0]):
            rows[parts[1]] = parts[0]
    for name in (
        f"refs/heads/{ref}",
        f"refs/tags/{ref}^{{}}",
        f"refs/tags/{ref}",
        ref,
    ):
        if name in rows:
            return rows[name]
    return next(iter(rows.values()), None)


def latest_commit(
    repo: str,
    ref: str,
    *,
    runner: Runner = subprocess.run,
    git: str | None = None,
) -> str:
    """Commit at ``ref`` on the remote, without cloning. Raises ``RuntimeError``."""
    if _FULL_SHA.match(ref.casefold()):
        return ref.casefold()
    git_bin = git or shutil.which("git")
    if git_bin is None:
        raise RuntimeError("git not found on PATH (needed for git ls-remote)")
    try:
        completed = runner(  # noqa: S603 — fixed argv, no shell
            [git_bin, "ls-remote", repo, ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=LS_REMOTE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"git ls-remote timed out after {LS_REMOTE_TIMEOUT_S:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to run git: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Could not reach {repo}: {detail or 'git ls-remote failed'}")
    commit = parse_ls_remote(completed.stdout or "", ref)
    if commit is None:
        raise RuntimeError(f"Ref {ref!r} not found on {repo}")
    return commit


def short_commit(commit: str | None) -> str:
    return commit[:7] if commit else "unknown"


def check_for_update(
    *,
    ref: str | None = None,
    repo: str | None = None,
    runner: Runner = subprocess.run,
    git: str | None = None,
    read: Callable[[], str | None] | None = None,
) -> CheckResult:
    """Compare the installed commit with the latest one at ``ref``; installs nothing."""
    resolved_repo = (repo or os.environ.get("JOBBOT_REPO") or DEFAULT_REPO).strip()
    resolved_ref = (ref or os.environ.get("JOBBOT_REF") or DEFAULT_REF).strip()
    info = installed_info(read)
    try:
        latest = latest_commit(resolved_repo, resolved_ref, runner=runner, git=git)
    except RuntimeError as exc:
        return CheckResult(status="error", message=str(exc), installed=info.commit)
    if info.editable:
        return CheckResult(
            status="editable",
            message=(
                f"Editable install (development checkout): use git pull. "
                f"{resolved_ref} is at {short_commit(latest)}."
            ),
            latest=latest,
        )
    if info.commit is None:
        return CheckResult(
            status="unknown",
            message=(
                f"Installed commit unknown (not a git install). "
                f"{resolved_ref} is at {short_commit(latest)}."
            ),
            latest=latest,
        )
    if info.commit == latest:
        return CheckResult(
            status="up_to_date",
            message=f"Up to date with {resolved_ref}: {short_commit(latest)}",
            installed=info.commit,
            latest=latest,
        )
    return CheckResult(
        status="available",
        message=(
            f"Update available on {resolved_ref}: "
            f"{short_commit(info.commit)} → {short_commit(latest)}"
        ),
        installed=info.commit,
        latest=latest,
    )
