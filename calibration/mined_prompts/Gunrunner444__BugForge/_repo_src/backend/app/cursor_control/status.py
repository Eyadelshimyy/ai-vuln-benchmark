"""Read-only status for Cursor-controlled operation. Does not call a model."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx

_GDK_URL = "http://127.0.0.1:3000/users/sign_in"


def git_identity(repo_root: Path) -> dict[str, str]:
    def rev(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "branch": rev("branch", "--show-current"),
        "commit": rev("rev-parse", "HEAD"),
    }


def gdk_connectivity() -> dict[str, Any]:
    """Probe the local GDK sign-in page. The URL is fixed; callers cannot retarget it."""

    try:
        response = httpx.get(_GDK_URL, timeout=0.8, follow_redirects=False)
    except httpx.HTTPError as exc:
        return {"url": _GDK_URL, "reachable": False, "error": type(exc).__name__}
    return {
        "url": _GDK_URL,
        "reachable": response.status_code < 500,
        "status": response.status_code,
    }
