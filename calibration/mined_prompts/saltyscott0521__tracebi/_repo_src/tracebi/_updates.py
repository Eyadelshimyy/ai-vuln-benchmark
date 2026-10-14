"""Is there a newer TraceBi, and how does this install get it?

Releases are published by ``.github/workflows/release.yml`` on a ``v*`` tag: a
GitHub release carrying the wheel (with the web UI built in) and a GHCR image
tagged with the version. This module asks GitHub for the latest release and
says how *this* install updates, because that differs:

* **pip** — install the release's wheel (a plain ``git+`` install has no UI).
* **docker** — pull the new image; a container can't upgrade itself.
* **checkout** — an editable install from a git checkout: ``git pull``.

The check is one anonymous GET, cached for a day, and fails quietly offline.
``TRACEBI_UPDATE_CHECK=0`` turns it off; ``TRACEBI_UPDATE_URL`` points it at a
mirror (any URL answering like GitHub's ``releases/latest``). Nothing about the
install or its data is sent.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Optional

from tracebi._version import get_version

RELEASES_URL = "https://api.github.com/repos/saltyscott0521/tracebi/releases/latest"
IMAGE = "ghcr.io/saltyscott0521/tracebi"
CACHE_SECONDS = 24 * 3600
TIMEOUT = 4


def enabled() -> bool:
    return os.environ.get("TRACEBI_UPDATE_CHECK", "1").strip().lower() not in ("0", "false", "no", "off")


def _cache_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "tracebi", "latest_release.json")


# ── Versions ────────────────────────────────────────────────────────────────

_PEP440 = re.compile(
    r"^v?(?P<rel>\d+(?:\.\d+)*)(?:[-_.]?(?P<pre>a|b|rc)(?P<pren>\d*))?"
    r"(?:[-_.]?post(?P<post>\d*))?(?:[-_.]?dev(?P<dev>\d*))?$")


def _fallback_key(v: str):
    """PEP 440's common forms, for when ``packaging`` isn't installed."""
    m = _PEP440.match(v.strip())
    if not m:
        return None
    rel = tuple(int(x) for x in m["rel"].split("."))
    rel += (0,) * (4 - len(rel))
    # within one release number: dev < a < b < rc < final < post
    if m["dev"] is not None and m["pre"] is None and m["post"] is None:
        phase = (-1, 0)
    elif m["pre"]:
        phase = ({"a": 0, "b": 1, "rc": 2}[m["pre"]], int(m["pren"] or 0))
    else:
        phase = (3, 0)
    post = int(m["post"] or 0) if m["post"] is not None else -1
    return rel + phase + (post,)


def _key(v: str):
    """A comparable key for *v*, or None if it isn't a version."""
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return _fallback_key(v)
    try:
        return Version(v.strip().lstrip("v"))
    except InvalidVersion:
        return None


def is_newer(latest: str, current: str) -> bool:
    a, b = _key(latest or ""), _key(current or "")
    return a is not None and b is not None and a > b


# ── The latest release ──────────────────────────────────────────────────────

def _fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"tracebi/{get_version()} (update check)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wheel = next((a.get("browser_download_url") for a in data.get("assets") or []
                  if str(a.get("name", "")).endswith(".whl")), None)
    return {
        "version": str(data.get("tag_name") or "").lstrip("v"),
        "url": data.get("html_url"),
        "published_at": data.get("published_at"),
        "notes": data.get("body") or "",
        "wheel_url": wheel,
    }


def latest_release(force: bool = False) -> Optional[dict]:
    """The latest published release, or None (checks off, offline, or none yet).

    Cached for CACHE_SECONDS in the user's cache directory; *force* skips it.
    """
    if not enabled():
        return None
    url = os.environ.get("TRACEBI_UPDATE_URL") or RELEASES_URL
    path = _cache_path()
    if not force:
        try:
            with open(path, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("url_checked") == url and time.time() - cached.get("checked_at", 0) < CACHE_SECONDS:
                return cached.get("release")
        except (OSError, ValueError):
            pass
    try:
        release = _fetch(url)
        if not release["version"]:
            release = None
    except Exception:  # noqa: BLE001 — offline, rate-limited, or no release yet
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"checked_at": time.time(), "url_checked": url, "release": release}, fh)
    except OSError:
        pass
    return release


# ── How this install updates ────────────────────────────────────────────────

def install_kind() -> str:
    """``docker``, ``checkout`` (editable, from a git clone) or ``pip``."""
    if os.environ.get("TRACEBI_IN_DOCKER") or os.path.exists("/.dockerenv"):
        return "docker"
    try:
        from importlib.metadata import distribution
        direct = distribution("tracebi").read_text("direct_url.json")
        if direct and json.loads(direct).get("dir_info", {}).get("editable"):
            return "checkout"
    except Exception:  # noqa: BLE001 — metadata missing: treat as a plain pip install
        pass
    return "pip"


def update_command(kind: str, release: dict) -> str:
    version = release["version"]
    if kind == "docker":
        return (f"TRACEBI_VERSION={version} docker compose -f deploy/compose.yml pull "
                f"&& TRACEBI_VERSION={version} docker compose -f deploy/compose.yml up -d")
    if kind == "checkout":
        return (f"git fetch --tags && git checkout v{version} && pip install -e . "
                f"&& (cd web/ui && npm ci && npm run build)")
    target = release.get("wheel_url") or (
        f"tracebi @ git+https://github.com/saltyscott0521/tracebi@v{version}")
    return f'pip install --upgrade "{target}"'


_refreshing = False
_last_attempt = 0.0
RETRY_SECONDS = 600          # offline: try again at most every ten minutes


def _cached_release() -> Optional[dict]:
    """The cached release, however old, without touching the network; a
    stale or missing cache starts one background refresh."""
    global _refreshing, _last_attempt
    url = os.environ.get("TRACEBI_UPDATE_URL") or RELEASES_URL
    cached = None
    try:
        with open(_cache_path(), encoding="utf-8") as fh:
            cached = json.load(fh)
    except (OSError, ValueError):
        pass
    fresh = (cached and cached.get("url_checked") == url
             and time.time() - cached.get("checked_at", 0) < CACHE_SECONDS)
    if not fresh and not _refreshing and time.time() - _last_attempt > RETRY_SECONDS:
        import threading

        _refreshing, _last_attempt = True, time.time()

        def refresh():
            global _refreshing
            try:
                latest_release(force=True)
            finally:
                _refreshing = False

        threading.Thread(target=refresh, name="tracebi-update-check", daemon=True).start()
    return (cached or {}).get("release") if cached and cached.get("url_checked") == url else None


def status(force: bool = False, wait: bool = True) -> dict:
    """What the CLI and ``/api/status`` report: current, latest, and how to update."""
    current = get_version()
    out = {"current": current, "checked": enabled(), "available": False,
           "latest": None, "url": None, "kind": install_kind(), "command": None}
    if not enabled():
        release = None
    elif wait:
        release = latest_release(force=force)
    else:                                # the web server: never block a request
        release = _cached_release()
    if release:
        out.update(latest=release["version"], url=release.get("url"))
        if is_newer(release["version"], current):
            out.update(available=True, command=update_command(out["kind"], release),
                       notes=release.get("notes", ""))
    return out
