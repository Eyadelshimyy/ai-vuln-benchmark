"""Helpers for attaching to a user-launched Chrome (CDP) — HITL, no CAPTCHA bypass."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_CDP_URL = "http://127.0.0.1:9222"
DEFAULT_CDP_PORT = 9222


def resolve_cdp_url(
    explicit: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str | None:
    """Prefer CLI/explicit, then JOBBOT_CDP_URL. Empty string means disabled."""
    if explicit is not None:
        return explicit.strip() or None
    env_map = env if env is not None else os.environ
    raw = env_map.get("JOBBOT_CDP_URL")
    if raw is None:
        return None
    return raw.strip() or None


def find_chrome_executable() -> Path | None:
    """Find any Chromium-based browser (Chrome, Edge, Brave, Chromium).
    
    Searches in order of preference:
    1. Platform-specific paths (Mac apps)
    2. Common executable names in PATH
    
    Returns the first browser found, or None if no Chromium browser is available.
    """
    # Mac: Check application bundles
    mac_browsers = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]
    for browser in mac_browsers:
        if browser.is_file():
            return browser
    
    # Linux/Windows: Check PATH for executables
    # Order: Chrome -> Edge -> Brave -> Chromium
    executables = [
        "google-chrome",     # Linux Chrome
        "chrome",            # Windows Chrome shorthand
        "microsoft-edge",    # Linux Edge
        "msedge",           # Windows Edge
        "brave-browser",     # Linux Brave
        "brave",            # Windows/Mac Brave
        "chromium-browser",  # Linux Chromium
        "chromium",         # General Chromium
    ]
    
    for exe_name in executables:
        found = shutil.which(exe_name)
        if found:
            return Path(found)
    
    return None


def chrome_debug_argv(
    *,
    profile_dir: Path,
    port: int = DEFAULT_CDP_PORT,
    start_url: str = "https://cl.indeed.com/",
    chrome: Path | None = None,
) -> list[str]:
    """Argv to launch a Chromium browser with remote debugging (user owns the window).
    
    Works with Chrome, Edge, Brave, or any Chromium-based browser.
    """
    exe = chrome or find_chrome_executable()
    if exe is None:
        msg = (
            "No Chromium-based browser found. "
            "Install Chrome, Edge, Brave, or Chromium, or pass chrome=…"
        )
        raise FileNotFoundError(msg)
    return [
        str(exe),
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        start_url,
    ]


def cdp_http_url(port: int = DEFAULT_CDP_PORT) -> str:
    return f"http://127.0.0.1:{port}"
