"""Open pages in the background: the terminal keeps the focus.

The tab is where the human finishes the step, so instead of raising the window
JobBot prints where the page is. ``JOBBOT_BROWSER_FOCUS=1`` brings back the old
foreground behaviour.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from typing import Any

logger = logging.getLogger("jobbot.browser.background")

FOCUS_ENV = "JOBBOT_BROWSER_FOCUS"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def wants_focus(env: Mapping[str, str] | None = None) -> bool:
    """True only when the user opted back into foreground windows."""
    env_map = env if env is not None else os.environ
    return env_map.get(FOCUS_ENV, "").strip().casefold() in _TRUTHY


def _spawn(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True)  # noqa: S603 — fixed argv, no shell


def macos_open_argv(url: str, *, focus: bool = False, app: str | None = None) -> list[str]:
    """``open -g`` hands the URL to the browser without activating it."""
    argv = ["open"]
    if not focus:
        argv.append("-g")
    if app:
        argv += ["-a", app]
    argv.append(url)
    return argv


def background_notice(
    url: str,
    *,
    where: str = "tu navegador",
    title: str | None = None,
    focus: bool | None = None,
) -> str:
    """The line printed instead of raising the window, so the user can find the tab."""
    if wants_focus() if focus is None else focus:
        return f"Abrí {url} en {where}"
    tab = f" (pestaña: {title})" if title else ""
    return f"Abrí {url} en segundo plano en {where}{tab}; el foco sigue en tu terminal."


def open_url(
    url: str,
    *,
    focus: bool | None = None,
    platform: str | None = None,
    spawn: Callable[[list[str]], object] | None = None,
    fallback: Callable[..., object] | None = None,
) -> str:
    """Open ``url`` in the system browser without stealing focus; returns the notice."""
    raise_window = wants_focus() if focus is None else focus
    system = platform or sys.platform
    browser_open = fallback or webbrowser.open
    if system == "darwin":
        try:
            (spawn or _spawn)(macos_open_argv(url, focus=raise_window))
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("open -g failed (%s); falling back to webbrowser", exc)
            browser_open(url, new=2, autoraise=raise_window)
    else:
        browser_open(url, new=2, autoraise=raise_window)
    return background_notice(url, focus=raise_window)


def _app_bundle(executable: str) -> str | None:
    marker = ".app/Contents/MacOS/"
    index = executable.find(marker)
    if index < 0:
        return None
    return executable[: index + len(".app")]


def background_launch_argv(
    argv: Sequence[str],
    *,
    platform: str | None = None,
    focus: bool | None = None,
) -> list[str]:
    """Launch a Chrome app bundle behind the current window (macOS ``open -g -n -a``).

    ``-n`` is required: without it ``open`` hands the flags to the Chrome already
    running, which ignores ``--remote-debugging-port`` and ``--user-data-dir``.
    Elsewhere there is no portable way to launch unfocused, so argv is unchanged.
    """
    raise_window = wants_focus() if focus is None else focus
    system = platform or sys.platform
    bundle = _app_bundle(argv[0]) if argv else None
    if raise_window or system != "darwin" or bundle is None:
        return list(argv)
    return ["open", "-g", "-n", "-a", bundle, "--args", *argv[1:]]


def launch_detached(argv: Sequence[str], *, focus: bool | None = None) -> list[str]:
    """Start a browser process in the background and return the argv actually run."""
    launched = background_launch_argv(argv, focus=focus)
    subprocess.Popen(launched, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
    return launched


def new_background_page(
    context: Any,
    *,
    browser: Any | None = None,
    focus: bool | None = None,
    timeout_ms: int = 10_000,
) -> Any:
    """New tab in a CDP-attached Chrome that does not become the active tab.

    Playwright's ``context.new_page()`` creates a foreground tab; CDP
    ``Target.createTarget`` with ``background: true`` does not. The target lands in
    the browser's default context, so only that context can use it.
    """
    raise_window = wants_focus() if focus is None else focus
    owner = browser if browser is not None else getattr(context, "browser", None)
    contexts = list(getattr(owner, "contexts", None) or [])
    if raise_window or owner is None or not contexts or contexts[0] is not context:
        return context.new_page()
    cdp = owner.new_browser_cdp_session()
    try:
        with context.expect_page(timeout=timeout_ms) as info:
            cdp.send("Target.createTarget", {"url": "about:blank", "background": True})
        return info.value
    except Exception as exc:  # noqa: BLE001 — Playwright error types vary
        logger.warning("Background tab failed (%s); opening a normal tab", exc)
        return context.new_page()
    finally:
        cdp.detach()
