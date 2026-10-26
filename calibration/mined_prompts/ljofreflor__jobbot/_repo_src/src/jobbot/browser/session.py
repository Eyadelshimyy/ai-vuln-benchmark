"""Persistent Playwright browser session."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from jobbot.browser.background import new_background_page
from jobbot.browser.sessions import ensure_profile_free

logger = logging.getLogger("jobbot.browser.session")


@dataclass(frozen=True)
class BrowserTimeouts:
    navigation_ms: int = 30_000
    action_ms: int = 10_000
    manual_interaction_seconds: int | None = None


def launch_persistent_kwargs(
    *,
    profile_dir: Path,
    headless: bool = False,
    slow_mo: float = 0,
    channel: str | None = "chrome",
) -> dict[str, Any]:
    """Build Playwright launch_persistent_context kwargs (unit-testable)."""
    kwargs: dict[str, Any] = {
        "user_data_dir": str(profile_dir),
        "headless": headless,
        "slow_mo": slow_mo,
        "viewport": {"width": 1280, "height": 900},
    }
    if channel:
        kwargs["channel"] = channel
    return kwargs


class BrowserSession:
    """Chromium / Chrome session — launch or attach via CDP (HITL)."""

    def __init__(
        self,
        profile_dir: Path,
        *,
        headless: bool = False,
        slow_mo: float = 0,
        channel: str | None = "chrome",
        cdp_url: str | None = None,
        timeouts: BrowserTimeouts | None = None,
        debug_root: Path | None = None,
    ) -> None:
        self.profile_dir = profile_dir
        self.headless = headless
        self.slow_mo = slow_mo
        # Prefer installed Chrome over bundled Chromium — Turnstile often loops on
        # automated Chromium. Not a CAPTCHA bypass; still HITL for challenges.
        self.channel = channel
        self.cdp_url = cdp_url
        self.timeouts = timeouts or BrowserTimeouts()
        self.debug_root = debug_root or Path("output/debug")
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._owns_context = True

    @property
    def page(self) -> Page:
        if self._page is None:
            msg = "BrowserSession is not started"
            raise RuntimeError(msg)
        return self._page

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            msg = "BrowserSession is not started"
            raise RuntimeError(msg)
        return self._context

    def __enter__(self) -> Self:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        if not self.cdp_url:
            ensure_profile_free(self.profile_dir)
        self._playwright = sync_playwright().start()
        if self.cdp_url:
            self._attach_cdp(self.cdp_url)
        else:
            self._context = self._launch_context()
            self._owns_context = True
            self._page = (
                self._context.pages[0] if self._context.pages else self._context.new_page()
            )
        self.context.set_default_navigation_timeout(self.timeouts.navigation_ms)
        self.context.set_default_timeout(self.timeouts.action_ms)
        return self

    def _attach_cdp(self, cdp_url: str) -> None:
        assert self._playwright is not None
        logger.info("Attaching to Chrome via CDP: %s", cdp_url)
        self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
        self._owns_context = False
        if self._browser.contexts:
            self._context = self._browser.contexts[0]
        else:
            self._context = self._browser.new_context()
        self._page = self._context.pages[0] if self._context.pages else self.new_tab()

    def new_tab(self) -> Page:
        """A new tab; attached to the user's Chrome it opens in the background."""
        if self._browser is None:
            return self.context.new_page()
        page: Page = new_background_page(self.context, browser=self._browser)
        return page

    def _launch_context(self) -> BrowserContext:
        assert self._playwright is not None
        kwargs = launch_persistent_kwargs(
            profile_dir=self.profile_dir,
            headless=self.headless,
            slow_mo=self.slow_mo,
            channel=self.channel,
        )
        try:
            return self._playwright.chromium.launch_persistent_context(**kwargs)
        except Exception as exc:
            if not self.channel:
                raise
            logger.warning(
                "Browser channel=%s failed (%s); falling back to bundled Chromium",
                self.channel,
                exc,
            )
            kwargs.pop("channel", None)
            return self._playwright.chromium.launch_persistent_context(**kwargs)

    def __exit__(self, *args: object) -> None:
        # CDP: disconnect only — never close the user's Chrome window.
        if self._owns_context and self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def pause_for_manual(
        self,
        reason: str,
        *,
        is_clear: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
        continue_sentinel: Path | None = None,
    ) -> bool:
        """Pause for CAPTCHA/2FA/consent — keep browser open. Returns True if clear."""
        from jobbot.browser.manual import wait_for_manual_clear

        timeout = timeout_seconds
        if timeout is None:
            timeout = float(self.timeouts.manual_interaction_seconds or 300)
        sentinel = continue_sentinel
        console_msg = (
            "\nAUTOMATION PAUSED\n"
            "Manual interaction required.\n"
            f"Reason:\n{reason}\n"
            "Complete the requested action in the browser.\n"
            "Press Enter to continue (or wait — agent runs poll until clear).\n"
        )
        if sentinel is not None:
            console_msg += f"Agent runs: touch {sentinel} when done.\n"
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            if sentinel.exists():
                sentinel.unlink()
        print(console_msg)

        def _ready() -> bool:
            if is_clear is not None and is_clear():
                return True
            return bool(sentinel is not None and sentinel.exists())

        if is_clear is None and sentinel is None:
            try:
                if sys.stdin.isatty():
                    input()
                    return True
            except EOFError:
                pass
            logger.warning("No stdin / clear-check for manual pause; continuing")
            return False

        return wait_for_manual_clear(
            is_clear=_ready,
            timeout_seconds=timeout,
            poll_seconds=2.0,
        )

    def dump_debug(self, adapter: str, action: str, **extra: Any) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = self.debug_root / f"{stamp}-{adapter}-{action}"
        out.mkdir(parents=True, exist_ok=True)
        page = self.page
        screenshot = out / "screenshot.png"
        html_path = out / "page.html"
        meta_path = out / "metadata.json"
        try:
            page.screenshot(path=str(screenshot), full_page=True)
        except Exception as exc:  # noqa: BLE001 — debug best-effort
            logger.warning("screenshot failed: %s", exc)
        try:
            html_path.write_text(page.content(), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("html dump failed: %s", exc)
        meta = {
            "adapter": adapter,
            "action": action,
            "url": page.url,
            "timestamp": stamp,
            **extra,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Debug snapshot: %s", out)
        return out
