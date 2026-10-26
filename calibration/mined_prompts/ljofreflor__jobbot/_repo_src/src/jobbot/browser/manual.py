"""Human-in-the-loop pauses for CAPTCHA / consent (no bypass)."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable

logger = logging.getLogger("jobbot.browser.manual")


def wait_for_manual_clear(
    *,
    is_clear: Callable[[], bool],
    timeout_seconds: float = 300.0,
    poll_seconds: float = 2.0,
    stdin_available: bool | None = None,
    read_enter: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> bool:
    """Wait until ``is_clear`` is true, or until timeout.

    Prefer Enter on a TTY when available; otherwise poll ``is_clear`` so agent
    runs (no stdin) still leave the browser open for the human.
    Returns True if cleared, False on timeout.
    """
    use_stdin = sys.stdin.isatty() if stdin_available is None else stdin_available
    enter = read_enter if read_enter is not None else input

    if use_stdin:
        try:
            enter()
        except EOFError:
            logger.warning("stdin EOF during manual pause; falling back to poll")
        else:
            return bool(is_clear())

    deadline = now() + timeout_seconds
    logger.info(
        "No interactive stdin — polling up to %.0fs for challenge clear",
        timeout_seconds,
    )
    while now() < deadline:
        if is_clear():
            return True
        sleep(poll_seconds)
    return bool(is_clear())
