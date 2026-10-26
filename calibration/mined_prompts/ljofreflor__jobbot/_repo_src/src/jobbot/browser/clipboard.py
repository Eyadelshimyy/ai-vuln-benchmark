"""Clipboard helper for HITL Indeed edits (macOS pbcopy; no CAPTCHA bypass)."""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger("jobbot.browser.clipboard")


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the system clipboard. Returns True on success."""
    if shutil.which("pbcopy"):
        try:
            subprocess.run(  # noqa: S603
                ["pbcopy"],
                input=text.encode("utf-8"),
                check=True,
            )
            return True
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("pbcopy failed: %s", exc)
            return False
    if shutil.which("xclip"):
        try:
            subprocess.run(  # noqa: S603
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=True,
            )
            return True
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("xclip failed: %s", exc)
            return False
    logger.warning("No clipboard tool (pbcopy/xclip) available")
    return False
