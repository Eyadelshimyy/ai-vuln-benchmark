"""Challenge / manual interaction helpers."""

from __future__ import annotations

from jobbot.browser.session import BrowserSession


def pause_if_challenge(session: BrowserSession, *, reason: str) -> None:
    session.pause_for_manual(reason)
