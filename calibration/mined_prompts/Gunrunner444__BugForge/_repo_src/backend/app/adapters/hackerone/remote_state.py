"""Map HackerOne remote report/intent states without collapsing unknowns."""

from __future__ import annotations

from enum import StrEnum


class HackerOneRemoteState(StrEnum):
    UNKNOWN = "unknown"
    NEW = "new"
    PENDING = "pending"
    TRIAGED = "triaged"
    RESOLVED = "resolved"
    CLOSED = "closed"
    DUPLICATE = "duplicate"
    INFORMATIVE = "informative"
    NOT_APPLICABLE = "not_applicable"
    SPAM = "spam"
    RETESTING = "retesting"
    NEEDS_MORE_INFO = "needs-more-info"
    NOT_FETCHED = "not_fetched"


_KNOWN: dict[str, HackerOneRemoteState] = {
    "new": HackerOneRemoteState.NEW,
    "pending": HackerOneRemoteState.PENDING,
    "pending-triage": HackerOneRemoteState.PENDING,
    "triaged": HackerOneRemoteState.TRIAGED,
    "resolved": HackerOneRemoteState.RESOLVED,
    "closed": HackerOneRemoteState.CLOSED,
    "duplicate": HackerOneRemoteState.DUPLICATE,
    "informative": HackerOneRemoteState.INFORMATIVE,
    "not-applicable": HackerOneRemoteState.NOT_APPLICABLE,
    "spam": HackerOneRemoteState.SPAM,
    "retesting": HackerOneRemoteState.RETESTING,
    "needs-more-info": HackerOneRemoteState.NEEDS_MORE_INFO,
}


def map_remote_state(raw: object) -> tuple[HackerOneRemoteState, str]:
    """Return (mapped, raw). Unknown values stay UNKNOWN with the original string.

    Local BugForge ``submitted`` is never treated as TRIAGED/RESOLVED/CLOSED.
    """
    text = str(raw or "").strip()
    if not text:
        return HackerOneRemoteState.NOT_FETCHED, ""
    key = text.lower().replace("_", "-")
    mapped = _KNOWN.get(key)
    if mapped is None:
        return HackerOneRemoteState.UNKNOWN, text
    return mapped, text
