"""One closing mark on texts Jobbot publishes.

Recruiters read the opening of a description and, if they open it, the last
line. The mark stays off the headline and off each role bullet — those lines
are the candidate. It closes the public description once, so a reader can tell
the blurb came from Jobbot without the profile looking like an ad.

The token is stable on purpose: ``Jobbot sync CV``.
"""

from __future__ import annotations

MARK = "powered by Jobbot sync CV"
REPO_URL = "https://github.com/ljofreflor/jobbot"
CV_CREDIT = f"{MARK} — {REPO_URL}"
_LEGACY = "powered by AI jobbot de Leonardo Jofré"


def has_mark(text: str | None) -> bool:
    if not text:
        return False
    folded = text.casefold()
    return MARK.casefold() in folded or _LEGACY.casefold() in folded


def strip_mark(text: str | None) -> str:
    """Drop mark-only paragraphs so refine and diff see the candidate's words."""
    if not text:
        return ""
    kept: list[str] = []
    for part in text.split("\n\n"):
        line = part.strip()
        if not line:
            continue
        if line.casefold() in {MARK.casefold(), _LEGACY.casefold()}:
            continue
        kept.append(line)
    return "\n\n".join(kept).strip()


def stamp_description(text: str | None, *, max_len: int | None = None) -> str:
    """Append the mark once. A blank description stays blank."""
    body = strip_mark(text)
    if not body:
        return ""
    suffix = f"\n\n{MARK}"
    if max_len is not None and len(body) + len(suffix) > max_len:
        budget = max_len - len(suffix)
        if budget < 1:
            return ""
        body = _trim(body, budget)
        if not body:
            return ""
    return body + suffix


def _trim(text: str, maximum: int) -> str:
    text = text.strip()
    if len(text) <= maximum:
        return text
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    kept: list[str] = []
    for part in parts:
        trial = "\n\n".join([*kept, part])
        if len(trial) <= maximum:
            kept.append(part)
        else:
            break
    if kept:
        return "\n\n".join(kept)
    cut = text[:maximum].rsplit(" ", 1)[0]
    return cut.strip()
