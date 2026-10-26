"""Detect a posting that says the vacancy is already filled. Evidence, or nothing."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from collections.abc import Callable

from bs4 import BeautifulSoup, Tag

from jobbot.models.job import JobPosting

# Phrases that state this opening is gone. A bare "closed" or "filled" is not
# enough: those words show up in ordinary descriptions.
_CLOSED_PHRASES: tuple[str, ...] = (
    "this position has been filled",
    "this job has been filled",
    "this role has been filled",
    "position has been filled",
    "the role has been filled",
    "the job you are trying to apply for has been filled",
    "vacancy has been filled",
    "no longer accepting applications",
    "this job is no longer available",
    "this position is no longer available",
    "this posting is no longer available",
    "posting has expired",
    "this job has expired",
    "cargo ya fue cubierto",
    "cargo ya está cubierto",
    "cargo ya esta cubierto",
    "el cargo ya fue cubierto",
    "el cargo ya está cubierto",
    "el cargo ya esta cubierto",
    "vacante cubierta",
    "vacante ya fue cubierta",
    "vacante ya ha sido cubierta",
    "esta vacante ha sido cubierta",
    "esta vacante ya fue cubierta",
    "posición cubierta",
    "posicion cubierta",
    "ya no acepta postulaciones",
    "no se reciben más postulantes",
    "no se reciben mas postulantes",
    "la oferta ya no está disponible",
    "la oferta ya no esta disponible",
    "esta oferta ha expirado",
    "esta vacante ya no está disponible",
    "esta vacante ya no esta disponible",
)

_WS = re.compile(r"\s+")
_LOOKS_LIKE_HTML = re.compile(r"<[a-zA-Z][^>]*>")
_DISPLAY_NONE = re.compile(r"display\s*:\s*none", re.I)
# Class token `hide` (Phenom), not attribute names like hide-sub-title=.
_HIDE_CLASS = re.compile(r"(?:^|\s)hide(?:\s|$)")


def visible_soup(html: str) -> BeautifulSoup:
    """Parse HTML and drop nodes the page itself marks as hidden.

    Career platforms keep an expired-job template in the DOM (`class="hide"`).
    Plain text is returned as a soup with a single string — callers that only
    need phrases should use ``closure_evidence``.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    if not html or not _LOOKS_LIKE_HTML.search(html):
        return soup
    hidden = [node for node in soup.find_all(True) if isinstance(node, Tag) and _is_hidden(node)]
    for node in hidden:
        node.decompose()
    return soup


def closure_evidence(text: str) -> str | None:
    """Return the phrase that says this vacancy is filled, or None if we cannot tell.

    HTML is flattened first so a banner split by tags still counts. Nodes that
    are hidden (class ``hide``, ``hidden``, ``aria-hidden``, ``display:none``)
    are dropped first: career platforms keep an "expired" template in the DOM
    and only un-hide it when the vacancy is gone. No phrase means unknown —
    not that the posting is open.
    """
    if not text:
        return None
    visible = _visible_text(text)
    plain = _WS.sub(" ", visible).casefold()
    found = [phrase for phrase in _CLOSED_PHRASES if phrase.casefold() in plain]
    if not found:
        return None
    return max(found, key=len)


def _visible_text(text: str) -> str:
    if not text or not _LOOKS_LIKE_HTML.search(text):
        return text
    return visible_soup(text).get_text(" ", strip=True)


def _is_hidden(node: Tag) -> bool:
    attrs = node.attrs or {}
    if "hidden" in attrs:
        return True
    if str(attrs.get("aria-hidden", "")).casefold() == "true":
        return True
    style = str(attrs.get("style") or "")
    if _DISPLAY_NONE.search(style):
        return True
    classes = attrs.get("class")
    if isinstance(classes, list):
        return "hide" in classes
    if isinstance(classes, str):
        return bool(_HIDE_CLASS.search(classes))
    return False


def fetch_posting_text(url: str, *, timeout: float = 20.0) -> str:
    """Read a public posting page. Raises OSError when it cannot be read."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "jobbot/0.1 (local; posting status)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read().decode(charset, errors="replace")
            return str(body)
    except urllib.error.HTTPError as exc:
        msg = f"posting page returned HTTP {exc.code}"
        raise OSError(msg) from exc


def closure_evidence_for_job(
    job: JobPosting,
    *,
    fetch: Callable[[str], str] | None = None,
) -> str | None:
    """Evidence from the stored text, then from the live page when `fetch` is given.

    A fetch that fails is unknown, not "still open" and not "filled".
    """
    for blob in (job.description, job.raw_description):
        found = closure_evidence(blob)
        if found:
            return found
    if fetch is None:
        return None
    url = job.ats_url or job.url
    if not url or url.lower().startswith("mailto:"):
        return None
    try:
        page = fetch(url)
    except OSError:
        return None
    return closure_evidence(page)
