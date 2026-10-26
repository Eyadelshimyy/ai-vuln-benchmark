"""Turn a public page about hiring into practices the CV advisor can use.

What is extracted is a practice: something to do to a CV. Who said it is dropped on
purpose — a recruiter's name, employer or contact is a person's data, and the advice
does not need it to be true. The extraction is structural (headings, list items,
sentences that state a rule), so it works the same on a page about nursing CVs and
one about engineering CVs.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from jobbot.ops.pii_guard import redact


class PracticeKind(StrEnum):
    """Which axis of the CV advisor this practice speaks to."""

    MACHINE = "machine"
    LANGUAGE = "language"
    LAYOUT = "layout"
    PROCESS = "process"


class Practice(BaseModel):
    """One thing to do to a CV, in one sentence."""

    kind: PracticeKind = PracticeKind.PROCESS
    text: str = Field(min_length=1, max_length=280)


PRACTICE_MAX = 280

# What each axis sounds like, in words about documents rather than about a trade.
_KIND_HINTS: tuple[tuple[PracticeKind, tuple[str, ...]], ...] = (
    (
        PracticeKind.LAYOUT,
        (
            "page",
            "página",
            "pagina",
            "first",
            "primera",
            "top",
            "length",
            "largo",
            "second",
            "segundo",
            "skim",
            "bullet",
            "viñeta",
            "vineta",
            "space",
            "espacio",
            "column",
            "columna",
        ),
    ),
    (
        PracticeKind.MACHINE,
        (
            "ats",
            "parse",
            "parser",
            "keyword",
            "palabra clave",
            "acronym",
            "acrónimo",
            "acronimo",
            "format",
            "formato",
            "pdf",
            "date",
            "fecha",
            "header",
            "encabezado",
            "table",
            "tabla",
        ),
    ),
    (
        PracticeKind.LANGUAGE,
        (
            "verb",
            "verbo",
            "quantify",
            "cuantifica",
            "number",
            "número",
            "numero",
            "wording",
            "redacción",
            "redaccion",
            "adjective",
            "adjetivo",
            "jargon",
            "jerga",
            "title",
            "título",
            "titulo",
            "write",
            "escribe",
        ),
    ),
)

# A sentence is a practice when it tells you to do something to the document.
_IMPERATIVE_HINTS: tuple[str, ...] = (
    "put",
    "keep",
    "write",
    "use",
    "avoid",
    "quantify",
    "start",
    "show",
    "include",
    "remove",
    "spend",
    "read",
    "look",
    "pon",
    "mantén",
    "manten",
    "escribe",
    "usa",
    "evita",
    "cuantifica",
    "empieza",
    "muestra",
    "incluye",
    "quita",
    "deja",
)

_CV_WORDS: tuple[str, ...] = ("cv", "resume", "resumé", "currículum", "curriculum", "perfil")

# Lines that are page furniture rather than advice.
_NOISE_RE = re.compile(
    r"(cookie|suscríbete|subscribe|newsletter|iniciar sesión|sign in|©|todos los derechos)",
    re.I,
)
# Bylines and contact lines: whoever wrote it is not part of the practice.
_BYLINE_RE = re.compile(
    r"^\s*(written by|por|escrito por|autor|author|by)\b|@|\bcontact\b|\bcontacto\b",
    re.I,
)


def extract_practices(html: str, *, limit: int = 12) -> list[Practice]:
    """The rules a public page states, one sentence each, without its author."""
    soup = BeautifulSoup(html or "", "html.parser")
    for node in soup.find_all(["script", "style", "nav", "footer", "form"]):
        node.decompose()
    chunks: list[str] = []
    for node in soup.find_all(["li", "p", "h2", "h3"]):
        text = node.get_text(" ", strip=True)
        if text:
            chunks.extend(_sentences(text))
    out: list[Practice] = []
    seen: set[str] = set()
    for chunk in chunks:
        practice = _practice_from(chunk)
        if practice is None:
            continue
        key = practice.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(practice)
        if len(out) >= limit:
            break
    return out


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _practice_from(sentence: str) -> Practice | None:
    clean = " ".join(sentence.split())
    if len(clean) < 20 or len(clean) > PRACTICE_MAX:
        return None
    if _NOISE_RE.search(clean) or _BYLINE_RE.search(clean):
        return None
    folded = clean.casefold()
    actionable = any(hint in folded for hint in _IMPERATIVE_HINTS)
    about_a_cv = any(word in folded for word in _CV_WORDS)
    if not actionable and not about_a_cv:
        return None
    return Practice(kind=_kind_of(folded), text=redact(clean))


def _kind_of(folded: str) -> PracticeKind:
    for kind, hints in _KIND_HINTS:
        if any(hint in folded for hint in hints):
            return kind
    return PracticeKind.PROCESS


def advisor_notes(root: Path, *, limit: int = 6) -> list[str]:
    """Practices from sources you promoted. Candidates advise nobody."""
    from jobbot.recruiters.sources import active_sources, default_recruiters_path, load_sources

    sources = active_sources(load_sources(default_recruiters_path(root)))
    notes: list[str] = []
    for source in sources:
        for practice in source.practices:
            if practice.text not in notes:
                notes.append(practice.text)
            if len(notes) >= limit:
                return notes
    return notes
