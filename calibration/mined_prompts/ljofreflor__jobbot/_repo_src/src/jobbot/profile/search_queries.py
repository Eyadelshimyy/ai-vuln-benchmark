"""Job-board searches derived from what the candidate has done, not from a job title.

A headline can be a degree, and a literal default query assumes one career for
everybody. What the experience backs is the honest search: the competencies the
roles mention, the phrases the achievements keep repeating, and the titles held.
Every query cites the experiences behind it; nothing is invented or translated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from jobbot.jobs.normalization import WordIndex, fold_text, stem_word
from jobbot.models.candidate import Candidate
from jobbot.models.experience import Experience

DEFAULT_QUERY_LIMIT = 8

_MIN_WORD = 4
_RECURRING_MIN_EXPERIENCES = 2
_RECURRING_UNSUPPORTED = 3
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")
_WORD_RE = re.compile(r"[^\W\d_]+")

_FUNCTION_WORDS = frozenset(
    {
        "para",
        "como",
        "sobre",
        "entre",
        "desde",
        "hasta",
        "mediante",
        "durante",
        "segun",
        "unos",
        "unas",
        "este",
        "esta",
        "estos",
        "estas",
        "with",
        "from",
        "into",
        "that",
        "this",
        "their",
        "through",
    }
)
# Connectors that keep a phrase whole: 'investigación de brotes', 'control of vectors'.
_CONNECTORS = frozenset({"de", "del", "en", "of", "in", "for"})
# Work vocabulary every field shares: never a search on its own.
_GENERIC_WORDS = frozenset(
    {
        "apoyo",
        "apoyar",
        "gestion",
        "coordinacion",
        "implementacion",
        "participacion",
        "desarrollo",
        "responsable",
        "trabajo",
        "tecnico",
        "tecnica",
        "profesional",
        "nacional",
        "regional",
        "general",
        "area",
        "equipo",
        "equipos",
        "realizar",
        "support",
        "management",
        "team",
        "work",
    }
)
_GENERIC_STEMS = frozenset(stem_word(word) for word in _GENERIC_WORDS)


class QueryOrigin(StrEnum):
    COMPETENCY = "competency"
    RECURRING = "achievements"
    TITLE = "title"


@dataclass(frozen=True)
class DerivedQuery:
    text: str
    origin: QueryOrigin
    evidence: tuple[str, ...]  # ids of the experiences that back it


@dataclass(frozen=True)
class ChosenQueries:
    queries: list[str]
    saved: bool


def search_queries_for(candidate: Candidate, *, limit: int = DEFAULT_QUERY_LIMIT) -> ChosenQueries:
    """The queries a search runs with: the saved list if there is one, else the derived set."""
    if candidate.search_queries:
        return ChosenQueries(queries=list(candidate.search_queries), saved=True)
    derived = derive_search_queries(candidate, limit=limit)
    return ChosenQueries(queries=[query.text for query in derived], saved=False)


def derive_search_queries(
    candidate: Candidate, *, limit: int = DEFAULT_QUERY_LIMIT
) -> list[DerivedQuery]:
    """Competencies, recurring achievement phrases and held titles, taken in turns."""
    experiences = sorted(candidate.experience, key=_recency, reverse=True)
    if not experiences or limit < 1:
        return []
    docs = {exp.id: WordIndex(frozenset(_content(_experience_text(exp)))) for exp in experiences}

    competencies = _competencies(candidate, docs)
    declared = {_key(query.text) for query in competencies}
    summary = WordIndex(frozenset(_content(candidate.summary or "")))
    recurring = [
        query
        for query in _recurring_phrases(experiences, summary)
        if _key(query.text) not in declared
    ]
    pools = [competencies, recurring, _held_titles(experiences)]
    chosen: list[DerivedQuery] = []
    keys: set[tuple[str, ...]] = set()
    while len(chosen) < limit and any(pools):
        for pool in pools:
            while pool:
                query = pool.pop(0)
                key = _key(query.text)
                if key and key not in keys:
                    keys.add(key)
                    chosen.append(query)
                    break
            if len(chosen) >= limit:
                break
    return chosen


def _competencies(candidate: Candidate, docs: dict[str, WordIndex]) -> list[DerivedQuery]:
    """Declared skills that some role actually mentions; a bare listing is not enough."""
    found: list[DerivedQuery] = []
    for claim in (*candidate.skills.all_skills(), *candidate.specialties):
        text = _PARENTHETICAL_RE.sub("", claim).strip()
        words = _content(text)
        if not words or _is_generic(words):
            continue
        backing = tuple(
            exp_id for exp_id, doc in docs.items() if all(doc.has(word) for word in words)
        )
        if backing:
            found.append(DerivedQuery(text=text, origin=QueryOrigin.COMPETENCY, evidence=backing))
    found.sort(key=lambda query: len(query.evidence), reverse=True)
    return found


def _recurring_phrases(experiences: list[Experience], summary: WordIndex) -> list[DerivedQuery]:
    """Phrases the achievements of several roles repeat.

    Two roles can share a phrase by accident ('red pública, privada'); the summary
    is the candidate's own account of what matters, so two roles are enough only
    when the summary says it too, and otherwise it takes three.
    """
    surfaces: dict[tuple[str, ...], str] = {}
    backing: dict[tuple[str, ...], list[str]] = {}
    for exp in experiences:
        seen_here: set[tuple[str, ...]] = set()
        for surface in _phrases(_prose(exp)):
            key = _key(surface)
            if not key or key in seen_here:
                continue
            seen_here.add(key)
            surfaces.setdefault(key, surface)
            backing.setdefault(key, []).append(exp.id)
    found: list[DerivedQuery] = []
    for key, ids in backing.items():
        words = _content(surfaces[key])
        if _is_generic(words):
            continue
        in_summary = all(summary.has(word) for word in words)
        needed = _RECURRING_MIN_EXPERIENCES if in_summary else _RECURRING_UNSUPPORTED
        if len(ids) >= needed:
            found.append(
                DerivedQuery(text=surfaces[key], origin=QueryOrigin.RECURRING, evidence=tuple(ids))
            )
    found.sort(key=lambda query: len(query.evidence), reverse=True)
    return found


def _held_titles(experiences: list[Experience]) -> list[DerivedQuery]:
    """Titles held, most recent first: an old title held often is not today's search.

    'Médico' and 'Médica' are one title, cited with every role that held it.
    """
    surfaces: dict[tuple[str, ...], str] = {}
    backing: dict[tuple[str, ...], list[str]] = {}
    for exp in experiences:
        words = _content(exp.title)
        if not words or _is_generic(words):
            continue
        key = _key(exp.title)
        surfaces.setdefault(key, exp.title.strip())
        backing.setdefault(key, []).append(exp.id)
    return [
        DerivedQuery(text=surfaces[key], origin=QueryOrigin.TITLE, evidence=tuple(ids))
        for key, ids in backing.items()
    ]


def _phrases(text: str) -> list[str]:
    """'vigilancia epidemiológica', 'investigación de brotes': content, connector, content."""
    tokens = [match.group(0) for match in _WORD_RE.finditer(text)]
    folded = [fold_text(token) for token in tokens]
    out: list[str] = []
    for i, word in enumerate(folded):
        if not _is_content(word):
            continue
        if i + 1 < len(folded) and _is_content(folded[i + 1]):
            out.append(f"{tokens[i]} {tokens[i + 1]}".lower())
        elif (
            i + 2 < len(folded)
            and folded[i + 1] in _CONNECTORS
            and _is_content(folded[i + 2])
        ):
            out.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}".lower())
    return out


def _experience_text(exp: Experience) -> str:
    return f"{exp.title}\n{_prose(exp)}"


def _prose(exp: Experience) -> str:
    return "\n".join([exp.description or "", *(ach.text for ach in exp.achievements)])


def _content(text: str) -> list[str]:
    return [word for word in fold_text(text).split() if _is_content(word)]


def _is_content(word: str) -> bool:
    return len(word) >= _MIN_WORD and word not in _FUNCTION_WORDS and not word.isdigit()


def _is_generic(words: list[str]) -> bool:
    return all(stem_word(word) in _GENERIC_STEMS for word in words)


def _key(text: str) -> tuple[str, ...]:
    return tuple(sorted(stem_word(word) for word in _content(text)))


def _recency(exp: Experience) -> tuple[bool, str, str]:
    return (exp.current, exp.end_date or exp.start_date, exp.start_date)
