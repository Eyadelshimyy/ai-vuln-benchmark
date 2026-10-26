"""Rule-based job matching — never invents candidate skills."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from jobbot.jobs.normalization import WordIndex, fold_text, normalize_skill, skills_in_text
from jobbot.jobs.parsing import extract_skills_from_text, looks_like_page_metadata
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch, MatchItem, MatchStrength

# A job whose posting names almost nothing cannot be a perfect fit, whatever it asks.
_THIN_EVIDENCE_ITEMS = 3
_THIN_EVIDENCE_CAP = 80.0

# Requirement sentences are prose: only content words carry evidence.
_STOPWORDS = frozenset(
    {
        "años",
        "anos",
        "para",
        "con",
        "como",
        "experiencia",
        "experience",
        "conocimiento",
        "conocimientos",
        "manejo",
        "titulo",
        "deseable",
        "excluyente",
        "trabajo",
        "equipo",
        "equipos",
        "afin",
        "otros",
        "otras",
        "sobre",
        "strong",
        "comfortable",
        "preferred",
        "nice",
        "have",
        "with",
        "and",
        "the",
        "role",
        "senior",
        "junior",
        "semi",
    }
)

_MAX_REQUIREMENT_CHARS = 140

# An achievement echoes a posting when most of its own content words are there.
_ECHO_MIN_WORD = 5
_ECHO_MIN_HITS = 4
_ECHO_SHARE = 0.5
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")


@dataclass(frozen=True)
class _Vocabulary:
    """What the profile actually claims: skill phrases plus their content words."""

    phrases: dict[str, str]  # folded phrase → original wording
    words: frozenset[str]
    index: WordIndex


class JobAnalyzer(Protocol):
    def analyze(self, candidate: Candidate, job: JobPosting) -> JobMatch: ...


class RuleBasedJobAnalyzer:
    """Deterministic matcher using skills, tags, keywords, seniority, role family."""

    def analyze(self, candidate: Candidate, job: JobPosting) -> JobMatch:
        candidate_tokens = _candidate_tokens(candidate)
        vocabulary = _candidate_vocabulary(candidate)
        items: list[MatchItem] = []

        required = _job_requirements(job)
        for token, label, text in required:
            evidence = _requirement_evidence(text, vocabulary)
            if token in candidate_tokens:
                items.append(
                    MatchItem(
                        label=label,
                        strength=MatchStrength.STRONG,
                        detail="present in profile",
                    )
                )
            elif evidence is not None:
                strength, detail = evidence
                items.append(MatchItem(label=label, strength=strength, detail=detail))
            elif _partial_token(token, candidate_tokens):
                items.append(
                    MatchItem(
                        label=label,
                        strength=MatchStrength.PARTIAL,
                        detail="related skill/tag in profile",
                    )
                )
            else:
                items.append(
                    MatchItem(
                        label=label,
                        strength=MatchStrength.MISSING,
                        detail="not found in profile",
                    )
                )

        items.extend(_profile_evidence(candidate, job, items))

        role_item, role_cap = _role_family_item(candidate, job)
        items.append(role_item)

        for lang in job.language_requirements:
            # Languages are often not structured in profile → unknown, never missing
            items.append(
                MatchItem(
                    label=f"{lang} proficiency",
                    strength=MatchStrength.UNKNOWN,
                    detail="not modeled in profile.yaml",
                )
            )

        if job.seniority:
            items.append(_seniority_item(candidate, job.seniority))

        score = _score(items, role_cap=role_cap, required_tokens={t for t, _, _ in required})
        return JobMatch(job_id=job.id, score=score, items=items)


def requirement_evidence(
    candidate: Candidate, requirement: str
) -> tuple[MatchStrength, str] | None:
    """The profile's own wording behind one requirement sentence, or None."""
    return _requirement_evidence(requirement, _candidate_vocabulary(candidate))


def _candidate_tokens(candidate: Candidate) -> set[str]:
    tokens: set[str] = set()
    for skill in candidate.skills.all_skills():
        tokens.add(normalize_skill(skill))
    for specialty in candidate.specialties:
        tokens.add(normalize_skill(specialty))
    for exp in candidate.experience:
        tokens.add(normalize_skill(exp.title))
        tokens.add(normalize_skill(exp.company))
        if exp.description:
            tokens |= _keyword_candidates(exp.description)
            tokens |= skills_in_text(exp.description)
        for ach in exp.achievements:
            for tag in ach.tags:
                tokens.add(normalize_skill(tag))
            tokens |= _keyword_candidates(ach.text)
            tokens |= skills_in_text(ach.text)
    for edu in candidate.education:
        tokens |= skills_in_text(edu.degree)
    if candidate.summary:
        tokens |= _keyword_candidates(candidate.summary)
        tokens |= skills_in_text(candidate.summary)
    return {t for t in tokens if t}


def _job_requirements(job: JobPosting) -> list[tuple[str, str, str]]:
    """(token, label, full text) per requirement; long Spanish lines stay in."""
    pairs: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for skill in job.skills:
        token = normalize_skill(skill)
        if token and token not in seen and not looks_like_page_metadata(skill):
            seen.add(token)
            pairs.append((token, skill, skill))
    for req in job.requirements:
        if re.search(r"(?i)\b(english|spanish|idioma|language|proficiency)\b", req):
            continue
        if looks_like_page_metadata(req):
            continue
        token = normalize_skill(req)
        if token and token not in seen and len(req) <= _MAX_REQUIREMENT_CHARS:
            seen.add(token)
            pairs.append((token, req[:80], req))
    blob = f"{job.title}\n{job.description}\n{job.raw_description}"
    if len(pairs) < 3:
        for hint in extract_skills_from_text(blob):
            token = normalize_skill(hint)
            if token and token not in seen:
                seen.add(token)
                pairs.append((token, hint, hint))
    return pairs


def _candidate_vocabulary(candidate: Candidate) -> _Vocabulary:
    """Skills, specialties, titles and degrees: what the profile literally claims."""
    phrases: dict[str, str] = {}
    for claim in (
        *candidate.skills.all_skills(),
        *candidate.specialties,
        *(exp.title for exp in candidate.experience),
        *(edu.degree for edu in candidate.education),
    ):
        folded = fold_text(claim)
        if len(folded) >= 3:
            phrases.setdefault(folded, claim)

    words = {
        word
        for folded in phrases
        for word in folded.split()
        if len(word) >= 4 and word not in _STOPWORDS
    }
    return _Vocabulary(phrases=phrases, words=frozenset(words), index=WordIndex(frozenset(words)))


def _requirement_evidence(
    requirement: str,
    vocabulary: _Vocabulary,
) -> tuple[MatchStrength, str] | None:
    """Match a requirement sentence against the profile's own wording."""
    folded = fold_text(requirement)
    if not folded:
        return None

    for phrase, original in vocabulary.phrases.items():
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", folded):
            return MatchStrength.STRONG, f"profile claims {original}"

    content = [word for word in folded.split() if len(word) >= 4 and word not in _STOPWORDS]
    hits = sorted({word for word in content if word in vocabulary.words})
    variants = sorted(
        {word for word in content if word not in hits and vocabulary.index.has(word)}
    )

    if len(hits) >= 2:
        return MatchStrength.STRONG, f"profile wording: {', '.join(hits)}"
    if hits and variants:
        return MatchStrength.STRONG, f"profile wording: {', '.join([*hits, *variants])}"
    if hits:
        return MatchStrength.PARTIAL, f"profile wording: {hits[0]}"
    if variants:
        return MatchStrength.PARTIAL, f"profile wording (variant): {variants[0]}"
    return None


def _profile_evidence(
    candidate: Candidate,
    job: JobPosting,
    matched: list[MatchItem],
) -> list[MatchItem]:
    """What the profile claims and the posting talks about, read from the profile's side.

    Requirements are read from the posting, and a career page full of chrome yields
    few real ones. The candidate's own skills, specialties, degrees and achievements
    found in the posting's text are evidence of fit in any field. Only what is found
    is added: a claim the posting does not mention is not a gap, so it costs nothing.
    """
    posting = WordIndex(
        frozenset(_keyword_candidates(f"{job.title}\n{job.description}\n{job.raw_description}"))
    )
    if not posting.words:
        return []
    already = [
        fold_text(item.label)
        for item in matched
        if item.strength in {MatchStrength.STRONG, MatchStrength.PARTIAL}
    ]
    items: list[MatchItem] = []
    seen: set[str] = set()
    for claim in (
        *candidate.skills.all_skills(),
        *candidate.specialties,
        *(edu.degree for edu in candidate.education),
    ):
        bare = _PARENTHETICAL_RE.sub("", claim).strip()
        folded = fold_text(bare)
        words = _keyword_candidates(bare)
        if not words or folded in seen or any(folded in done or done in folded for done in already):
            continue
        seen.add(folded)
        if all(posting.has(word) for word in words):
            items.append(
                MatchItem(
                    label=f"profile:{bare}",
                    strength=MatchStrength.STRONG,
                    detail="the posting mentions it",
                )
            )
    for exp in candidate.experience:
        for ach in exp.achievements:
            words = {word for word in _keyword_candidates(ach.text) if len(word) >= _ECHO_MIN_WORD}
            hits = sorted(word for word in words if posting.has(word))
            if len(hits) >= _ECHO_MIN_HITS and len(hits) >= _ECHO_SHARE * len(words):
                items.append(
                    MatchItem(
                        label=f"achievement:{ach.id}",
                        strength=MatchStrength.STRONG,
                        detail=f"the posting echoes: {', '.join(hits[:5])}",
                    )
                )
    return items


def _keyword_candidates(text: str) -> set[str]:
    """Content words of the candidate's own prose — no vocabulary to belong to."""
    return {
        word
        for word in fold_text(text).split()
        if len(word) >= 4 and word not in _STOPWORDS
    }


def _partial_token(token: str, candidate_tokens: set[str]) -> bool:
    """A requirement is partially met when the profile names part of that phrase."""
    words = [word for word in token.split("_") if len(word) >= 4 and word not in _STOPWORDS]
    if not words:
        return False
    return any(word in candidate_tokens for word in words)


def _seniority_item(candidate: Candidate, required: str) -> MatchItem:
    titles = " ".join(e.title.lower() for e in candidate.experience)
    req = required.lower()
    if req in titles or (req == "senior" and "senior" in titles):
        return MatchItem(
            label=f"seniority:{required}",
            strength=MatchStrength.STRONG,
            detail="title history includes seniority signal",
        )
    if req in {"senior", "staff", "lead"} and any(
        x in titles for x in ("senior", "lead", "staff", "principal")
    ):
        return MatchItem(
            label=f"seniority:{required}",
            strength=MatchStrength.PARTIAL,
            detail="adjacent seniority in profile",
        )
    return MatchItem(
        label=f"seniority:{required}",
        strength=MatchStrength.UNKNOWN,
        detail="cannot verify years/level precisely from profile",
    )


def _role_family_item(candidate: Candidate, job: JobPosting) -> tuple[MatchItem, float | None]:
    """Compare the job title with the titles the candidate has actually held.

    A table of role families only knows the families someone wrote down, and a
    candidate whose field is missing from it gets judged by a family they never
    claimed. The titles in the profile are the only evidence there is.
    """
    label = f"role:{job.title}"
    held = _title_words(
        *(exp.title for exp in candidate.experience),
        candidate.personal.headline or "",
    )
    wanted = _title_words(job.title)
    held_index = WordIndex(frozenset(held))
    shared = sorted(word for word in wanted if held_index.has(word))

    if len(shared) >= 2:
        return (
            MatchItem(
                label=label,
                strength=MatchStrength.STRONG,
                detail=f"held titles share: {', '.join(shared)}",
            ),
            None,
        )
    if len(shared) == 1:
        return (
            MatchItem(
                label=label,
                strength=MatchStrength.PARTIAL,
                detail=f"held titles share: {shared[0]}",
            ),
            70.0,
        )
    if not wanted:
        return (
            MatchItem(
                label=label,
                strength=MatchStrength.UNKNOWN,
                detail="job title carries no comparable words",
            ),
            None,
        )
    return (
        MatchItem(
            label=label,
            strength=MatchStrength.MISSING,
            detail="job title does not match any title in the profile",
        ),
        40.0,
    )


def _title_words(*titles: str) -> set[str]:
    return {
        word
        for title in titles
        for word in fold_text(title).split()
        if len(word) >= 4 and word not in _STOPWORDS
    }


def _score(
    items: list[MatchItem],
    *,
    role_cap: float | None,
    required_tokens: set[str],
) -> float:
    """Score only on strong/partial/missing skill-like items; unknowns excluded."""
    relevant = [i for i in items if i.strength != MatchStrength.UNKNOWN]
    if not relevant:
        return 0.0
    points = 0.0
    for item in relevant:
        if item.strength == MatchStrength.STRONG:
            points += 1.0
        elif item.strength == MatchStrength.PARTIAL:
            points += 0.5
        # missing contributes 0
    score = 100.0 * points / len(relevant)
    # A posting that names almost nothing is thin evidence, not a perfect fit.
    if len(required_tokens - {""}) < _THIN_EVIDENCE_ITEMS:
        score = min(score, _THIN_EVIDENCE_CAP)
    if role_cap is not None:
        score = min(score, role_cap)
    return round(score, 1)
