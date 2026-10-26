"""Opt-in last page of a job CV: which posting it was built for, the match, and where
each strong requirement is evidenced. Gaps are never listed; the ATS text never carries it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from jobbot.companies.urls import PrivateRouteRejected, public_url
from jobbot.cv.selection import SelectionResult, filter_experiences
from jobbot.jobs.indeed_url import IndeedUrlError, canonical_indeed_job_url
from jobbot.jobs.language import Language, posting_language
from jobbot.jobs.normalization import fold_text, normalize_skill, skills_in_text
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch, MatchStrength
from jobbot.portals.detect import AtsKind, detect_ats

MAX_EVIDENCE_LINES = 8

# Matcher items that describe the posting as a whole rather than one requirement.
_NON_REQUIREMENT_PREFIXES = ("role:", "seniority:")

EvidenceKind = Literal["skills", "experience", "project", "education", "summary"]


@dataclass(frozen=True)
class LogText:
    title: str
    created: str
    for_posting: str
    generated: str
    match: str
    match_note: str
    evidence: str


# The log is read by the posting's recruiter, so it speaks the posting's language.
LOG_TEXT: dict[Language, LogText] = {
    "es": LogText(
        title="Registro de la postulación",
        created="Este CV fue creado con JobBot",
        for_posting="para esta oferta:",
        generated="Fecha de generación",
        match="Afinidad con la oferta",
        match_note="calculada por JobBot comparando los requisitos del aviso con este perfil",
        evidence="Requisitos del aviso y evidencia en este CV",
    ),
    "en": LogText(
        title="Application log",
        created="This CV was created with JobBot",
        for_posting="for this posting:",
        generated="Generated on",
        match="Match with the posting",
        match_note="computed by JobBot comparing the posting's requirements with this profile",
        evidence="Posting requirements and evidence in this CV",
    ),
}


@dataclass(frozen=True)
class ApplicationLogOptions:
    project_url: str | None = None
    generated_on: date | None = None


@dataclass(frozen=True)
class EvidenceLine:
    requirement: str
    kind: EvidenceKind
    pointer: str


@dataclass(frozen=True)
class ApplicationLog:
    job_title: str
    company: str
    job_url: str | None
    generated_on: date
    score: float
    project_url: str | None
    language: Language = "es"
    evidence: list[EvidenceLine] = field(default_factory=list)

    @property
    def text(self) -> LogText:
        return LOG_TEXT[self.language]

    @property
    def percent(self) -> int:
        """Whole percent, half up (95.5 → 96), as a reader expects."""
        return int(self.score + 0.5)


def posting_url(job: JobPosting) -> str | None:
    """The posting's public URL: tracking dropped, Indeed keeps its job key, never an email."""
    for raw in (job.url, job.ats_url):
        if not raw:
            continue
        if detect_ats(raw) == AtsKind.INDEED:
            try:
                return canonical_indeed_job_url(raw)
            except IndeedUrlError:
                pass
        try:
            return public_url(raw)
        except PrivateRouteRejected:
            continue
    return None


def build_application_log(
    candidate: Candidate,
    job: JobPosting,
    match: JobMatch,
    *,
    selection: SelectionResult,
    options: ApplicationLogOptions,
    fallback_language: Language = "es",
) -> ApplicationLog:
    """``fallback_language`` is the CV's own, used when the posting's text is unclear."""
    return ApplicationLog(
        job_title=job.title,
        company=job.company,
        job_url=posting_url(job),
        generated_on=options.generated_on or date.today(),
        score=match.score,
        project_url=options.project_url,
        language=posting_language(job) or fallback_language,
        evidence=strong_requirement_evidence(candidate, match, selection),
    )


def strong_requirement_evidence(
    candidate: Candidate,
    match: JobMatch,
    selection: SelectionResult,
    *,
    limit: int = MAX_EVIDENCE_LINES,
) -> list[EvidenceLine]:
    """Strong requirements only, each pointing at content this CV actually shows."""
    lines: list[EvidenceLine] = []
    seen: set[str] = set()
    for item in match.by_strength(MatchStrength.STRONG):
        if item.label.startswith(_NON_REQUIREMENT_PREFIXES) or item.label in seen:
            continue
        found = _locate(candidate, selection, item.label)
        if found is None:
            continue
        seen.add(item.label)
        lines.append(EvidenceLine(requirement=item.label, kind=found[0], pointer=found[1]))
        if len(lines) >= limit:
            break
    return lines


def _locate(
    candidate: Candidate,
    selection: SelectionResult,
    requirement: str,
) -> tuple[EvidenceKind, str] | None:
    token = normalize_skill(requirement)
    for group, items in candidate.skills.as_dict().items():
        if any(normalize_skill(skill) == token for skill in items):
            return "skills", group.replace("_", " ").title()
    for exp, achievements in filter_experiences(candidate, selection):
        texts = [exp.title, exp.description or ""]
        texts += [ach.text for ach in achievements]
        tags = [tag for ach in achievements for tag in ach.tags]
        if any(normalize_skill(tag) == token for tag in tags) or _mentions(texts, requirement):
            return "experience", f"{exp.title}, {exp.company}"
    for project in candidate.projects:
        if any(normalize_skill(tag) == token for tag in project.tags) or _mentions(
            [project.name, project.description], requirement
        ):
            return "project", project.name
    for edu in candidate.education:
        if _mentions([edu.degree, edu.details or ""], requirement):
            return "education", f"{edu.degree}, {edu.institution}"
    if candidate.summary and _mentions([candidate.summary], requirement):
        return "summary", ""
    return None


def _mentions(texts: list[str], requirement: str) -> bool:
    phrase = fold_text(requirement)
    token = normalize_skill(requirement)
    if not phrase:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
    for text in texts:
        if not text:
            continue
        if pattern.search(fold_text(text)) or (token and token in skills_in_text(text)):
            return True
    return False
