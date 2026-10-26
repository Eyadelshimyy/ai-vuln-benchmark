"""Achievement / content selection for CV builds."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from jobbot.jobs.normalization import fold_text, normalize_skill
from jobbot.jobs.parsing import extract_skills_from_text
from jobbot.models.candidate import Candidate
from jobbot.models.experience import Achievement, Experience
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch

# A terse LinkedIn / email-apply post is not enough signal to shrink the CV.
_SHORT_JD_CHARS = 400
_MIN_EXPERIENCES = 4
_MIN_ACHIEVEMENTS = 4
_FLOOR_REASON = "floor: short JD — kept recent evidence so the CV stays usable"


@dataclass
class SelectedAchievement:
    id: str
    reason: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    """Which profile facts to include in a derived CV."""

    experience_ids: list[str]
    selected_achievements: list[SelectedAchievement]
    skill_names: list[str]
    include_publications: bool = True
    job_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "experience_ids": self.experience_ids,
            "selected_achievements": [
                {"id": a.id, "reason": a.reason} for a in self.selected_achievements
            ],
            "skill_names": self.skill_names,
            "include_publications": self.include_publications,
        }
        if self.job_id:
            payload["job_id"] = self.job_id
        return payload


def select_for_base_cv(candidate: Candidate) -> SelectionResult:
    """Select all profile content for the base (non-job-specific) CV."""
    achievements: list[SelectedAchievement] = []
    for exp in candidate.experience:
        for ach in exp.achievements:
            achievements.append(SelectedAchievement(id=ach.id, reason=["included in base CV"]))
    return SelectionResult(
        experience_ids=[e.id for e in candidate.experience],
        selected_achievements=achievements,
        skill_names=candidate.skills.all_skills(),
        include_publications=True,
    )


def select_for_job(
    candidate: Candidate,
    job: JobPosting,
    match: JobMatch | None = None,
) -> SelectionResult:
    """Select achievements/skills overlapping the job — never invent."""
    job_tokens = _job_tokens(job)
    selected: list[SelectedAchievement] = []
    exp_ids: list[str] = []

    for exp in candidate.experience:
        exp_selected = False
        for ach in exp.achievements:
            reasons = _achievement_reasons(ach, exp, job_tokens, job)
            if reasons:
                selected.append(SelectedAchievement(id=ach.id, reason=reasons))
                exp_selected = True
        if exp_selected or _experience_relevant(exp, job_tokens):
            exp_ids.append(exp.id)

    if not selected:
        # Fallback: keep top experiences but mark reasons honestly
        for exp in candidate.experience[:3]:
            exp_ids.append(exp.id)
            for ach in exp.achievements[:2]:
                selected.append(
                    SelectedAchievement(
                        id=ach.id,
                        reason=["fallback: no strong keyword overlap; kept recent evidence"],
                    )
                )

    jd_blob = " ".join(
        part
        for part in (job.description, job.title, "\n".join(job.requirements))
        if part
    )
    if len(jd_blob) < _SHORT_JD_CHARS or len({e for e in exp_ids}) < _MIN_EXPERIENCES:
        _pad_selection_floor(candidate, exp_ids, selected)

    # Skills: intersection only
    skill_names = [
        s for s in candidate.skills.all_skills() if normalize_skill(s) in job_tokens
    ]
    if not skill_names:
        skill_names = candidate.skills.all_skills()[:8]

    # Dedupe experience ids preserving order
    seen: set[str] = set()
    ordered_exp: list[str] = []
    for eid in exp_ids:
        if eid not in seen:
            seen.add(eid)
            ordered_exp.append(eid)

    _ = match  # reserved for future weighting
    return SelectionResult(
        experience_ids=ordered_exp,
        selected_achievements=selected,
        skill_names=skill_names,
        include_publications=True,
        job_id=job.id,
    )


def _pad_selection_floor(
    candidate: Candidate,
    exp_ids: list[str],
    selected: list[SelectedAchievement],
) -> None:
    """Keep enough recent evidence when the posting is too short to select by keywords."""
    selected_ids = {item.id for item in selected}
    for exp in candidate.experience:
        if len({eid for eid in exp_ids}) >= _MIN_EXPERIENCES and len(selected) >= _MIN_ACHIEVEMENTS:
            return
        if exp.id not in exp_ids:
            exp_ids.append(exp.id)
        for ach in exp.achievements:
            if ach.id in selected_ids:
                continue
            selected.append(SelectedAchievement(id=ach.id, reason=[_FLOOR_REASON]))
            selected_ids.add(ach.id)
            if (
                len(selected) >= _MIN_ACHIEVEMENTS
                and len({eid for eid in exp_ids}) >= _MIN_EXPERIENCES
            ):
                return


def write_selection_json(selection: SelectionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def filter_experiences(
    candidate: Candidate,
    selection: SelectionResult,
) -> list[tuple[Experience, list[Achievement]]]:
    """Return experiences with only selected achievements, preserving order."""
    selected_ids = {a.id for a in selection.selected_achievements}
    allowed_exp = set(selection.experience_ids)
    result: list[tuple[Experience, list[Achievement]]] = []
    for exp in candidate.experience:
        if exp.id not in allowed_exp:
            continue
        achs = [a for a in exp.achievements if a.id in selected_ids]
        result.append((exp, achs))
    return result


def _job_tokens(job: JobPosting) -> set[str]:
    """What the job asks for, in its own words: skills it names plus its title."""
    named = [*job.skills, *extract_skills_from_text(job.title)]
    if len(named) < 3:
        # Short or prose-only postings: read the requirement lines as well.
        named += extract_skills_from_text("\n".join(job.requirements))
    tokens = {normalize_skill(skill) for skill in named}
    return {token for token in tokens if token}


def _achievement_reasons(
    ach: Achievement,
    exp: Experience,
    job_tokens: set[str],
    job: JobPosting,
) -> list[str]:
    reasons: list[str] = []
    for tag in ach.tags:
        token = normalize_skill(tag)
        if token in job_tokens:
            reasons.append(f"tag overlap: {tag}")
    folded = fold_text(ach.text)
    for token in job_tokens:
        phrase = fold_text(token.replace("_", " "))
        if phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", folded):
            reasons.append(f"achievement mentions {phrase}")
    if reasons and ach.metrics:
        # A quantified bullet is stronger evidence, whatever the unit measures.
        units = ", ".join(sorted(ach.metrics)[:3])
        reasons.append(f"quantified result ({units})")
    # Dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _experience_relevant(exp: Experience, job_tokens: set[str]) -> bool:
    blob = fold_text(f"{exp.title} {exp.company} {exp.description or ''}")
    return any(
        re.search(rf"(?<!\w){re.escape(fold_text(token.replace('_', ' ')))}(?!\w)", blob)
        for token in job_tokens
        if token
    )
