"""Build Indeed sync package texts from Candidate (facts only; no invention)."""

from __future__ import annotations

from dataclasses import dataclass

from jobbot.branding import stamp_description
from jobbot.models.candidate import Candidate
from jobbot.models.targets import DEFAULT_CONSTRAINTS, ProfileTarget


@dataclass(frozen=True)
class IndeedSyncPackage:
    headline: str
    summary: str
    skills: list[str]
    experience_blocks: list[str]


def truncate(text: str, max_len: int | None) -> str:
    text = text.strip()
    if max_len is None or len(text) <= max_len:
        return text
    cut = text[: max_len - 1].rstrip()
    return cut + "…"


def build_indeed_sync_package(candidate: Candidate) -> IndeedSyncPackage:
    cons = DEFAULT_CONSTRAINTS[ProfileTarget.INDEED]
    headline = truncate(candidate.personal.headline, cons.headline_max)
    summary = stamp_description(candidate.summary or "", max_len=cons.summary_max)
    skills = list(candidate.skills.all_skills())
    blocks: list[str] = []
    for exp in candidate.experience:
        lines = [f"{exp.title} @ {exp.company}"]
        if exp.start_date or exp.end_date or exp.current:
            end = "present" if exp.current else (exp.end_date or "?")
            lines.append(f"{exp.start_date} – {end}")
        if exp.location:
            lines.append(exp.location)
        for ach in exp.achievements:
            lines.append(f"• {ach.text}")
        body = "\n".join(lines)
        blocks.append(truncate(body, cons.experience_description_max))
    return IndeedSyncPackage(
        headline=headline,
        summary=summary,
        skills=skills,
        experience_blocks=blocks,
    )


def render_sync_package_markdown(package: IndeedSyncPackage) -> str:
    lines = [
        "# Indeed sync package (from profile.yaml)",
        "",
        "## Headline",
        package.headline,
        "",
        "## Summary",
        package.summary or "_(empty)_",
        "",
        "## Skills",
        ", ".join(package.skills) if package.skills else "_(none)_",
        "",
        "## Experience (paste per role)",
    ]
    for i, block in enumerate(package.experience_blocks, start=1):
        lines.extend(["", f"### Role {i}", "```", block, "```"])
    lines.append("")
    return "\n".join(lines)
