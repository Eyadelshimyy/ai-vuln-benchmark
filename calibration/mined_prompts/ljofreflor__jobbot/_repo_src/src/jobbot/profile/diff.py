"""Lightweight structural diff helpers for profiles."""

from __future__ import annotations

from typing import Any


def summarize_profile(data: dict[str, Any]) -> dict[str, Any]:
    personal = data.get("personal") or {}
    experience = data.get("experience") or []
    education = data.get("education") or []
    publications = data.get("publications") or []
    skills = data.get("skills") or {}
    skill_count = sum(len(v) for v in skills.values() if isinstance(v, list))
    achievements = sum(len(e.get("achievements") or []) for e in experience)
    return {
        "name": personal.get("name"),
        "headline": personal.get("headline"),
        "experiences": len(experience),
        "achievements": achievements,
        "education": len(education),
        "skills": skill_count,
        "publications": len(publications),
    }


def compare_summaries(
    local: dict[str, Any],
    generated: dict[str, Any],
) -> list[str]:
    """Return human-readable lines comparing two profile summaries."""
    lines: list[str] = []
    keys = (
        "name",
        "headline",
        "experiences",
        "achievements",
        "education",
        "skills",
        "publications",
    )
    for key in keys:
        left = local.get(key)
        right = generated.get(key)
        mark = "=" if left == right else "!"
        lines.append(f"{mark} {key:14} local={left!r}  generated={right!r}")
    return lines
