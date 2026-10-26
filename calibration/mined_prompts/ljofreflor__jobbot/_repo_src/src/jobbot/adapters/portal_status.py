"""Profile consistency status across local / Indeed / LinkedIn snapshots."""

from __future__ import annotations

from typing import Any

from jobbot.adapters.diff_engine import load_snapshot
from jobbot.config import JobbotConfig
from jobbot.profile.loader import load_profile


def build_profile_status(config: JobbotConfig) -> dict[str, Any]:
    candidate = load_profile(config.profile_path)
    indeed = load_snapshot(config.output_dir, "indeed")
    linkedin = load_snapshot(config.output_dir, "linkedin")

    def mark(local: str | None, remote: str | None) -> str:
        if remote is None:
            return "—"
        if not local and not remote:
            return "—"
        if (local or "").strip() == (remote or "").strip():
            return "✓"
        return "!"

    rows = [
        {
            "section": "Headline",
            "local": "✓",
            "indeed": mark(candidate.personal.headline, indeed.headline if indeed else None),
            "linkedin": mark(
                candidate.personal.headline, linkedin.headline if linkedin else None
            ),
        },
        {
            "section": "Experience",
            "local": "✓",
            "indeed": (
                "✓"
                if indeed and indeed.experience
                else ("—" if indeed is None else "!")
            ),
            "linkedin": (
                "✓"
                if linkedin and linkedin.experience
                else ("—" if linkedin is None else "!")
            ),
        },
        {
            "section": "Skills",
            "local": "✓",
            "indeed": (
                "✓"
                if indeed and indeed.skills
                else ("—" if indeed is None else "!")
            ),
            "linkedin": (
                "✓"
                if linkedin and linkedin.skills
                else ("—" if linkedin is None else "!")
            ),
        },
    ]
    return {"rows": rows}
