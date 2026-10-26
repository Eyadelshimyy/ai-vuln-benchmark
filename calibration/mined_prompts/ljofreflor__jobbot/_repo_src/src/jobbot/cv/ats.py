"""ATS-oriented CV generation helpers."""

from __future__ import annotations

from pathlib import Path

from jobbot.cv.renderer import render_cv_ats
from jobbot.cv.selection import SelectionResult
from jobbot.models.candidate import Candidate


def build_ats_text(
    candidate: Candidate,
    templates_dir: Path,
    selection: SelectionResult | None = None,
    *,
    credit: bool = False,
) -> str:
    return render_cv_ats(candidate, templates_dir, selection=selection, credit=credit)
