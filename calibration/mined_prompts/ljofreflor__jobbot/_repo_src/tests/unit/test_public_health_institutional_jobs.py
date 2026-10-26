"""Anonymous institutional-health fixtures for #118 (comment on non-tech 0% scores).

Three families of role a public-health profile fits by experience (networks,
sanitary authorities, regulatory work) — not by a tech title. Acceptance from
the issue comment: score above 0%, select_for_job keeps a tagged achievement
with that reason, page chrome is never a missing requirement, and an all-zero
batch surfaces the matcher alert.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.cv.selection import select_for_job, write_selection_json
from jobbot.jobs.parsing import parse_job_text
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.matching.scoring import blind_matcher_warning
from jobbot.models.candidate import Candidate
from jobbot.models.match import MatchStrength
from tests.fixtures.profile import public_health_profile_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jobs"

INSTITUTIONAL_JOBS = (
    "kam_institucional_salud.txt",
    "acceso_mercado_salud.txt",
    "asuntos_institucionales_salud.txt",
)

_TAGGED = frozenset(
    {
        "coordinacion_interinstitucional",
        "gestion_de_redes",
        "relacionamiento_institucional",
        "autoridad_sanitaria",
        "regulatorio",
    }
)

_PAGE_CHROME = (
    "Deadline",
    "11:59",
    "Closing Date",
    "DESCRIPTION OF DUTIES",
    "PURPOSE OF THE ROLE",
    "Primary Location",
    "Contractual Agreement",
    "Eastern Time",
    "Renta bruta",
)


def _candidate() -> Candidate:
    return Candidate.model_validate(public_health_profile_dict())


def _job(name: str, job_id: str):
    return parse_job_text((FIXTURES / name).read_text(encoding="utf-8"), job_id=job_id)


@pytest.mark.parametrize("fixture_name", INSTITUTIONAL_JOBS)
def test_institutional_health_postings_score_above_zero(fixture_name: str) -> None:
    match = RuleBasedJobAnalyzer().analyze(_candidate(), _job(fixture_name, "J0800"))

    assert match.score > 0


@pytest.mark.parametrize("fixture_name", INSTITUTIONAL_JOBS)
def test_select_for_job_keeps_a_tagged_achievement_with_reason(
    fixture_name: str, tmp_path: Path
) -> None:
    candidate = _candidate()
    job = _job(fixture_name, "J0801")
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    selection = select_for_job(candidate, job, match)

    tagged = [
        item
        for item in selection.selected_achievements
        if any(reason.startswith("tag overlap:") for reason in item.reason)
        and any(
            tag in reason
            for reason in item.reason
            for tag in _TAGGED
            if reason.startswith("tag overlap:")
        )
    ]
    assert tagged, selection.selected_achievements

    path = tmp_path / "selection.json"
    write_selection_json(selection, path)
    payload = path.read_text(encoding="utf-8")
    assert "tag overlap:" in payload
    assert any(tag in payload for tag in _TAGGED)


@pytest.mark.parametrize("fixture_name", INSTITUTIONAL_JOBS)
def test_page_chrome_is_not_a_missing_requirement_on_institutional_fixtures(
    fixture_name: str,
) -> None:
    match = RuleBasedJobAnalyzer().analyze(_candidate(), _job(fixture_name, "J0802"))
    missing = " | ".join(i.label for i in match.by_strength(MatchStrength.MISSING))

    for chrome in _PAGE_CHROME:
        assert chrome not in missing, chrome


def test_all_zero_institutional_batch_shows_matcher_alert() -> None:
    """If every institutional posting scored 0%, the CLI must not stay silent."""
    warning = blind_matcher_warning([0.0, 0.0, 0.0])

    assert warning is not None
    assert "3" in warning
