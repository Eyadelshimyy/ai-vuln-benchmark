"""A public-health candidate must be scored on her experience, not on a trade's words.

Every vacancy that fitted such a profile used to score 0%: the posting's page chrome
(deadlines, clock times, salary figures, upper-case headings) counted as missing
requirements, and nothing compared what the profile does claim with what the
posting talks about.
"""

from __future__ import annotations

from pathlib import Path

from jobbot.jobs.parsing import looks_like_page_metadata, parse_job_text
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.matching.scoring import blind_matcher_warning
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch, MatchStrength
from tests.fixtures.profile import public_health_profile_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jobs"


def _job(name: str, job_id: str) -> JobPosting:
    return parse_job_text((FIXTURES / name).read_text(encoding="utf-8"), job_id=job_id)


def _candidate() -> Candidate:
    return Candidate.model_validate(public_health_profile_dict())


def _roster_match(job_id: str) -> JobMatch:
    return RuleBasedJobAnalyzer().analyze(_candidate(), _job("zoonosis_roster_page_en.txt", job_id))


def test_a_matching_public_health_posting_scores_well() -> None:
    match = RuleBasedJobAnalyzer().analyze(
        _candidate(), _job("public_health_epidemiology_es.txt", "J0700")
    )

    assert match.score >= 50
    strong = " ".join(i.label.casefold() for i in match.by_strength(MatchStrength.STRONG))
    assert "vigilancia epidemiológica" in strong
    assert "investigación de brotes" in strong


def test_achievements_the_posting_echoes_are_evidence() -> None:
    match = RuleBasedJobAnalyzer().analyze(
        _candidate(), _job("public_health_epidemiology_es.txt", "J0709")
    )

    strong = match.by_strength(MatchStrength.STRONG)
    echoed = [i for i in strong if i.label.startswith("achievement:")]
    assert echoed
    assert all("echoes" in item.detail for item in echoed)


def test_a_roster_page_scores_above_zero_on_the_candidates_own_words() -> None:
    """Regression: a zoonosis roster page scored 1% for a zoonosis epidemiologist."""
    match = _roster_match("J0701")

    assert match.score >= 20
    credited = " | ".join(
        i.label.casefold()
        for i in match.items
        if i.strength in {MatchStrength.STRONG, MatchStrength.PARTIAL}
    )
    assert "epidemiology" in credited
    assert "laboratory" in credited


def test_page_chrome_is_never_a_missing_requirement() -> None:
    match = _roster_match("J0702")

    missing = " | ".join(i.label for i in match.by_strength(MatchStrength.MISSING))
    for chrome in (
        "Deadline",
        "11:59",
        "USD",
        "September 18",
        "DESCRIPTION OF DUTIES",
        "Closing Date",
        "Primary Location",
        "Contractual Agreement",
    ):
        assert chrome not in missing


def test_page_chrome_is_filtered_from_jobs_stored_before_the_fix() -> None:
    """Postings already in the database keep their noisy skills; the matcher skips them."""
    job = JobPosting(
        id="J0703",
        title="Advisor, Health Emergencies",
        company="Neutral Health Organization",
        skills=["Deadline: 2026-10-13 23:59 COT", "027.00 USD", "Closing Date", "October 13"],
        description="Advisor for health emergencies and epidemiological surveillance.",
    )

    match = RuleBasedJobAnalyzer().analyze(_candidate(), job)

    missing = " | ".join(i.label for i in match.by_strength(MatchStrength.MISSING))
    assert "Deadline" not in missing
    assert "USD" not in missing
    assert "October 13" not in missing


def test_an_unrelated_posting_gets_no_free_credit() -> None:
    """Domain-agnostic cuts both ways: sales stays low for an epidemiologist."""
    analyzer = RuleBasedJobAnalyzer()
    candidate = _candidate()

    sales = analyzer.analyze(candidate, _job("field_sales_es.txt", "J0704"))
    fitting = analyzer.analyze(candidate, _job("public_health_epidemiology_es.txt", "J0705"))

    assert sales.score < 30
    assert not any(i.label.startswith("profile:") for i in sales.items)
    assert fitting.score > sales.score


def test_held_titles_count_across_gender_and_language() -> None:
    """'Consultora' held and 'Consultant' wanted are the same title."""
    match = _roster_match("J0706")

    role = next(i for i in match.items if i.label.startswith("role:"))
    assert role.strength in {MatchStrength.STRONG, MatchStrength.PARTIAL}


def test_profile_evidence_never_lowers_a_score() -> None:
    """A claim the posting mentions adds credit; one it does not mention costs nothing."""
    candidate = _candidate()
    job = _job("public_health_epidemiology_es.txt", "J0707")
    raw = public_health_profile_dict()
    raw["skills"]["extra"] = ["Cartografía participativa", "Educación comunitaria"]

    base = RuleBasedJobAnalyzer().analyze(candidate, job)
    richer = RuleBasedJobAnalyzer().analyze(Candidate.model_validate(raw), job)

    assert richer.score >= base.score


def test_page_metadata_is_not_a_skill() -> None:
    job = _job("zoonosis_roster_page_en.txt", "J0708")

    skills = " | ".join(job.skills)
    for chrome in ("Deadline", "11:59 PM", "027.00 USD", "September 30", "PURPOSE OF CONSULTANCY"):
        assert chrome not in skills


def test_metadata_shapes_are_recognised_and_real_skills_are_not() -> None:
    for chrome in (
        "Deadline: 2026-09-30 23:59 ET",
        "11:59 PM Eastern Time",
        "027.00 USD",
        "USD",
        "September 18",
        "1 Oct 2026",
        "Closing: 30/09/2026",
        "DESCRIPTION OF DUTIES",
        "Renta bruta mensual: $2.300.000",
        "Primary Location: Santiago",
        "Contractual Agreement: Fixed-term",
        "Closing Date: October 15",
    ):
        assert looks_like_page_metadata(chrome), chrome
    for skill in (
        "Vigilancia epidemiológica",
        "COVID-19",
        "ISO 9001",
        "SEO",
        "C++",
        "IT SKILLS",
        "Ventilación mecánica",
    ):
        assert not looks_like_page_metadata(skill), skill


def test_every_job_at_zero_is_a_matcher_alert() -> None:
    warning = blind_matcher_warning([0.0, 0.0, 0.0])

    assert warning is not None
    assert "3" in warning


def test_one_job_above_zero_is_not_an_alert() -> None:
    assert blind_matcher_warning([0.0, 12.5, 0.0]) is None


def test_a_single_job_at_zero_is_not_enough_evidence() -> None:
    assert blind_matcher_warning([0.0]) is None
    assert blind_matcher_warning([]) is None
