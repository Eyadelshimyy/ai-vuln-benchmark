"""Matching unit tests — never invent skills."""

from __future__ import annotations

from pathlib import Path

from jobbot.jobs.parsing import parse_job_text
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.models.candidate import Candidate
from jobbot.models.match import MatchStrength
from jobbot.profile.loader import load_profile
from tests.fixtures.profile import non_data_profile_dict, sample_profile_dict

MINING_JOB = (
    "Title: Coordinador de Proyectos Mineros\n"
    "Company: Andes Ingeniería SpA\n"
    "Location: Santiago, Chile\n"
    "Requisitos:\n"
    "- Título de Ingeniero (Geofísico, Civil, Minas o afín).\n"
    "- 5+ años de experiencia en gestión de proyectos en minería o energía.\n"
    "- Metodologías ágiles (Scrum) y gestión de proyectos.\n"
    "- Manejo de AutoCAD y herramientas de procesamiento de datos geofísicos.\n"
)


def test_strong_match_for_aligned_job() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    job = parse_job_text(
        Path("tests/fixtures/jobs/senior_ds_retail.txt").read_text(encoding="utf-8"),
        job_id="J0001",
    )
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    strong = {i.label for i in match.by_strength(MatchStrength.STRONG)}
    assert match.score >= 50
    assert any("Python" in s for s in strong) or any(
        i.strength == MatchStrength.STRONG for i in match.items if "python" in i.label.lower()
    )


def test_unknown_language_not_converted_to_missing() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    job = parse_job_text(
        Path("tests/fixtures/jobs/senior_ds_retail.txt").read_text(encoding="utf-8"),
        job_id="J0001",
    )
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    unknowns = match.by_strength(MatchStrength.UNKNOWN)
    missings = match.by_strength(MatchStrength.MISSING)
    assert any("english" in i.label.lower() for i in unknowns)
    assert not any("english" in i.label.lower() for i in missings)


def test_poor_fit_has_missing_frontend_skills(project_root: Path) -> None:
    candidate = load_profile(project_root / "data" / "profile.example.yaml")
    job = parse_job_text(
        (project_root / "tests/fixtures/jobs/junior_frontend.txt").read_text(encoding="utf-8"),
        job_id="J0002",
    )
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    assert match.score <= 50
    missing = match.by_strength(MatchStrength.MISSING)
    # Frontend stack should not be invented as known
    assert match.score < 70
    assert missing or match.score < 40
    strong_labels = " ".join(i.label.lower() for i in match.by_strength(MatchStrength.STRONG))
    assert "react" not in strong_labels
    assert "typescript" not in strong_labels


def test_product_manager_is_not_perfect_ds_match() -> None:
    """Regression: ubiquitous Python/SQL/ML mentions must not score 100% on a PM role."""
    candidate = Candidate.model_validate(sample_profile_dict())
    job = parse_job_text(
        "Title: Product Manager\nCompany: NeuralWorks\n"
        "We need Python, SQL, and machine learning literacy to work with DS.\n",
        job_id="J0098",
    )
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    assert match.score <= 40
    role = next(i for i in match.items if i.label.startswith("role:"))
    assert role.strength == MatchStrength.MISSING


def test_non_data_candidate_matches_a_job_in_their_own_field() -> None:
    """Regression: the matcher scored 0% for any profile outside data science."""
    candidate = Candidate.model_validate(non_data_profile_dict())
    job = parse_job_text(MINING_JOB, job_id="J0500")

    match = RuleBasedJobAnalyzer().analyze(candidate, job)

    assert match.score >= 60
    strong = " ".join(i.label.lower() for i in match.by_strength(MatchStrength.STRONG))
    assert "scrum" in strong
    assert "autocad" in strong
    assert "geofísico" in strong


def test_long_spanish_requirements_are_not_dropped() -> None:
    """Requirement lines over 60 characters used to disappear from the report."""
    candidate = Candidate.model_validate(non_data_profile_dict())
    job = parse_job_text(MINING_JOB, job_id="J0501")

    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    labels = " ".join(i.label for i in match.items)

    assert "gestión de proyectos en minería o energía" in labels
    assert "herramientas de procesamiento de datos geofísicos" in labels


def test_unclassified_candidate_is_not_assumed_to_be_a_data_scientist() -> None:
    """A geophysicist applying to a DS role must not get a strong role match."""
    candidate = Candidate.model_validate(non_data_profile_dict())
    job = parse_job_text(
        "Title: Senior Data Scientist\nCompany: NeuralWorks\n"
        "Requirements:\n- Python\n- SQL\n- Machine Learning\n",
        job_id="J0502",
    )

    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    role = next(i for i in match.items if i.label.startswith("role:"))

    assert role.strength != MatchStrength.STRONG
    assert match.score < 50


ENGLISH_DS_JOB = (
    "Title: Senior Data Scientist\n"
    "Company: Example Remote Co\n"
    "Requirements:\n"
    "- Statistics\n"
    "- Predictive modelling\n"
    "- Data science\n"
    "- Pandas\n"
)


def test_spanish_profile_backs_english_requirements() -> None:
    """Regression: an MSc in Estadística and 'modelos predictivos' read as missing.

    The posting names the skill in English and the profile in Spanish; a phrase
    of the candidate's own prose ('data science') was only half matched because
    prose was split into single words.
    """
    profile = sample_profile_dict()
    profile["summary"] = (
        "Experiencia construyendo modelos predictivos y liderando equipos de data science."
    )
    candidate = Candidate.model_validate(profile)
    job = parse_job_text(ENGLISH_DS_JOB, job_id="J0503")

    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    by_label = {i.label.lower(): i.strength for i in match.items}

    assert by_label["statistics"] == MatchStrength.STRONG
    assert by_label["predictive modelling"] == MatchStrength.STRONG
    assert by_label["data science"] == MatchStrength.STRONG
    assert by_label["pandas"] == MatchStrength.MISSING


def test_applied_scientist_outscores_product_manager() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    ds = parse_job_text(
        "Title: Applied Scientist\nCompany: NeuralWorks\n"
        "Requirements:\n- Python\n- SQL\n- Causal Inference\n- RCT and DiD\n",
        job_id="J0097",
    )
    pm = parse_job_text(
        "Title: Product Manager\nCompany: NeuralWorks\n"
        "Requirements:\n- Python\n- SQL\n- Machine Learning\n",
        job_id="J0096",
    )
    analyzer = RuleBasedJobAnalyzer()
    ds_match = analyzer.analyze(candidate, ds)
    pm_match = analyzer.analyze(candidate, pm)
    assert ds_match.score > pm_match.score
    assert ds_match.score >= 50
