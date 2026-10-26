"""Free-text application answers are checked against the profile and the posting."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jobbot.applications.answer_check import (
    AnswerCheck,
    QuestionKind,
    Verdict,
    check_answer,
    classify_question,
    render_answer_check,
)
from jobbot.exit_codes import MANUAL_CHALLENGE, SUCCESS, VALIDATION_FAILURE
from jobbot.jobs.parsing import parse_job_text
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from tests.fixtures.profile import two_degrees_profile_dict

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "jobs" / (
    "analista_planificacion_es.txt"
)
FACTUAL_ES = "Describe tu formación y tu experiencia con pronóstico de demanda."
WHY_ES = "¿Por qué te interesa trabajar en Fabrikam Logística?"


@pytest.fixture
def candidate() -> Candidate:
    return Candidate.model_validate(two_degrees_profile_dict())


@pytest.fixture
def job() -> JobPosting:
    return parse_job_text(FIXTURE.read_text(encoding="utf-8"), job_id="J0700")


def _problems(check: AnswerCheck) -> str:
    return " | ".join(p for s in check.sentences for p in s.problems)


def test_a_degree_paired_with_the_wrong_institution_is_misattributed(
    candidate: Candidate, job: JobPosting
) -> None:
    answer = "Soy Ingeniera Civil en Computación y Magíster en Estadística (Universidad del Sur)."

    check = check_answer(FACTUAL_ES, answer, candidate, job)

    assert check.verdict is Verdict.REJECTED
    problems = _problems(check)
    assert "misattributed" in problems
    assert "Ingeniería Civil en Computación" in problems
    assert "Universidad del Litoral" in problems
    assert "Magíster en Estadística" not in problems


def test_degrees_each_with_its_own_institution_pass(
    candidate: Candidate, job: JobPosting
) -> None:
    answer = (
        "Soy Ingeniera Civil en Computación de la Universidad del Litoral "
        "y Magíster en Estadística de la Universidad del Sur."
    )

    check = check_answer(FACTUAL_ES, answer, candidate, job)

    assert check.verdict is Verdict.OK, _problems(check)
    wheres = {e.where for e in check.sentences[0].evidence}
    assert "education litoral-ingenieria" in wheres
    assert "education sur-magister" in wheres


def test_a_fully_backed_factual_answer_passes_with_its_trace(
    candidate: Candidate, job: JobPosting
) -> None:
    answer = (
        "En Northwind Analítica, como Analista de Datos, reduje en 18% el quiebre de stock "
        "con un modelo de pronóstico semanal. Estudié el Magíster en Estadística en la "
        "Universidad del Sur."
    )

    check = check_answer(FACTUAL_ES, answer, candidate, job)

    assert check.kind is QuestionKind.FACTUAL
    assert check.verdict is Verdict.OK, _problems(check)
    first = {e.where for e in check.sentences[0].evidence}
    assert "experience northwind-analista · northwind-quiebre" in first


def test_an_invented_figure_is_rejected(candidate: Candidate, job: JobPosting) -> None:
    answer = "En Northwind Analítica reduje en 35% el quiebre de stock."

    check = check_answer(FACTUAL_ES, answer, candidate, job)

    assert check.verdict is Verdict.REJECTED
    assert "35" in _problems(check)


def test_a_figure_credited_to_the_wrong_employer_is_misattributed(
    candidate: Candidate, job: JobPosting
) -> None:
    answer = "En Contoso Retail reduje en 18% el quiebre de stock."

    check = check_answer(FACTUAL_ES, answer, candidate, job)

    assert check.verdict is Verdict.REJECTED
    problems = _problems(check)
    assert "misattributed" in problems
    assert "Northwind Analítica" in problems


def test_a_skill_the_posting_asks_for_is_not_backed_by_the_posting(
    candidate: Candidate, job: JobPosting
) -> None:
    """Echoing the posting's requirement is the invention most likely to slip through."""
    check = check_answer(FACTUAL_ES, "Tengo experiencia en optimización de rutas.", candidate, job)

    assert check.verdict is Verdict.REJECTED
    assert "optimización de rutas" in _problems(check).casefold()


def test_an_unknown_institution_is_rejected(candidate: Candidate, job: JobPosting) -> None:
    check = check_answer(
        FACTUAL_ES, "Estudié el Magíster en Estadística en la Universidad de Chile.", candidate, job
    )

    assert check.verdict is Verdict.REJECTED
    assert "Universidad de Chile" in _problems(check)


@pytest.mark.parametrize(
    "question",
    [
        WHY_ES,
        "¿Qué te motiva a postular a este cargo?",
        "¿Por qué quieres unirte a nuestro equipo?",
        "Carta de presentación",
        "Why do you want to work at Fabrikam Logística?",
        "Why are you interested in this role?",
        "What motivates you to apply?",
        "Why Fabrikam?",
        "Cover letter",
    ],
)
def test_motivation_questions_are_classified_in_spanish_and_english(question: str) -> None:
    assert classify_question(question, company="Fabrikam Logística") is QuestionKind.MOTIVATION


@pytest.mark.parametrize(
    "question",
    [
        FACTUAL_ES,
        "Describe your experience with SQL.",
        "¿Cuántos años de experiencia tienes con Python?",
        "Explain why your last forecast model worked.",
    ],
)
def test_factual_questions_are_not_motivation(question: str) -> None:
    assert classify_question(question, company="Fabrikam Logística") is QuestionKind.FACTUAL


def test_a_motivation_answer_is_never_auto_approved(candidate: Candidate, job: JobPosting) -> None:
    answer = (
        "Me interesa Fabrikam Logística porque su equipo de planificación construye modelos "
        "de pronóstico para la red de centros de distribución."
    )

    check = check_answer(WHY_ES, answer, candidate, job)

    assert check.kind is QuestionKind.MOTIVATION
    assert check.verdict is Verdict.NEEDS_CANDIDATE, _problems(check)
    assert not check.warnings
    quotes = [e.quote for e in check.sentences[0].evidence if e.source == "posting"]
    assert any("centros de distribución" in q for q in quotes)


def test_an_english_motivation_question_is_needs_candidate(
    candidate: Candidate, job: JobPosting
) -> None:
    check = check_answer(
        "Why do you want to work with us?",
        "Su equipo construye modelos de pronóstico para centros de distribución.",
        candidate,
        job,
    )

    assert check.verdict is Verdict.NEEDS_CANDIDATE, _problems(check)


def test_a_why_company_answer_listing_achievements_does_not_address_the_company(
    candidate: Candidate, job: JobPosting
) -> None:
    answer = "Reduje en 18% el quiebre de stock. Migré 40 servicios a contenedores."

    check = check_answer(WHY_ES, answer, candidate, job)

    assert check.verdict is Verdict.NEEDS_CANDIDATE
    assert any("does not address the company" in w for w in check.warnings)
    assert check.posting_hooks
    assert any("centros de distribución" in hook for hook in check.posting_hooks)


def test_a_company_claim_the_posting_does_not_make_is_rejected(
    candidate: Candidate, job: JobPosting
) -> None:
    answer = "Me interesa porque Fabrikam Logística es líder mundial en comercio electrónico."

    check = check_answer(WHY_ES, answer, candidate, job)

    assert check.verdict is Verdict.REJECTED
    assert "posting" in _problems(check)


def test_the_rendered_trace_names_backing_and_problems(
    candidate: Candidate, job: JobPosting
) -> None:
    answer = "Soy Ingeniera Civil en Computación y Magíster en Estadística (Universidad del Sur)."

    text = "\n".join(render_answer_check(check_answer(FACTUAL_ES, answer, candidate, job), job))

    assert "rejected" in text
    assert "misattributed" in text
    assert "education sur-magister" in text


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    (tmp_path / "data").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "data" / "profile.yaml").write_text(
        yaml.safe_dump(two_degrees_profile_dict(), allow_unicode=True), encoding="utf-8"
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    parsed = parse_job_text(FIXTURE.read_text(encoding="utf-8"), job_id="PENDING")
    stored = JobRepository(session).upsert_external(parsed)
    return stored.id


def test_cli_rejects_a_misattributed_answer(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    job_id = _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        [
            "application",
            "check-answer",
            job_id,
            "--question",
            FACTUAL_ES,
            "--text",
            "Soy Ingeniera Civil en Computación y Magíster en Estadística (Universidad del Sur).",
        ],
        standalone_mode=False,
    )

    out = capsys.readouterr().out
    assert code == VALIDATION_FAILURE
    assert "misattributed" in out
    assert "Universidad del Litoral" in out
    assert not list((tmp_path / "output" / "ops").glob("failures/*")), "a verdict is not a defect"


def test_cli_reads_the_answer_from_a_file_and_passes(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = _workspace(tmp_path, project_root, monkeypatch)
    answer = tmp_path / "answer.txt"
    answer.write_text(
        "Estudié el Magíster en Estadística en la Universidad del Sur.", encoding="utf-8"
    )
    from jobbot.cli import run_cli

    code = run_cli(
        ["application", "check-answer", job_id, "-q", FACTUAL_ES, "--file", str(answer)],
        standalone_mode=False,
    )

    assert code == SUCCESS


def test_cli_motivation_needs_the_candidate(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        [
            "application",
            "check-answer",
            job_id,
            "-q",
            WHY_ES,
            "--text",
            "Reduje en 18% el quiebre de stock.",
        ],
        standalone_mode=False,
    )

    assert code == MANUAL_CHALLENGE
