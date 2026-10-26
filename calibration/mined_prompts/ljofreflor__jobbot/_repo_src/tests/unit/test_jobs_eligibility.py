"""Each posting condition against the candidate: meets, ask the candidate, or dealbreaker."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from jobbot.jobs.conditions import Condition, ConditionKind, posting_conditions
from jobbot.jobs.eligibility import (
    ApplicationAnswers,
    Verdict,
    VerdictStatus,
    assess_conditions,
    default_answers_path,
    format_conditions_report,
    has_dealbreaker,
    load_application_answers,
)
from jobbot.jobs.parsing import parse_job_text
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.ops.pii_guard import is_blocked_path
from tests.fixtures.profile import sample_profile_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jobs"


def _job(name: str) -> JobPosting:
    return parse_job_text((FIXTURES / name).read_text(encoding="utf-8"), job_id="J0910")


def _candidate(**personal: Any) -> Candidate:
    data = sample_profile_dict()
    data["personal"].update(personal)
    return Candidate.model_validate(data)


def _verdicts(
    name: str,
    *,
    answers: ApplicationAnswers | None = None,
    candidate: Candidate | None = None,
) -> list[Verdict]:
    return assess_conditions(
        posting_conditions(_job(name)),
        candidate or _candidate(country="Chile"),
        answers or ApplicationAnswers(),
        today=date(2026, 10, 3),
    )


def _of(verdicts: list[Verdict], kind: ConditionKind) -> list[Verdict]:
    return [v for v in verdicts if v.condition.kind is kind]


def test_closed_posting_is_a_dealbreaker() -> None:
    verdicts = _verdicts("conditions_closed_es.txt")
    assert _of(verdicts, ConditionKind.CLOSED)[0].status is VerdictStatus.DEALBREAKER
    assert has_dealbreaker(verdicts)


def test_residency_in_other_countries_is_a_dealbreaker() -> None:
    verdict = _of(_verdicts("conditions_residency_en.txt"), ConditionKind.RESIDENCY)[0]
    assert verdict.status is VerdictStatus.DEALBREAKER
    assert "CL" in verdict.reason or "Chile" in verdict.reason


def test_residency_including_the_candidate_country_meets_with_evidence() -> None:
    verdict = _of(_verdicts("conditions_residency_es.txt"), ConditionKind.RESIDENCY)[0]
    assert verdict.status is VerdictStatus.MEETS
    assert "profile.yaml" in verdict.reason


def test_answers_residence_country_wins_over_profile_country() -> None:
    answers = ApplicationAnswers(residence_country="Peru")
    verdict = _of(
        _verdicts("conditions_residency_en.txt", answers=answers), ConditionKind.RESIDENCY
    )[0]
    assert verdict.status is VerdictStatus.MEETS


def test_unknown_candidate_country_asks() -> None:
    verdict = _of(
        _verdicts("conditions_residency_en.txt", candidate=_candidate(country=None)),
        ConditionKind.RESIDENCY,
    )[0]
    assert verdict.status is VerdictStatus.ASK
    assert verdict.question


def test_c1_required_vs_b1_answer_is_a_dealbreaker() -> None:
    answers = ApplicationAnswers(languages={"english": "B1"})
    verdict = _of(
        _verdicts("conditions_residency_en.txt", answers=answers), ConditionKind.LANGUAGE
    )[0]
    assert verdict.status is VerdictStatus.DEALBREAKER


def test_conversational_vs_b1_answer_meets() -> None:
    answers = ApplicationAnswers(languages={"inglés": "B1"})
    verdict = _of(
        _verdicts("conditions_anywhere_en.txt", answers=answers), ConditionKind.LANGUAGE
    )[0]
    assert verdict.status is VerdictStatus.MEETS
    assert "application_answers" in verdict.reason


def test_unknown_language_level_asks_the_exact_question() -> None:
    verdict = _of(_verdicts("conditions_residency_en.txt"), ConditionKind.LANGUAGE)[0]
    assert verdict.status is VerdictStatus.ASK
    assert verdict.question is not None
    assert "English" in verdict.question
    assert "C1" in verdict.question


def test_profile_language_skill_counts_as_evidence() -> None:
    data = sample_profile_dict()
    data["personal"]["country"] = "Chile"
    data["skills"]["idiomas"] = ["Inglés avanzado (C1)"]
    candidate = Candidate.model_validate(data)
    verdict = _of(
        _verdicts("conditions_residency_en.txt", candidate=candidate), ConditionKind.LANGUAGE
    )[0]
    assert verdict.status is VerdictStatus.MEETS
    assert "profile.yaml" in verdict.reason


def test_desirable_language_below_level_is_only_a_note() -> None:
    condition = Condition(
        kind=ConditionKind.LANGUAGE,
        source="Inglés C1 deseable",
        language="en",
        level="C1",
        mandatory=False,
    )
    verdict = assess_conditions(
        [condition], _candidate(), ApplicationAnswers(languages={"en": "A2"})
    )[0]
    assert verdict.status is VerdictStatus.NOTE


def test_excluyente_requirement_not_in_profile_asks_never_rejects() -> None:
    verdicts = _of(_verdicts("conditions_residency_es.txt"), ConditionKind.REQUIREMENT)
    mandatory = [v for v in verdicts if v.condition.mandatory]
    assert mandatory[0].status is VerdictStatus.ASK
    assert "conciliación bancaria" in (mandatory[0].question or "")
    desirable = [v for v in verdicts if v.condition.mandatory is False]
    assert all(v.status is VerdictStatus.NOTE for v in desirable)


def test_excluyente_requirement_backed_by_the_profile_meets() -> None:
    data = sample_profile_dict()
    data["skills"]["finance"] = ["Conciliación Bancaria"]
    verdicts = _of(
        _verdicts("conditions_residency_es.txt", candidate=Candidate.model_validate(data)),
        ConditionKind.REQUIREMENT,
    )
    mandatory = [v for v in verdicts if v.condition.mandatory]
    assert mandatory[0].status is VerdictStatus.MEETS
    assert "Conciliación Bancaria" in mandatory[0].reason


def test_contractor_contract_follows_the_answer() -> None:
    def status(value: bool | None) -> VerdictStatus:
        answers = ApplicationAnswers(accepts_contractor=value)
        verdicts = _verdicts("conditions_residency_es.txt", answers=answers)
        return _of(verdicts, ConditionKind.CONTRACT)[0].status

    assert status(True) is VerdictStatus.MEETS
    assert status(False) is VerdictStatus.DEALBREAKER
    assert status(None) is VerdictStatus.ASK


def test_hidden_salary_asks_for_an_expectation_until_answered() -> None:
    unanswered = _of(_verdicts("conditions_residency_en.txt"), ConditionKind.SALARY)[0]
    assert unanswered.status is VerdictStatus.ASK
    answered = _of(
        _verdicts(
            "conditions_residency_en.txt",
            answers=ApplicationAnswers(salary_expectation="USD 2000 monthly"),
        ),
        ConditionKind.SALARY,
    )[0]
    assert answered.status is VerdictStatus.NOTE


def test_past_deadline_is_a_dealbreaker_and_future_one_a_note() -> None:
    def status(when: date) -> VerdictStatus:
        condition = Condition(kind=ConditionKind.DEADLINE, source="Fecha límite", when=when)
        return assess_conditions(
            [condition], _candidate(), ApplicationAnswers(), today=date(2026, 10, 3)
        )[0].status

    assert status(date(2026, 9, 1)) is VerdictStatus.DEALBREAKER
    assert status(date(2026, 10, 30)) is VerdictStatus.NOTE


def test_onsite_in_another_country_asks_about_relocation() -> None:
    job = JobPosting(
        id="J0911",
        title="Analyst",
        company="Estudio Neutro",
        location="Colombia",
        remote_type="onsite",
        description="Presencial en nuestra oficina.",
    )
    conditions = posting_conditions(job)

    def status(relocation: bool | None) -> VerdictStatus:
        verdicts = assess_conditions(
            conditions, _candidate(country="Chile"), ApplicationAnswers(relocation=relocation)
        )
        return _of(verdicts, ConditionKind.MODALITY)[0].status

    assert status(None) is VerdictStatus.ASK
    assert status(True) is VerdictStatus.MEETS
    assert status(False) is VerdictStatus.DEALBREAKER


def test_work_authorization_against_answered_countries() -> None:
    condition = Condition(
        kind=ConditionKind.WORK_AUTHORIZATION,
        source="Visa: US citizen/visa only",
        countries=("US",),
        mandatory=True,
    )

    def status(countries: list[str]) -> VerdictStatus:
        answers = ApplicationAnswers(work_authorization=countries)
        return assess_conditions([condition], _candidate(), answers)[0].status

    assert status([]) is VerdictStatus.ASK
    assert status(["CL"]) is VerdictStatus.DEALBREAKER
    assert status(["United States"]) is VerdictStatus.MEETS


def test_answers_file_loads_and_job_answers_override(tmp_path: Path) -> None:
    root = tmp_path
    path = default_answers_path(root)
    path.parent.mkdir(parents=True)
    path.write_text(
        "languages:\n  english: B1\naccepts_contractor: false\nresidence_country: Chile\n",
        encoding="utf-8",
    )
    job_answers = tmp_path / "answers.yaml"
    job_answers.write_text(
        "english_level: C1\nrelocation: sí\nwork_authorization: Chile, Perú\nnotice_period:\n",
        encoding="utf-8",
    )

    answers = load_application_answers(path, job_answers=job_answers)

    assert answers.language_level("en") == "C1"
    assert answers.accepts_contractor is False
    assert answers.relocation is True
    assert answers.work_authorization == ["CL", "PE"]
    assert answers.notice_period is None


def test_missing_answers_file_means_every_answer_is_unknown(tmp_path: Path) -> None:
    answers = load_application_answers(default_answers_path(tmp_path))
    assert answers == ApplicationAnswers()


def test_example_answers_file_is_valid(project_root: Path) -> None:
    example = project_root / "data" / "application_answers.example.yaml"
    answers = load_application_answers(example)
    assert answers.accepts_contractor is None


def test_answers_file_is_local_only() -> None:
    assert is_blocked_path("data/application_answers.yaml")
    assert not is_blocked_path("data/application_answers.example.yaml")


def test_report_shows_marks_questions_and_banner() -> None:
    job = _job("conditions_residency_en.txt")
    verdicts = _verdicts("conditions_residency_en.txt")
    report = format_conditions_report(job, verdicts)
    assert "❌" in report
    assert "⚠️" in report
    assert "DEALBREAKER" in report
    assert "must reside in" in report.casefold()
