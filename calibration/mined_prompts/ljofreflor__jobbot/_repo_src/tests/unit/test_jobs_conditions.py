"""Read a stored posting's special conditions: closed, residency, language, contract…"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from jobbot.jobs.conditions import (
    Condition,
    ConditionKind,
    cefr_rank,
    language_code,
    level_from_text,
    posting_conditions,
)
from jobbot.jobs.parsing import parse_job_text
from jobbot.models.job import JobPosting

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jobs"


def _job(name: str) -> JobPosting:
    return parse_job_text((FIXTURES / name).read_text(encoding="utf-8"), job_id="J0900")


def _kind(conditions: list[Condition], kind: ConditionKind) -> list[Condition]:
    return [c for c in conditions if c.kind is kind]


def test_closed_get_on_board_banner_is_a_condition() -> None:
    conditions = posting_conditions(_job("conditions_closed_es.txt"))

    closed = _kind(conditions, ConditionKind.CLOSED)
    assert closed
    assert "no se reciben" in closed[0].source.casefold()


def test_english_closed_banner_is_a_condition() -> None:
    job = JobPosting(
        id="J0901",
        title="Analyst",
        company="Estudio Neutro",
        description="Closed job - No longer accepting applications\nWe were hiring.",
    )
    assert _kind(posting_conditions(job), ConditionKind.CLOSED)


def test_residency_lists_the_countries_named_in_english() -> None:
    conditions = posting_conditions(_job("conditions_residency_en.txt"))

    residency = _kind(conditions, ConditionKind.RESIDENCY)
    assert len(residency) == 1
    assert residency[0].countries == ("PE", "CO")
    assert "must reside in" in residency[0].source.casefold()


def test_residency_lists_the_countries_named_in_spanish() -> None:
    residency = _kind(
        posting_conditions(_job("conditions_residency_es.txt")), ConditionKind.RESIDENCY
    )
    assert residency[0].countries == ("CL", "AR", "PE", "CO")


def test_reside_anywhere_is_open_residency() -> None:
    residency = _kind(
        posting_conditions(_job("conditions_anywhere_en.txt")), ConditionKind.RESIDENCY
    )
    assert residency[0].anywhere
    assert residency[0].countries == ()


def test_required_cefr_level_is_read_with_its_language() -> None:
    languages = _kind(
        posting_conditions(_job("conditions_residency_en.txt")), ConditionKind.LANGUAGE
    )
    assert len(languages) == 1
    assert languages[0].language == "en"
    assert languages[0].level == "C1"
    assert languages[0].mandatory is True


def test_spanish_level_with_plus_and_necesario_is_mandatory() -> None:
    languages = _kind(
        posting_conditions(_job("conditions_residency_es.txt")), ConditionKind.LANGUAGE
    )
    assert [(c.language, c.level, c.mandatory) for c in languages] == [("en", "B2", True)]


def test_word_level_is_mapped_to_cefr() -> None:
    languages = _kind(
        posting_conditions(_job("conditions_anywhere_en.txt")), ConditionKind.LANGUAGE
    )
    assert languages[0].level == "B1"


def test_excluyente_heading_marks_its_items_mandatory() -> None:
    requirements = _kind(
        posting_conditions(_job("conditions_residency_es.txt")), ConditionKind.REQUIREMENT
    )
    by_text = {c.detail: c.mandatory for c in requirements}
    assert by_text["Experiencia en conciliación bancaria"] is True
    assert by_text["Manejo de planillas dinámicas"] is False
    # The language line under the same heading is a language condition, not a requirement.
    assert not any("inglés" in (c.detail or "").casefold() for c in requirements)


def test_inline_required_and_nice_to_have_in_english() -> None:
    requirements = _kind(
        posting_conditions(_job("conditions_residency_en.txt")), ConditionKind.REQUIREMENT
    )
    by_text = {c.detail: c.mandatory for c in requirements}
    assert by_text["Bank reconciliation experience"] is True
    assert by_text["dashboard design"] is False


def test_contractor_contract_in_both_languages() -> None:
    for name in ("conditions_residency_en.txt", "conditions_residency_es.txt"):
        contracts = _kind(posting_conditions(_job(name)), ConditionKind.CONTRACT)
        assert [c.contract for c in contracts] == ["contractor"], name


def test_indefinite_contract_is_not_contractor() -> None:
    contracts = _kind(
        posting_conditions(_job("conditions_anywhere_en.txt")), ConditionKind.CONTRACT
    )
    assert [c.contract for c in contracts] == ["indefinite"]


def test_salary_hidden_vs_shown() -> None:
    hidden = _kind(posting_conditions(_job("conditions_residency_en.txt")), ConditionKind.SALARY)
    shown = _kind(posting_conditions(_job("conditions_residency_es.txt")), ConditionKind.SALARY)
    assert hidden[0].shown is False
    assert shown[0].shown is True
    assert "1.500" in (shown[0].detail or "")


def test_modality_from_text_and_availability() -> None:
    conditions = posting_conditions(_job("conditions_residency_es.txt"))
    assert [c.modality for c in _kind(conditions, ConditionKind.MODALITY)] == ["remote"]
    assert _kind(conditions, ConditionKind.AVAILABILITY)


def test_apply_instructions_are_quoted() -> None:
    instructions = _kind(
        posting_conditions(_job("conditions_residency_en.txt")), ConditionKind.INSTRUCTION
    )
    assert any("send your cv in english" in c.source.casefold() for c in instructions)


def test_deadline_is_read_as_a_date() -> None:
    job = JobPosting(
        id="J0902",
        title="Analyst",
        company="Estudio Neutro",
        description="Fecha límite: 2026-09-01\nPostula con tu CV.",
    )
    deadlines = _kind(posting_conditions(job), ConditionKind.DEADLINE)
    assert deadlines[0].when == date(2026, 9, 1)


def test_structured_fields_are_read_when_text_is_silent() -> None:
    job = JobPosting(
        id="J0903",
        title="Analyst",
        company="Estudio Neutro",
        description="Analyst for our team.",
        remote_type="hybrid",
        employment_type="freelance",
        seniority="senior",
    )
    conditions = posting_conditions(job)
    assert [c.modality for c in _kind(conditions, ConditionKind.MODALITY)] == ["hybrid"]
    assert [c.contract for c in _kind(conditions, ConditionKind.CONTRACT)] == ["contractor"]
    assert [c.detail for c in _kind(conditions, ConditionKind.SENIORITY)] == ["senior"]


def test_a_plain_posting_has_no_invented_conditions() -> None:
    job = JobPosting(
        id="J0904",
        title="Analyst",
        company="Estudio Neutro",
        description="We keep the monthly close on time and talk to every team.",
    )
    assert posting_conditions(job) == []


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("C1", "C1"),
        ("b2+", "B2"),
        ("fully-fluent", "C1"),
        ("conversational", "B1"),
        ("nativo", "C2"),
        ("intermedio alto", "B2"),
        ("básico", "A2"),
        ("whatever", None),
    ],
)
def test_level_from_text(text: str, level: str | None) -> None:
    assert level_from_text(text) == level


def test_cefr_rank_orders_levels() -> None:
    assert cefr_rank("A1") < cefr_rank("B1") < cefr_rank("C2")
    assert cefr_rank("zz") is None


def test_language_code_reads_spanish_and_english_names() -> None:
    assert language_code("Inglés") == "en"
    assert language_code("english") == "en"
    assert language_code("portugués") == "pt"
    assert language_code("en") == "en"
    assert language_code("cocina") is None


def test_business_to_business_sales_is_not_a_contract() -> None:
    """'B2B' names a market, not a contractor arrangement."""
    job = JobPosting(
        id="J0905",
        title="Ejecutivo Comercial",
        company="Estudio Neutro",
        description="Requisitos:\n- Experiencia en ventas B2B.\n",
    )
    assert not _kind(posting_conditions(job), ConditionKind.CONTRACT)
