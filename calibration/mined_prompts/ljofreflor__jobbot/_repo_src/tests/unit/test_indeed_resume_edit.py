"""Unit tests for Indeed resume edit helpers (no live browser)."""

from jobbot.adapters.indeed.resume_edit import (
    _MONTHS_ES,
    _experience_description,
    _parse_ym,
    parse_resume_page_text,
)
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import sample_profile_dict


def test_parse_ym_and_months() -> None:
    assert _parse_ym("2025-10") == (2025, 10)
    assert _MONTHS_ES[10] == "Octubre"


def test_experience_description_includes_achievements() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    text = _experience_description(candidate.experience[0])
    assert "Share of Wallet" in text
    assert text.startswith("•") or "Customer analytics" in text


def test_parse_resume_page_text_summary() -> None:
    body = """
Editar CV
Resumen
Senior Data Scientist con experiencia en fintech.
Datos personales
Destaca datos personales
Experiencia laboral
Habilidades
Python
SQL
Certificaciones y licencias
"""
    parsed = parse_resume_page_text(body)
    assert parsed["summary"] and "fintech" in str(parsed["summary"])
    assert "Python" in parsed["skills"]  # type: ignore[operator]
