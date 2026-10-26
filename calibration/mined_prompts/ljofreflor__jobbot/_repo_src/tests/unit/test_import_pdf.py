"""Tests for the PDF CV importer (Word-exported layout)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS
from jobbot.models.candidate import Candidate
from jobbot.profile.importer_common import write_generated_profile
from jobbot.profile.importer_pdf import PdfImportError, import_pdf_cv, parse_cv_text
from jobbot.profile.loader import load_profile
from jobbot.profile.validator import validate_candidate

# Word exports its bullet as a Symbol glyph in the Unicode private use area.
WORD_BULLET = "\uf0b7"


def test_personal_block_from_header_lines(sample_cv_pdf: Path) -> None:
    personal = import_pdf_cv(sample_cv_pdf).data["personal"]

    assert personal["name"] == "Marta Soto Vera"
    assert "SCRUM MASTER" in personal["headline"]
    assert personal["email"] == "marta.soto@example.com"
    assert personal["phone"] == "+56 9 00000000"
    assert personal["linkedin"] == "https://www.linkedin.com/in/marta-soto-vera/"
    assert personal["city"] == "Santiago"
    assert personal["country"] == "Chile"


def test_summary_is_the_paragraph_before_the_first_section(sample_cv_pdf: Path) -> None:
    summary = import_pdf_cv(sample_cv_pdf).data["summary"]

    assert summary is not None
    assert "sector minero" in summary
    assert "COMPETENCIAS" not in summary


def test_experience_entries_keep_company_dates_and_bullets(sample_cv_pdf: Path) -> None:
    experiences = import_pdf_cv(sample_cv_pdf).data["experience"]

    assert len(experiences) == 4

    first = experiences[0]
    assert first["company"] == "NORTE CONSULTORES LTDA"
    assert first["location"] == "SANTIAGO DE CHILE"
    assert first["title"] == "Coordinadora Administrativa"
    assert first["start_date"] == "2024-10"
    assert first["current"] is True
    assert first.get("end_date") is None
    assert first["description"] is not None
    assert first["description"].startswith("Supervisar las actividades diarias")
    assert "recepción de materiales" in first["description"]
    assert [a["text"] for a in first["achievements"]] == [
        "Reportes: informe mensual de ubicaciones disponibles.",
        "Gestión de personal: control de asistencia y turnos.",
    ]

    second = experiences[1]
    assert second["company"] == "GEODETEC INGENIERIA SPA"
    assert second["start_date"] == "2022-07"
    assert second["end_date"] == "2023-11"
    assert second["current"] is False


def test_second_role_inherits_the_employer_of_the_previous_entry(sample_cv_pdf: Path) -> None:
    """PDVSA-style CVs list one company header and two roles under it."""
    experiences = import_pdf_cv(sample_cv_pdf).data["experience"]

    assert experiences[2]["company"] == "PETRÓLEOS DEL SUR S.A"
    assert experiences[2]["start_date"] == "2010-09"
    assert experiences[2]["end_date"] == "2016-06"

    assert experiences[3]["company"] == "PETRÓLEOS DEL SUR S.A"
    assert experiences[3]["title"] == "Ingeniera de Terreno"


def test_broken_end_date_warns_instead_of_inventing_a_year(sample_cv_pdf: Path) -> None:
    result = import_pdf_cv(sample_cv_pdf)

    broken = result.data["experience"][3]
    assert broken["start_date"] == "2008-09"
    assert broken["end_date"] == "2008-09"
    assert any("end_date" in warning for warning in result.warnings)


def test_education_reads_dates_inside_the_line(sample_cv_pdf: Path) -> None:
    education = import_pdf_cv(sample_cv_pdf).data["education"]

    assert len(education) == 2

    degree = education[0]
    assert "Universidad del Litoral" in degree["institution"]
    assert degree["degree"] == "Ingeniera Geofísica"
    assert degree["start_date"] == "2000-09"
    assert degree["end_date"] == "2007-10"

    diploma = education[1]
    assert "Diplomado" in diploma["degree"]
    assert diploma["start_date"] == "2023-01"
    assert diploma["end_date"] == "2024-12"


def test_skills_come_from_letter_spaced_sections(sample_cv_pdf: Path) -> None:
    skills = import_pdf_cv(sample_cv_pdf).data["skills"]
    flat = [skill for group in skills.values() for skill in group]

    assert "Scrum" in flat
    assert "Gestión de Proyectos" in flat
    assert "AutoCAD" in flat
    assert "Oasis Montaj" in flat
    assert "Logros" not in flat


def test_generated_profile_validates(sample_cv_pdf: Path, tmp_path: Path) -> None:
    result = import_pdf_cv(sample_cv_pdf)
    out = tmp_path / "profile.generated.yaml"

    write_generated_profile(result, out, command="profile import-pdf")

    candidate = load_profile(out)
    assert isinstance(candidate, Candidate)
    assert validate_candidate(candidate).ok
    assert "profile import-pdf" in out.read_text(encoding="utf-8")


def test_word_bullets_do_not_leak_into_skill_names() -> None:
    """Two-column skill blocks are split by the bullet, and the column labels are dropped."""
    text = (
        "Ana Prueba Ramos\n"
        "INGENIERA DE PROYECTOS\n"
        "C O M P E T E N C I A S    P R O F E S I O N A L E S\n"
        f"{WORD_BULLET} Interpersonales {WORD_BULLET} Técnicas\n"
        f"{WORD_BULLET} Comunicación Efectiva\n"
        f"{WORD_BULLET} Scrum\n"
    )

    skills = parse_cv_text(text, Path("memo.pdf")).data["skills"]
    flat = [skill for group in skills.values() for skill in group]

    assert flat == ["Comunicación Efectiva", "Scrum"]


def test_word_bullets_start_achievements() -> None:
    text = (
        "Ana Prueba Ramos\n"
        "INGENIERA DE PROYECTOS\n"
        "E X P E R I E N C I A   P R O F E S I O N A L\n"
        "ACME SPA - SANTIAGO DE CHILE\n"
        "01/2020 –12/2021 Ingeniera de Proyectos\n"
        "Logros\n"
        f"{WORD_BULLET} Primer logro medible.\n"
        f"{WORD_BULLET} Segundo logro medible.\n"
    )

    experience = parse_cv_text(text, Path("memo.pdf")).data["experience"][0]

    assert [a["text"] for a in experience["achievements"]] == [
        "Primer logro medible.",
        "Segundo logro medible.",
    ]


def test_pdf_without_text_layer_is_rejected(scanned_cv_pdf: Path) -> None:
    with pytest.raises(PdfImportError, match="text"):
        import_pdf_cv(scanned_cv_pdf)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PdfImportError):
        import_pdf_cv(tmp_path / "nope.pdf")


def _run(argv: list[str]) -> int:
    from jobbot.cli import run_cli

    return run_cli(argv, standalone_mode=False)


def test_cli_import_pdf_writes_the_generated_profile(
    sample_cv_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(["profile", "import-pdf", str(sample_cv_pdf)]) == SUCCESS

    generated = tmp_path / "data" / "profile.generated.yaml"
    assert generated.is_file()
    assert "Marta Soto Vera" in generated.read_text(encoding="utf-8")
    assert not (tmp_path / "data" / "profile.yaml").exists()


def test_cli_import_pdf_reports_a_scanned_pdf(
    scanned_cv_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run(["profile", "import-pdf", str(scanned_cv_pdf)]) == GENERIC_FAILURE
    assert not (tmp_path / "data" / "profile.generated.yaml").exists()
