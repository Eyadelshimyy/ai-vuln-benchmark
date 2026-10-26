"""One test per real-world CV layout: every layout must yield experience entries.

A PDF carries no structure, so each CV design puts the company, the role and the
dates in a different place. These fixtures mirror layouts found in the wild; the
data is invented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jobbot.profile.importer_pdf import parse_cv_text


def _experiences(project_root: Path, layout: str) -> list[dict[str, Any]]:
    path = project_root / "tests" / "fixtures" / "cv_layouts" / f"{layout}.txt"
    result = parse_cv_text(path.read_text(encoding="utf-8"), path)
    return list(result.data["experience"])


def test_role_and_dates_on_one_line_with_company_above(project_root: Path) -> None:
    """Company on its own line; the role line carries right-aligned dates."""
    experiences = _experiences(project_root, "role_dates_right")

    assert len(experiences) == 2

    first = experiences[0]
    assert first["company"] == "Auditora Consultores - Forensic"
    assert first["title"] == "Investigaciones y análisis"
    assert first["start_date"] == "2022-09"
    assert first["end_date"] == "2023-10"
    assert len(first["achievements"]) == 2

    assert experiences[1]["company"] == "Contadores del Sur - Auditoría"
    assert experiences[1]["title"] == "Asistente de auditoría"


def test_company_on_the_line_below_the_role(project_root: Path) -> None:
    """Role plus dates first, company and location underneath, prose instead of bullets."""
    experiences = _experiences(project_root, "company_below")

    assert len(experiences) == 2

    first = experiences[0]
    assert first["title"] == "Analista Senior de Control de Gestión"
    assert first["company"] == "Banco del Sur Cobranza"
    assert first["location"] == "Santiago, Chile"
    assert first["start_date"] == "2023-01"
    assert first["current"] is True
    assert len(first["achievements"]) == 2
    assert first["achievements"][0]["text"].startswith("Lidero la gestión")

    assert experiences[1]["title"] == "Segment Manager Analítico"
    assert experiences[1]["company"] == "Caja de Compensación del Norte"
    assert experiences[1]["end_date"] == "2022-12"


def test_dates_in_their_own_block_above_company_and_role(project_root: Path) -> None:
    """English résumé: '2022 to / present' wrapped, then company, department, role."""
    experiences = _experiences(project_root, "date_block_english")

    assert len(experiences) == 2

    first = experiences[0]
    assert first["company"] == "Northbank"
    assert first["location"] == "Toronto, Canada"
    assert first["title"] == "Manager, Provisioning and Data Science"
    assert first["start_date"] == "2022-01"
    assert first["current"] is True
    assert first["description"] is not None
    assert "Model and Data Risk Management" in first["description"]
    assert len(first["achievements"]) == 2

    second = experiences[1]
    assert second["company"] == "Southbank"
    assert second["title"] == "Director, Modelling and Provisions"
    assert second["start_date"] == "2020-01"
    assert second["end_date"] == "2022-12"


def test_letter_spaced_design_with_company_and_dates_together(project_root: Path) -> None:
    """Two-column design CV: letter-spaced headings and roles, 'COMPANY | 2023 – 2026'."""
    experiences = _experiences(project_root, "letter_spaced_design")

    assert len(experiences) == 2

    first = experiences[0]
    assert first["title"] == "ANALISTA DE CONTROL DE GESTIÓN SENIOR"
    assert first["company"] == "BANCO DEL SUR COBRANZA"
    assert first["start_date"] == "2023-01"
    assert first["end_date"] == "2026-12"
    assert len(first["achievements"]) == 2

    assert experiences[1]["company"] == "CAJA DE COMPENSACIÓN DEL NORTE"
    assert experiences[1]["title"] == "SEGMENT MANAGER ANALÍTICO"


@pytest.mark.parametrize(
    "layout",
    ["role_dates_right", "company_below", "date_block_english", "letter_spaced_design"],
)
def test_every_layout_yields_experience_and_dates(project_root: Path, layout: str) -> None:
    """The regression that started this: some layouts silently produced zero roles."""
    experiences = _experiences(project_root, layout)

    assert experiences
    for entry in experiences:
        assert entry["company"] != "Unknown"
        assert entry["title"] != "Unknown"
        assert entry["start_date"]


def test_name_is_found_when_the_document_starts_with_a_title() -> None:
    """Some résumés head the page with 'Resume', so the name sits inside the section."""
    text = (
        "Summary\n"
        "Jorge Prado Vega\n"
        "+56 9 00000002\n"
        "jorge.prado@example.com\n"
        "Professional with 15 years of experience in risk modelling.\n"
    )

    personal = parse_cv_text(text, Path("titled.pdf")).data["personal"]

    assert personal["name"] == "Jorge Prado Vega"
    assert personal["email"] == "jorge.prado@example.com"


def test_education_written_as_institution_dash_degree_then_dates() -> None:
    """'Universidad X - Título  Ene 2015 - Dic 2020': one line, no comma to split on."""
    text = (
        "Marco Vidal Ruiz\n"
        "EDUCACIÓN\n"
        "Universidad del Valle - Contador Público y Auditor En 2015 - Dic 2020\n"
        "Universidad del Norte – Magíster en Data Science En 2023 - Dic 2025\n"
    )

    result = parse_cv_text(text, Path("education.pdf"))
    education = result.data["education"]

    assert [item["institution"] for item in education] == [
        "Universidad del Valle",
        "Universidad del Norte",
    ]
    assert education[0]["degree"] == "Contador Público y Auditor"
    assert education[0]["start_date"] == "2015-01"
    assert education[0]["end_date"] == "2020-12"
    assert education[1]["degree"] == "Magíster en Data Science"


def test_roles_stranded_outside_the_experience_section_are_reported() -> None:
    """Two-column PDFs interleave columns: roles land under the wrong heading."""
    text = (
        "Susana Prado Vega\n"
        "EXPERIENCIA\n"
        "Analista de Control de Gestión\n"
        "Banco del Sur | 2023 - 2026\n"
        "• Diseñé paneles de control para el centro de contacto.\n"
        "HABILIDADES\n"
        "SQL, Python, Power BI\n"
        "Jefa de Operaciones\n"
        "Seguros del Norte | 2019 - 2022\n"
        "• Coordiné la mesa de ayuda de la sucursal.\n"
    )

    result = parse_cv_text(text, Path("two_columns.pdf"))

    assert len(result.data["experience"]) == 1
    assert any("outside the experience section" in warning for warning in result.warnings)


def test_a_range_whose_two_months_share_one_year() -> None:
    """'julio – agosto 2026': the CV writes the year once, at the end of the range."""
    text = (
        "Elena Prado Vega\n"
        "EXPERIENCIA LABORAL\n"
        "Consultoría en terreno                              julio – agosto 2026\n"
        "Depto de Emergencias, Instituto del Norte, Chile.\n"
        "Apoyo a la respuesta epidemiológica regional ante emergencias.\n"
    )

    experiences = parse_cv_text(text, Path("shared_year.pdf")).data["experience"]

    assert len(experiences) == 1
    assert experiences[0]["title"] == "Consultoría en terreno"
    assert experiences[0]["start_date"] == "2026-07"
    assert experiences[0]["end_date"] == "2026-08"


def test_a_role_dated_with_a_single_year() -> None:
    """'Docente   2013' is a role that lasted one year, not a line without dates."""
    text = (
        "Elena Prado Vega\n"
        "EXPERIENCIA LABORAL\n"
        "Docente                                                          2013\n"
        "Carrera de Medicina, Universidad del Sur, Santiago, Chile.\n"
        "Responsable del curso teórico y práctico de la asignatura.\n"
    )

    experiences = parse_cv_text(text, Path("single_year.pdf")).data["experience"]

    assert len(experiences) == 1
    assert experiences[0]["title"] == "Docente"
    assert experiences[0]["start_date"] == "2013-01"
    assert experiences[0]["end_date"] == "2013-12"
    assert experiences[0]["company"] == "Carrera de Medicina, Universidad del Sur"
    assert experiences[0]["location"] == "Santiago, Chile"


def test_an_employer_that_wraps_and_ends_in_a_full_stop() -> None:
    """A ministry's full name runs past any header-like line and over two of them."""
    text = (
        "Elena Prado Vega\n"
        "EXPERIENCIA LABORAL\n"
        "Referente Territorial                                     2021 - 2022\n"
        "Unidad de Diagnóstico, Programa Nacional de Laboratorios, Gabinete del Director,\n"
        "Instituto del Norte, Chile.\n"
        "Coordinación de la red pública y privada de laboratorios clínicos.\n"
    )

    experiences = parse_cv_text(text, Path("wrapped_employer.pdf")).data["experience"]

    assert len(experiences) == 1
    assert experiences[0]["company"] == (
        "Unidad de Diagnóstico, Programa Nacional de Laboratorios, "
        "Gabinete del Director, Instituto del Norte"
    )
    assert experiences[0]["location"] == "Chile"
    assert [item["text"] for item in experiences[0]["achievements"]] == [
        "Coordinación de la red pública y privada de laboratorios clínicos."
    ]


def test_a_duty_that_names_a_country_is_not_read_as_the_employer() -> None:
    """'… mesas de ayuda regionales en Chile.' is a duty, not the employer's seat."""
    text = (
        "Elena Prado Vega\n"
        "EXPERIENCIA LABORAL\n"
        "Analista de Riesgo                                        2021 - 2022\n"
        "Aseguradora del Norte, Santiago, Chile.\n"
        "Coordiné el despliegue de once mesas de ayuda regionales en Chile.\n"
    )

    experiences = parse_cv_text(text, Path("duty_with_country.pdf")).data["experience"]

    assert len(experiences) == 1
    assert experiences[0]["company"] == "Aseguradora del Norte"
    assert experiences[0]["location"] == "Santiago, Chile"
    assert [item["text"] for item in experiences[0]["achievements"]] == [
        "Coordiné el despliegue de once mesas de ayuda regionales en Chile."
    ]


def test_every_role_without_a_company_line_is_reported() -> None:
    """'Unknown' must not travel down the CV as if it were a real employer."""
    text = (
        "Elena Prado Vega\n"
        "EXPERIENCIA LABORAL\n"
        "Analista de Riesgo                                        2021 - 2022\n"
        "Coordiné el despliegue de once mesas de ayuda regionales.\n"
        "Asistente de Operaciones                                  2019 - 2020\n"
        "Apoyé la mesa de ayuda de la sucursal durante el turno de tarde.\n"
    )

    result = parse_cv_text(text, Path("missing_company.pdf"))

    assert [entry["company"] for entry in result.data["experience"]] == ["Unknown", "Unknown"]
    missing = [w for w in result.warnings if "no company line near the dates" in w]
    assert len(missing) == 2


def test_text_without_word_spaces_is_reported() -> None:
    """Some PDFs carry no space glyphs: the text is usable but must be flagged."""
    text = (
        "Susana Prado Vega\n"
        "Ingeniera Industrial\n"
        "Perfil profesional\n"
        "Ingenieraindustrialconmásdenueveañosdeexperienciaencontroldegestiónyanalítica\n"
        "declientesenbanca,segurosyserviciosfinancierosparaladecisióndenegocio.\n"
    )

    result = parse_cv_text(text, Path("memo.pdf"))

    assert any("without spaces" in warning for warning in result.warnings)
