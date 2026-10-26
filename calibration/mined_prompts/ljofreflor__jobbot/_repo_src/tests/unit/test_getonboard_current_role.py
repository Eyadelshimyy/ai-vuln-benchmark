"""The Get on Board draft must say which role is current, as profile.yaml says it."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from jobbot.adapters.getonboard.draft import (
    EDUCATION_MAX,
    EXPERIENCE_MAX,
    build_permanent_profile_fields,
    draft_getonboard_fields,
    fields_from_seed_text,
    save_permanent_profile,
)
from jobbot.applications.manager import prepare_application_package
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.nlp.refine import refine_permanent_profile

CURRENT_HEAD = "Jefa de Operaciones en Logística Austral (2026-08–actualidad)."
ENDED_HEAD = "Analista Senior en Distribuidora Costera (2025-10–2026-07)."
STALE_HEAD = "Analista Senior en Distribuidora Costera (2025-10–actualidad)."


def _role(
    rid: str,
    company: str,
    title: str,
    start: str,
    end: str | None,
    achievement: str,
) -> dict[str, Any]:
    return {
        "id": rid,
        "company": company,
        "title": title,
        "location": "Concepción, Chile",
        "start_date": start,
        "end_date": end,
        "current": end is None,
        "description": None,
        "achievements": [{"id": f"{rid}-a1", "text": achievement, "tags": [], "metrics": {}}],
    }


CURRENT = _role(
    "austral-jefa",
    "Logística Austral",
    "Jefa de Operaciones",
    "2026-08",
    None,
    "Coordino tres centros de distribución y sus equipos de despacho.",
)
ENDED = _role(
    "costera-analista",
    "Distribuidora Costera",
    "Analista Senior",
    "2025-10",
    "2026-07",
    "Rediseñé la planificación semanal de rutas y reduje los retrasos de entrega.",
)
OLDER = _role(
    "valle-analista",
    "Comercial del Valle",
    "Analista de Inventario",
    "2022-04",
    "2025-05",
    "Ordené el maestro de productos de cuatro bodegas regionales.",
)


def _profile(experience: list[dict[str, Any]], *, summary: str) -> dict[str, Any]:
    return {
        "personal": {
            "name": "Ana Ejemplo Prueba",
            "headline": "Jefa de Operaciones | Logística",
            "city": "Concepción",
            "country": "Chile",
            "email": "ana.ejemplo@example.com",
        },
        "summary": summary,
        "specialties": ["Logística"],
        "experience": deepcopy(experience),
        "education": [
            {
                "id": "ing-industrial",
                "institution": "Universidad del Sur",
                "degree": "Ingeniería Civil Industrial",
                "start_date": "2012-03",
                "end_date": "2018-12",
            }
        ],
        "skills": {"operaciones": ["Planificación", "Inventario", "Excel", "Ruteo"]},
        "publications": [],
    }


LONG_SUMMARY = (
    "Profesional de operaciones con foco en distribución y mejora continua. "
    + "Lidero equipos de terreno y planifico la operación con datos propios. " * 30
)


def _today() -> Candidate:
    return Candidate.model_validate(_profile([CURRENT, ENDED, OLDER], summary=LONG_SUMMARY))


def _draft_from_before_the_new_job() -> Candidate:
    """The profile as it was when the ended role was still the current one."""
    was_current = {**ENDED, "end_date": None, "current": True}
    return Candidate.model_validate(_profile([was_current, OLDER], summary=LONG_SUMMARY))


def test_cold_draft_marks_only_the_current_role_as_current() -> None:
    text = build_permanent_profile_fields(_today()).experiencia_y_perfil
    assert CURRENT_HEAD in text
    assert ENDED_HEAD in text
    assert text.count("actualidad") == 1


def test_cold_draft_keeps_the_current_role_whatever_the_yaml_order() -> None:
    candidate = Candidate.model_validate(
        _profile([ENDED, OLDER, CURRENT], summary="Profesional de operaciones.")
    )
    text = build_permanent_profile_fields(candidate).experiencia_y_perfil
    assert CURRENT_HEAD in text
    assert text.index(CURRENT_HEAD) < text.index(ENDED_HEAD)
    assert "Comercial del Valle" not in text


def test_refine_lets_the_profile_win_over_a_stale_previous_draft() -> None:
    """Regression: the refine kept 'actualidad' on an ended role and lost the current one."""
    previous = build_permanent_profile_fields(_draft_from_before_the_new_job())
    assert STALE_HEAD in previous.experiencia_y_perfil

    result = refine_permanent_profile(_today(), previous)
    text = result.fields.experiencia_y_perfil

    assert CURRENT_HEAD in text
    assert ENDED_HEAD in text
    assert STALE_HEAD not in text
    assert text.count("actualidad") == 1
    assert text.index(CURRENT_HEAD) < text.index(ENDED_HEAD)
    assert len(text) <= EXPERIENCE_MAX


def test_refine_keeps_the_body_a_previous_draft_wrote_for_a_role() -> None:
    previous = build_permanent_profile_fields(_draft_from_before_the_new_job())
    edited_body = "Armé el tablero de despacho que usa hoy la gerencia."
    edited = previous.experiencia_y_perfil.replace(
        "Rediseñé la planificación semanal de rutas y reduje los retrasos de entrega.",
        edited_body,
    )
    previous = type(previous)(
        experiencia_y_perfil=edited,
        formacion_academica=previous.formacion_academica,
        headline=previous.headline,
        skills=previous.skills,
    )

    text = refine_permanent_profile(_today(), previous).fields.experiencia_y_perfil

    assert f"{ENDED_HEAD} {edited_body}" in text
    assert text.count("Distribuidora Costera (") == 1


def test_refining_the_fixed_draft_again_is_stable() -> None:
    previous = build_permanent_profile_fields(_draft_from_before_the_new_job())
    first = refine_permanent_profile(_today(), previous)
    second = refine_permanent_profile(_today(), first.fields)
    assert first.fields.experiencia_y_perfil == second.fields.experiencia_y_perfil


def test_application_fields_correct_a_stale_stored_permanent_profile() -> None:
    stale = build_permanent_profile_fields(_draft_from_before_the_new_job())

    text = draft_getonboard_fields(_today(), permanent=stale)["experiencia_y_perfil"]

    assert CURRENT_HEAD in text
    assert ENDED_HEAD in text
    assert STALE_HEAD not in text
    assert text.count("actualidad") == 1


def test_application_prepare_sheet_states_the_current_role(tmp_path: Path) -> None:
    """Regression: the per-job sheet pasted a stored draft saved before the new job."""
    stale = build_permanent_profile_fields(_draft_from_before_the_new_job())
    save_permanent_profile(stale, tmp_path)
    job = JobPosting(
        id="J0001",
        title="Coordinadora de Operaciones",
        company="Operaciones Demo",
        ats_kind="getonboard",
        ats_url="https://www.getonbrd.com/jobs/x",
        description="Coordinación de centros de distribución.",
    )

    app_dir = prepare_application_package(job, tmp_path, candidate=_today())
    sheet = (app_dir / "getonboard_es.md").read_text(encoding="utf-8")

    assert CURRENT_HEAD in sheet
    assert ENDED_HEAD in sheet
    assert STALE_HEAD not in sheet
    assert sheet.count("actualidad") == 1
    assert sheet.index(CURRENT_HEAD) < sheet.index(ENDED_HEAD)


def test_seed_without_education_falls_back_on_a_word_boundary() -> None:
    data = _profile([CURRENT], summary="Profesional de operaciones.")
    data["education"][0]["details"] = "Formación en logística y planificación. " * 80
    candidate = Candidate.model_validate(data)

    fields = fields_from_seed_text(
        experiencia="Profesional de operaciones.", formacion="", candidate=candidate
    )

    education = fields.formacion_academica
    assert len(education) <= EDUCATION_MAX
    assert education.split()[-1] in {"logística", "y", "planificación.", "Formación", "en"}
