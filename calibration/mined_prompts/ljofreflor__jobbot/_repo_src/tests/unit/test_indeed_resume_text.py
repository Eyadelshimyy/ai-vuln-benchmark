"""Regression: single-character skills were dropped from the resume text parse.

Indeed listed "R" and "c" as skills, but the length filter skipped them, so every
snapshot missed them and `cv propagate` kept planning an ADD that already existed.
"""

from __future__ import annotations

from jobbot.adapters.indeed.resume_edit import parse_resume_page_text

BODY = """
Resumen
Científico de datos con foco en riesgo.
Datos personales
Habilidades
Python
R
c
SQL
Agregar habilidad
Certificaciones
"""


def test_single_letter_skills_survive_the_text_parse() -> None:
    parsed = parse_resume_page_text(BODY)
    assert parsed["skills"] == ["Python", "R", "c", "SQL"]
