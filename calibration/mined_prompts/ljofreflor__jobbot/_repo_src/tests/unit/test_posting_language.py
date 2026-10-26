"""Deterministic Spanish/English detection of a posting's own text."""

from __future__ import annotations

from jobbot.jobs.language import detect_language, posting_language
from jobbot.models.job import JobPosting

ENGLISH = (
    "Senior Data Analyst\n"
    "You will drive the marketing strategy by building models for our growth team.\n"
    "- Python (potential to develop)\n"
    "- SQL (potential to develop)\n"
)

SPANISH = (
    "Buscamos una Enfermera Clínica para la unidad de paciente crítico.\n"
    "Requisitos: título de enfermera y experiencia en ventilación mecánica con "
    "pacientes de alta complejidad.\n"
)


def test_english_text_is_english() -> None:
    assert detect_language(ENGLISH) == "en"


def test_spanish_text_is_spanish() -> None:
    assert detect_language(SPANISH) == "es"


def test_labels_added_by_jobbot_do_not_flip_an_english_posting() -> None:
    """A stored summary may carry our own Spanish labels ('Modalidad: …')."""
    text = ENGLISH + "Modalidad: Remote / anywhere\nCompensación: a convenir\n"
    assert detect_language(text) == "en"


def test_too_little_text_is_unclear() -> None:
    assert detect_language("Python, SQL, BigQuery") is None
    assert detect_language("") is None


def test_an_even_mix_is_unclear() -> None:
    assert detect_language("the data and the models de la empresa y el equipo") is None


def test_posting_language_reads_title_description_and_requirements() -> None:
    job = JobPosting(
        id="J0001",
        title="Editor de Contenidos",
        company="Medio Regional",
        requirements=["Experiencia en edición digital y redacción de notas para la web."],
    )
    assert posting_language(job) == "es"
