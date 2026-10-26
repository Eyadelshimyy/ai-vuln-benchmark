"""Empleos Públicos (Chile) — public concurso board; fixture-first."""

from jobbot.adapters.empleospublicos.jobs import (
    EmpleosPublicosJobSource,
    canonical_ficha_url,
    job_from_ficha_html,
    jobs_from_search_payload,
)

__all__ = [
    "EmpleosPublicosJobSource",
    "canonical_ficha_url",
    "job_from_ficha_html",
    "jobs_from_search_payload",
]
