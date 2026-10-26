"""Job-specific achievement selection tests."""

from pathlib import Path

from jobbot.cv.selection import select_for_job
from jobbot.jobs.parsing import parse_job_text
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import sample_profile_dict


def test_select_for_job_prefers_overlapping_achievements(project_root: Path) -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    job = parse_job_text(
        (project_root / "tests/fixtures/jobs/senior_ds_retail.txt").read_text(encoding="utf-8"),
        job_id="J0001",
    )
    selection = select_for_job(candidate, job)
    assert selection.job_id == "J0001"
    ids = {a.id for a in selection.selected_achievements}
    assert "meli-share-of-wallet" in ids
    selected = next(a for a in selection.selected_achievements if a.id == "meli-share-of-wallet")
    assert selected.reason


def _wide_profile() -> Candidate:
    """Enough roles that a terse LinkedIn post must not wipe the CV."""
    experiences = []
    for i in range(1, 7):
        experiences.append(
            {
                "id": f"role-{i}",
                "company": f"Empresa {i}",
                "title": f"Cargo {i}",
                "location": "Santiago, Chile",
                "start_date": f"{2015 + i}-01",
                "end_date": f"{2015 + i}-12",
                "current": False,
                "description": f"Trabajo en dominio genérico número {i}.",
                "achievements": [
                    {
                        "id": f"ach-{i}a",
                        "text": f"Logré un resultado medible en el rol {i}.",
                        "tags": [],
                        "metrics": {"percent": 10},
                    },
                    {
                        "id": f"ach-{i}b",
                        "text": f"Coordiné un equipo en el rol {i}.",
                        "tags": [],
                        "metrics": {},
                    },
                ],
            }
        )
    return Candidate.model_validate(
        {
            "personal": {
                "name": "Ana Ejemplo",
                "headline": "Profesional",
                "email": "ana@example.com",
            },
            "summary": "Perfil amplio.",
            "experience": experiences,
            "education": [],
            "skills": {"other": ["Comunicación", "Excel", "Gestión"]},
            "publications": [],
        }
    )


def test_a_short_linkedin_post_keeps_a_usable_cv_floor() -> None:
    """Email-apply JDs are often 2–4 lines; they must not collapse the derived CV."""
    candidate = _wide_profile()
    job = parse_job_text(
        "Title: Profesional\nCompany: Acme\n"
        "Enviar CV a seleccion@empresa.cl. Vacante abierta en Santiago.\n",
        job_id="J0090",
    )
    assert len(job.description or "") < 400

    selection = select_for_job(candidate, job)

    assert len(selection.experience_ids) >= 4
    assert len(selection.selected_achievements) >= 4
    reasons = " ".join(r for a in selection.selected_achievements for r in a.reason)
    assert "floor" in reasons.casefold() or "fallback" in reasons.casefold()
