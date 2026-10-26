"""Get on Board permanent profile maintainer tests."""

from pathlib import Path

from jobbot.adapters.getonboard.draft import (
    EXPERIENCE_MAX,
    build_permanent_profile_fields,
    draft_getonboard_fields,
    load_permanent_profile,
    save_permanent_profile,
)
from jobbot.applications.manager import prepare_application_package
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from tests.fixtures.profile import sample_profile_dict


def test_permanent_profile_within_limits_and_no_invention() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    fields = build_permanent_profile_fields(candidate)
    assert len(fields.experiencia_y_perfil) <= EXPERIENCE_MAX
    assert "Mercado Libre" in fields.experiencia_y_perfil or "Data Scientist" in (
        fields.experiencia_y_perfil + fields.headline
    )
    assert "…" not in fields.experiencia_y_perfil
    # Sample fixture includes Mercado Libre experience — must survive GoB length fit.
    assert "Mercado Libre" in fields.experiencia_y_perfil or "Data Scientist" in fields.headline


def test_summary_is_not_cut_mid_sentence_when_there_is_room() -> None:
    """A long summary was hard-cut at 549 chars, so GoB showed a dangling clause."""
    data = sample_profile_dict()
    data["summary"] = (
        "Senior Data Scientist con experiencia en fintech, retail y sector público. "
        + "Construyo modelos predictivos y plataformas de datos usadas a escala regional. "
        * 6
        + "Combino profundidad estadística con liderazgo técnico."
    )
    candidate = Candidate.model_validate(data)

    fields = build_permanent_profile_fields(candidate)
    summary_paragraph = fields.experiencia_y_perfil.split("\n\n")[0]

    assert len(fields.experiencia_y_perfil) <= EXPERIENCE_MAX
    assert summary_paragraph.endswith("."), summary_paragraph[-60:]
    # The whole summary fits under the cap, so nothing should have been dropped.
    assert summary_paragraph == data["summary"].strip()


def test_a_summary_too_long_for_the_cap_ends_on_a_sentence() -> None:
    data = sample_profile_dict()
    data["summary"] = "Lidero equipos de data science y productos de datos medibles. " * 40
    candidate = Candidate.model_validate(data)

    fields = build_permanent_profile_fields(candidate)
    summary_paragraph = fields.experiencia_y_perfil.split("\n\n")[0]

    assert len(fields.experiencia_y_perfil) <= EXPERIENCE_MAX
    assert summary_paragraph.endswith(".")
    assert "," not in summary_paragraph[-2:]


def test_save_and_load_permanent_profile(tmp_path: Path) -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    fields = build_permanent_profile_fields(candidate)
    md = save_permanent_profile(fields, tmp_path)
    assert md.is_file()
    assert "perfil permanente" in md.read_text(encoding="utf-8").casefold()
    loaded = load_permanent_profile(tmp_path)
    assert loaded is not None
    assert loaded.experiencia_y_perfil == fields.experiencia_y_perfil


def test_prepare_reuses_permanent_profile(tmp_path: Path) -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    permanent = build_permanent_profile_fields(candidate)
    # Mark permanent with a unique substring
    from jobbot.adapters.getonboard.draft import PermanentProfileFields

    custom = PermanentProfileFields(
        experiencia_y_perfil=permanent.experiencia_y_perfil + "\n\nPERMANENT_MARKER.",
        formacion_academica=permanent.formacion_academica,
        headline=permanent.headline,
        skills=permanent.skills,
    )
    save_permanent_profile(custom, tmp_path)
    job = JobPosting(
        id="J0091",
        title="Applied Scientist",
        company="NeuralWorks",
        ats_kind="getonboard",
        ats_url="https://www.getonbrd.com/jobs/x",
        description="Requires Kubernetes mastery.",
    )
    app_dir = prepare_application_package(job, tmp_path, candidate=candidate)
    text = (app_dir / "getonboard_es.md").read_text(encoding="utf-8")
    assert "PERMANENT_MARKER" in text
    assert "Kubernetes" not in text


def test_draft_getonboard_fields_dict_shape() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    fields = draft_getonboard_fields(candidate)
    assert "experiencia_y_perfil" in fields
    assert "formacion_academica" in fields
