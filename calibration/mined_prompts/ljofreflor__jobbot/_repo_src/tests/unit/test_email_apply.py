"""Email apply extraction, LinkedIn sweep, and Gmail HITL compose."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from jobbot.adapters.ats.apply import build_apply_plan
from jobbot.adapters.ats.email_apply import (
    build_email_draft,
    gmail_compose_url,
    open_gmail_compose,
    resolve_cv_path,
)
from jobbot.adapters.base import ApplyMethod
from jobbot.adapters.linkedin.sweep import parse_post_blob, parse_posts_fixture, post_to_job
from jobbot.models.candidate import Candidate, PersonalInfo
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind
from jobbot.portals.email_apply import extract_emails, first_apply_email


def test_first_apply_email_with_enviar_cv_context() -> None:
    text = (
        "We're hiring a Senior Data Scientist.\n"
        "Enviar CV a seleccion@empresa.cl.\n"
    )
    assert first_apply_email(text) == "seleccion@empresa.cl"


def test_first_apply_email_ignores_personal_without_apply_context() -> None:
    text = (
        "We're hiring a Data Scientist. Reach me at hola@gmail.com for coffee tips.\n"
        "No apply instructions here.\n"
    )
    assert first_apply_email(text) is None


def test_first_apply_email_with_a_pronoun_between_verb_and_cv() -> None:
    """'send me your CV at …' is the usual English wording and was being missed."""
    text = (
        "We're hiring Senior & Lead AI/ML talent.\n"
        "You can apply directly through the links above or send me your CV "
        "in English at recruiter@example.com."
    )
    assert first_apply_email(text) == "recruiter@example.com"


def test_first_apply_email_with_spanish_pronoun_wording() -> None:
    assert first_apply_email("Envíame tu CV a talento@example.com") == "talento@example.com"
    assert (
        first_apply_email("Compárteme tu hoja de vida a talento@example.org")
        == "talento@example.org"
    )


def test_first_apply_email_still_ignores_networking_invitations() -> None:
    text = "Great post! Write me at someone@example.net if you want to chat about hiking."
    assert first_apply_email(text) is None


def test_extract_emails_unique_order() -> None:
    assert extract_emails("a@x.com then b@y.com then a@x.com") == ["a@x.com", "b@y.com"]


def test_sweep_email_only_post_sets_mailto_and_jd_is_post(
    project_root: Path,
) -> None:
    text = (project_root / "tests/fixtures/linkedin_posts.txt").read_text(encoding="utf-8")
    posts = parse_posts_fixture(text)
    email_posts = [p for p in posts if p.ats_kind == AtsKind.EMAIL]
    assert len(email_posts) == 1
    post = email_posts[0]
    assert post.ats_url == "mailto:seleccion@empresa.cl"
    job = post_to_job(post, job_id="J0099")
    assert job.description == post.text
    assert job.raw_description == post.text
    assert job.ats_kind == "email"
    assert "Enviar CV" in job.description


def test_ats_url_wins_over_email_in_same_post() -> None:
    post = parse_post_blob(
        "Hiring DS. Apply https://boards.greenhouse.io/acme/jobs/1 "
        "or email seleccion@empresa.cl"
    )
    assert post.ats_kind == AtsKind.GREENHOUSE
    assert post.ats_url and "greenhouse" in post.ats_url


def test_build_apply_plan_email_method() -> None:
    cand = Candidate(
        personal=PersonalInfo(name="Ana Ejemplo", headline="DS", email="ana@example.com")
    )
    job = JobPosting(
        id="J1",
        title="Senior Data Scientist",
        company="Empresa Acequia",
        description="Enviar CV a seleccion@empresa.cl. Causal inference.",
        ats_url="mailto:seleccion@empresa.cl",
        ats_kind="email",
    )
    plan = build_apply_plan(cand, job)
    assert plan.method == ApplyMethod.EMAIL
    assert plan.ats_kind == AtsKind.EMAIL


def test_email_draft_includes_post_as_job_description() -> None:
    cand = Candidate(
        personal=PersonalInfo(name="Ana Ejemplo", headline="Senior DS", email="ana@example.com"),
        summary="Causal inference.",
    )
    jd = "We're hiring. Enviar CV a seleccion@empresa.cl.\nPython and SQL."
    job = JobPosting(
        id="J1",
        title="Senior Data Scientist",
        company="Empresa Acequia",
        description=jd,
        ats_url="mailto:seleccion@empresa.cl",
        ats_kind="email",
    )
    draft = build_email_draft(cand, job, cv_path=Path("/tmp/cv.pdf"))
    assert draft.to == "seleccion@empresa.cl"
    assert "Senior Data Scientist" in draft.subject
    assert "Ana Ejemplo" in draft.subject
    assert "Descripción del cargo" in draft.body
    assert jd in draft.body
    assert "Invented skill XYZ" not in draft.body


def test_gmail_compose_url_contains_to_subject_body() -> None:
    cand = Candidate(personal=PersonalInfo(name="Ana", headline="DS"))
    job = JobPosting(
        id="J1",
        title="DS",
        company="Co",
        description="Enviar CV a seleccion@empresa.cl",
        ats_url="mailto:seleccion@empresa.cl",
        ats_kind="email",
    )
    draft = build_email_draft(cand, job)
    url = gmail_compose_url(draft)
    assert "mail.google.com" in url
    assert "view=cm" in url
    qs = parse_qs(urlparse(url).query)
    assert qs["to"] == ["seleccion@empresa.cl"]
    assert "DS" in unquote(qs["su"][0])
    assert "Descripción del cargo" in unquote(qs["body"][0])


def test_open_gmail_compose_calls_opener() -> None:
    opened: list[str] = []
    cand = Candidate(personal=PersonalInfo(name="Ana", headline="DS"))
    job = JobPosting(
        id="J1",
        title="DS",
        company="Co",
        description="Enviar CV a seleccion@empresa.cl",
        ats_url="mailto:seleccion@empresa.cl",
        ats_kind="email",
    )
    draft = build_email_draft(cand, job)
    url = open_gmail_compose(draft, opener=opened.append)
    assert opened == [url]
    assert "mail.google.com" in url


def test_resolve_cv_path_prefers_job_tailored_cv(tmp_path: Path) -> None:
    """Regression: email apply must attach the job CV, not the base one."""
    output = tmp_path / "output"
    base = output / "base"
    base.mkdir(parents=True)
    (base / "cv.pdf").write_bytes(b"base")
    # Only base exists → fallback
    assert resolve_cv_path(output, "J0005") == base / "cv.pdf"

    job_dir = output / "jobs" / "J0005"
    job_dir.mkdir(parents=True)
    (job_dir / "cv.pdf").write_bytes(b"tailored")
    assert resolve_cv_path(output, "J0005") == job_dir / "cv.pdf"

    packaged = job_dir / "application"
    packaged.mkdir()
    (packaged / "cv.pdf").write_bytes(b"packaged")
    assert resolve_cv_path(output, "J0005") == packaged / "cv.pdf"


def test_resolve_cv_path_none_when_nothing_built(tmp_path: Path) -> None:
    assert resolve_cv_path(tmp_path / "output", "J0005") is None


def test_draft_carries_tailored_cv_path(tmp_path: Path) -> None:
    output = tmp_path / "output"
    job_dir = output / "jobs" / "J0007"
    job_dir.mkdir(parents=True)
    (job_dir / "cv.pdf").write_bytes(b"tailored")
    cand = Candidate(personal=PersonalInfo(name="Ana", headline="DS"))
    job = JobPosting(
        id="J0007",
        title="DS",
        company="Co",
        description="Enviar CV a seleccion@empresa.cl",
        ats_url="mailto:seleccion@empresa.cl",
        ats_kind="email",
    )
    draft = build_email_draft(cand, job, cv_path=resolve_cv_path(output, job.id))
    assert draft.cv_path == job_dir / "cv.pdf"


def test_dry_run_does_not_need_webbrowser() -> None:
    """Building a draft never opens a browser (dry-run path)."""
    cand = Candidate(personal=PersonalInfo(name="Ana", headline="DS"))
    job = JobPosting(
        id="J1",
        title="DS",
        company="Co",
        description="Enviar CV a seleccion@empresa.cl",
        ats_url="mailto:seleccion@empresa.cl",
        ats_kind="email",
    )
    draft = build_email_draft(cand, job)
    assert draft.to
    # gmail_compose_url alone must not open anything
    assert gmail_compose_url(draft).startswith("https://mail.google.com/")


def test_extract_emails_drops_addresses_that_are_not_deliverable_shapes() -> None:
    """The regex accepts shapes an RFC-aware validator rejects; do not offer those."""
    text = (
        "Escribe a seleccion@empresa.cl o a rrhh@empresa.com.mx. "
        "Versión mal escrita: hola@empresa..cl, y un dominio inválido: alguien@-empresa.cl. "
        "Referencias: v1.2@3.4 no es correo."
    )
    found = extract_emails(text)

    assert "seleccion@empresa.cl" in found
    assert "rrhh@empresa.com.mx" in found
    assert "hola@empresa..cl" not in found
    assert "alguien@-empresa.cl" not in found
    assert "v1.2@3.4" not in found


def test_extract_emails_keeps_order_and_dedupes_case_insensitively() -> None:
    text = "A: Seleccion@Empresa.cl, B: jobs@acme.io, A otra vez: seleccion@empresa.cl"
    found = extract_emails(text)
    assert found == ["Seleccion@Empresa.cl", "jobs@acme.io"]


def test_first_apply_email_still_finds_the_real_mailbox() -> None:
    text = "Interesados enviar CV a gforton@eratalent.one antes del viernes."
    assert first_apply_email(text) == "gforton@eratalent.one"


def test_first_apply_email_refuses_an_invalid_address_next_to_the_hint() -> None:
    """Regression: the hint window path bypassed validation and could return garbage."""
    assert first_apply_email("Interesados enviar CV a hola@empresa..cl") is None


def test_first_apply_email_skips_invalid_hit_and_keeps_looking() -> None:
    """The hint-window path scans the regex itself, so it must validate there too."""
    text = "Enviar CV a hola@empresa..cl o a maria.perez@empresa.cl"
    assert first_apply_email(text) == "maria.perez@empresa.cl"
