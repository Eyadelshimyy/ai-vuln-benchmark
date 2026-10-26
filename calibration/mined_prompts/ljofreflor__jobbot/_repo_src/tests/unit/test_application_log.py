"""Opt-in application log: a last page saying how and for which posting the CV was built."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from jobbot.config import CvConfig, JobbotConfig, PathsConfig
from jobbot.cv.application_log import (
    ApplicationLog,
    ApplicationLogOptions,
    build_application_log,
    posting_url,
)
from jobbot.cv.build import BuildTarget, build_cv
from jobbot.cv.renderer import CvStyle, render_cv_tex
from jobbot.cv.selection import select_for_job
from jobbot.jobs.language import Language
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch, MatchItem, MatchStrength
from tests.fixtures.profile import sample_profile_dict

TODAY = date(2026, 9, 27)
TOOL_URL = "https://example.org/tool"


SPANISH_JD = (
    "Buscamos un Senior Data Scientist para el equipo de datos de la empresa. "
    "Trabajarás con Python y SQL en modelos para nuestros clientes."
)
ENGLISH_JD = (
    "We are looking for a Senior Data Scientist to join our data team. "
    "You will work with Python and SQL on models for the business."
)


def _job(**overrides: object) -> JobPosting:
    data: dict[str, object] = {
        "id": "J0900",
        "title": "Senior Data Scientist",
        "company": "Acme Analytics",
        "url": "https://jobs.example.com/postings/42?utm_source=linkedin&rcm=ACoAA123#apply",
        "skills": ["Python", "SQL", "Kubernetes"],
        "description": SPANISH_JD,
    }
    data.update(overrides)
    return JobPosting.model_validate(data)


def _candidate() -> Candidate:
    return Candidate.model_validate(sample_profile_dict())


def _log(
    job: JobPosting | None = None,
    *,
    project_url: str | None = TOOL_URL,
    fallback_language: Language = "es",
) -> ApplicationLog:
    candidate = _candidate()
    job = job or _job()
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    return build_application_log(
        candidate,
        job,
        match,
        selection=select_for_job(candidate, job, match),
        options=ApplicationLogOptions(project_url=project_url, generated_on=TODAY),
        fallback_language=fallback_language,
    )


def test_posting_url_strips_tracking() -> None:
    assert posting_url(_job()) == "https://jobs.example.com/postings/42"


def test_posting_url_keeps_the_indeed_job_key() -> None:
    job = _job(url="https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=serp&vjs=3")
    assert posting_url(job) == "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"


def test_posting_url_falls_back_to_ats_url_and_never_to_an_email() -> None:
    assert posting_url(_job(url=None, ats_url="https://ats.example.com/j/7?ref=x")) == (
        "https://ats.example.com/j/7"
    )
    assert posting_url(_job(url=None, ats_url="mailto:jobs@example.com")) is None
    assert posting_url(_job(url=None, ats_url=None)) is None


def test_log_lists_only_strong_requirements_with_a_pointer() -> None:
    log = _log()

    requirements = [line.requirement for line in log.evidence]
    assert "Python" in requirements
    assert "SQL" in requirements
    # a missing requirement would document a gap: never listed
    assert "Kubernetes" not in requirements
    python = next(line for line in log.evidence if line.requirement == "Python")
    assert python.kind == "skills"
    assert python.pointer == "Programming"


def test_log_never_lists_partial_requirements() -> None:
    candidate = _candidate()
    job = _job()
    match = JobMatch(
        job_id=job.id,
        score=50,
        items=[
            MatchItem(label="Python", strength=MatchStrength.STRONG, detail="present"),
            MatchItem(label="SQL", strength=MatchStrength.PARTIAL, detail="related"),
            MatchItem(label="Kubernetes", strength=MatchStrength.MISSING, detail="not found"),
        ],
    )
    log = build_application_log(
        candidate,
        job,
        match,
        selection=select_for_job(candidate, job, match),
        options=ApplicationLogOptions(generated_on=TODAY),
    )
    assert [line.requirement for line in log.evidence] == ["Python"]


def test_strong_requirement_without_a_pointer_in_this_cv_is_dropped() -> None:
    """Evidence must point at something the reader can find in this CV."""
    candidate = _candidate()
    job = _job()
    match = JobMatch(
        job_id=job.id,
        score=100,
        items=[MatchItem(label="Quantum Origami", strength=MatchStrength.STRONG)],
    )
    log = build_application_log(
        candidate,
        job,
        match,
        selection=select_for_job(candidate, job, match),
        options=ApplicationLogOptions(generated_on=TODAY),
    )
    assert log.evidence == []


def test_experience_pointer_names_title_and_company() -> None:
    candidate = _candidate()
    job = _job()
    match = JobMatch(
        job_id=job.id,
        score=100,
        items=[MatchItem(label="Share of Wallet", strength=MatchStrength.STRONG)],
    )
    log = build_application_log(
        candidate,
        job,
        match,
        selection=select_for_job(candidate, job, match),
        options=ApplicationLogOptions(generated_on=TODAY),
    )
    assert log.evidence[0].kind == "experience"
    assert log.evidence[0].pointer == "Senior Data Scientist, Mercado Libre"


def test_evidence_is_capped() -> None:
    candidate = _candidate()
    job = _job()
    match = JobMatch(
        job_id=job.id,
        score=100,
        items=[MatchItem(label="Python", strength=MatchStrength.STRONG)] * 20,
    )
    log = build_application_log(
        candidate,
        job,
        match,
        selection=select_for_job(candidate, job, match),
        options=ApplicationLogOptions(generated_on=TODAY),
    )
    assert len(log.evidence) <= 8


def _tex(root: Path, style: CvStyle, log: ApplicationLog | None) -> str:
    candidate = _candidate()
    job = _job()
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    selection = select_for_job(candidate, job, match)
    return render_cv_tex(
        candidate,
        root / "templates",
        selection,
        style=style,
        application_log=log,
    )


def test_moderncv_without_log_is_unchanged(project_root: Path) -> None:
    tex = _tex(project_root, CvStyle.MODERNCV, None)
    assert "Registro de la postulación" not in tex
    assert r"\clearpage" not in tex
    assert "JobBot" not in tex


def test_moderncv_log_is_a_last_page_after_the_body(project_root: Path) -> None:
    log = _log()
    tex = _tex(project_root, CvStyle.MODERNCV, log)

    body_end = tex.index(r"\section{Publicaciones}")
    page_break = tex.index(r"\clearpage")
    title = tex.index(r"\section{Registro de la postulación}")
    assert body_end < page_break < title < tex.index(r"\end{document}")
    tail = tex[title:]
    assert r"Este CV fue creado con JobBot (\href{https://example.org/tool}" in tail
    assert "Senior Data Scientist @ Acme Analytics" in tail
    assert "https://jobs.example.com/postings/42}" in tail
    assert "utm_source" not in tex
    assert "rcm=" not in tex
    assert "2026-09-27" in tail
    assert rf"Afinidad con la oferta: {log.percent}\%" in tail
    assert "calculada por JobBot comparando los requisitos del aviso con este perfil" in tail
    assert "Requisitos del aviso y evidencia en este CV" in tail
    assert r"Python $\rightarrow$ Habilidades (Programming)" in tail
    assert "Kubernetes" not in tex


def test_english_posting_gets_an_english_log_on_a_spanish_cv(project_root: Path) -> None:
    """The reader is the posting's recruiter: the log speaks the posting's language."""
    log = _log(_job(description=ENGLISH_JD), fallback_language="es")
    tex = _tex(project_root, CvStyle.MODERNCV, log)

    title = tex.index(r"\section{Application log}")
    assert tex.index(r"\clearpage") < title
    tail = tex[title:]
    assert r"This CV was created with JobBot (\href{https://example.org/tool}" in tail
    assert "for this posting:" in tail
    assert "Generated on: 2026-09-27" in tail
    assert rf"Match with the posting: {log.percent}\%" in tail
    assert "computed by JobBot comparing the posting's requirements with this profile" in tail
    assert "Posting requirements and evidence in this CV" in tail
    # the pointer names the heading as this CV prints it, so the reader can find it
    assert r"Python $\rightarrow$ Habilidades (Programming)" in tail
    assert "Registro de la postulación" not in tex
    assert "Kubernetes" not in tex


def test_spanish_posting_gets_a_spanish_log_on_an_english_cv(project_root: Path) -> None:
    log = _log(_job(description=SPANISH_JD), fallback_language="en")
    tex = _tex(project_root, CvStyle.PLAIN, log)

    title = tex.index(r"\section*{Registro de la postulación}")
    assert tex.index(r"\clearpage") < title
    tail = tex[title:]
    assert "Este CV fue creado con JobBot" in tail
    assert "Afinidad con la oferta:" in tail
    assert r"Python $\rightarrow$ Skills (Programming)" in tail
    assert "Application log" not in tex


def test_unclear_posting_language_falls_back_to_the_cv_language() -> None:
    unclear = _job(description="", skills=["Python", "SQL"])
    assert _log(unclear, fallback_language="es").language == "es"
    assert _log(unclear, fallback_language="en").language == "en"


def test_log_without_urls(project_root: Path) -> None:
    log = _log(_job(url=None, ats_url=None), project_url=None)
    tex = _tex(project_root, CvStyle.MODERNCV, log)
    tail = tex[tex.index("Registro de la postulación") :]

    assert "Este CV fue creado con JobBot para esta oferta" in tail
    assert r"\href" not in tail
    assert "None" not in tail


def test_ats_text_never_carries_the_log(project_root: Path, tmp_path: Path) -> None:
    candidate = _candidate()
    job = _job()
    outputs = build_cv(
        candidate,
        project_root / "templates",
        tmp_path,
        target=BuildTarget.ATS,
        job=job,
        match=RuleBasedJobAnalyzer().analyze(candidate, job),
        application_log=ApplicationLogOptions(project_url=TOOL_URL, generated_on=TODAY),
    )
    ats = outputs[-1].read_text(encoding="utf-8")
    assert "JobBot" not in ats
    assert "Registro" not in ats
    assert "Application log" not in ats


def test_build_falls_back_to_the_style_language() -> None:
    from jobbot.cv.renderer import cv_language

    assert cv_language(CvStyle.MODERNCV) == "es"
    assert cv_language(CvStyle.PLAIN) == "en"


def test_cv_config_signature_is_off_by_default(tmp_path: Path) -> None:
    from jobbot.config import load_config

    config = load_config(tmp_path)
    assert config.cv.jobbot_signature is False
    assert config.cv.jobbot_project_url is None


def test_cv_config_reads_the_opt_in(tmp_path: Path) -> None:
    from jobbot.config import load_config

    (tmp_path / ".jobbot.toml").write_text(
        f'[cv]\njobbot_signature = true\njobbot_project_url = "{TOOL_URL}"\n',
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.cv.jobbot_signature is True
    assert config.cv.jobbot_project_url == TOOL_URL


def _fake_xelatex(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(tex_path: Path, work_dir: Path) -> Path:
        pdf = work_dir / "cv.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        return pdf

    monkeypatch.setattr("jobbot.cv.build._compile_xelatex", fake)


def _config(tmp_path: Path, project_root: Path, *, on: bool) -> JobbotConfig:
    return JobbotConfig(
        paths=PathsConfig(templates=project_root / "templates"),
        root=tmp_path,
        cv=CvConfig(jobbot_signature=on, jobbot_project_url=TOOL_URL if on else None),
    )


def _job_outputs(tmp_path: Path) -> tuple[str, str]:
    job_dir = tmp_path / "output" / "jobs" / "J0900"
    return (
        (job_dir / "cv.tex").read_text(encoding="utf-8"),
        (job_dir / "cv_ats.txt").read_text(encoding="utf-8"),
    )


def test_jobbot_get_rebuild_includes_the_log_when_on(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`jobbot get` rebuilt the job CV through its own build_cv call, without the log."""
    from jobbot.jobs.from_url import _build_adapted_cv

    _fake_xelatex(monkeypatch)
    candidate, job = _candidate(), _job()
    match = RuleBasedJobAnalyzer().analyze(candidate, job)

    _build_adapted_cv(_config(tmp_path, project_root, on=True), candidate, job, match)

    tex, ats = _job_outputs(tmp_path)
    assert r"\section{Registro de la postulación}" in tex
    assert "Registro" not in ats


def test_application_apply_rebuild_includes_the_log_when_on(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot.cli import _build_job_cv

    _fake_xelatex(monkeypatch)
    _build_job_cv(_config(tmp_path, project_root, on=True), _candidate(), _job())

    tex, ats = _job_outputs(tmp_path)
    assert r"\section{Registro de la postulación}" in tex
    assert "Registro" not in ats


def test_rebuild_paths_leave_the_log_out_when_off(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot.cli import _build_job_cv

    _fake_xelatex(monkeypatch)
    _build_job_cv(_config(tmp_path, project_root, on=False), _candidate(), _job())

    tex, _ = _job_outputs(tmp_path)
    assert "Registro de la postulación" not in tex
    assert r"\clearpage" not in tex


def test_turning_the_setting_on_rebuilds_a_cv_that_is_otherwise_current(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CV newer than the profile was kept as is, so the log never showed up."""
    from jobbot.cv.build import build_adapted_cv

    _fake_xelatex(monkeypatch)
    candidate, job = _candidate(), _job()
    match = RuleBasedJobAnalyzer().analyze(candidate, job)

    assert build_adapted_cv(_config(tmp_path, project_root, on=False), candidate, job, match)
    # same setting, nothing changed: the CV on disk is current
    assert build_adapted_cv(_config(tmp_path, project_root, on=False), candidate, job, match) == []

    assert build_adapted_cv(_config(tmp_path, project_root, on=True), candidate, job, match)
    tex, _ = _job_outputs(tmp_path)
    assert "Registro de la postulación" in tex

    assert build_adapted_cv(_config(tmp_path, project_root, on=False), candidate, job, match)
    tex, _ = _job_outputs(tmp_path)
    assert "Registro de la postulación" not in tex


def test_signature_setting_also_controls_the_repository_credit(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off by default: neither the PDF source nor the ATS text names the tool."""
    from jobbot.cv.build import build_adapted_cv

    _fake_xelatex(monkeypatch)
    candidate, job = _candidate(), _job()
    match = RuleBasedJobAnalyzer().analyze(candidate, job)

    build_adapted_cv(_config(tmp_path, project_root, on=False), candidate, job, match)
    tex, ats = _job_outputs(tmp_path)
    assert "powered by Jobbot" not in tex
    assert "powered by Jobbot" not in ats

    build_adapted_cv(_config(tmp_path, project_root, on=True), candidate, job, match)
    tex, ats = _job_outputs(tmp_path)
    assert "powered by Jobbot sync CV" in tex
    assert "powered by Jobbot sync CV" in ats
