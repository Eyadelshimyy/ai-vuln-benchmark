"""CV build pipeline: profile → Jinja2 → tex/ats → optional XeLaTeX PDF."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path

from jobbot.config import CvConfig, JobbotConfig
from jobbot.cv.application_log import ApplicationLogOptions, build_application_log
from jobbot.cv.ats import build_ats_text
from jobbot.cv.renderer import CvStyle, cv_language, render_cv_ats, render_cv_tex
from jobbot.cv.selection import (
    select_for_base_cv,
    select_for_job,
    write_selection_json,
)
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch

logger = logging.getLogger("jobbot.cv.build")

XELATEX_INSTALL_HINT = """\
XeLaTeX is required to build the PDF CV but was not found on PATH.

Install options (macOS):
  brew install --cask basictex
  # then open a new terminal and run:
  sudo tlmgr update --self
  sudo tlmgr install collection-fontsrecommended

Or install MacTeX:
  brew install --cask mactex-no-gui

Verify with: which xelatex

ATS text output does not require XeLaTeX:
  jobbot cv build --target ats
"""


class BuildTarget(StrEnum):
    CV = "cv"
    ATS = "ats"


def build_cv(
    candidate: Candidate,
    templates_dir: Path,
    output_dir: Path,
    target: BuildTarget = BuildTarget.CV,
    *,
    job: JobPosting | None = None,
    match: JobMatch | None = None,
    style: CvStyle = CvStyle.MODERNCV,
    application_log: ApplicationLogOptions | None = None,
    credit: bool = False,
) -> list[Path]:
    """Build CV artifacts under output/base/ or output/jobs/<id>/.

    ``application_log`` adds a last page to a job PDF; the base CV and the ATS text
    never carry it, since the ATS text is what gets pasted into forms. ``credit``
    closes the CV with the repository link, only when the candidate opted in.
    """
    if not templates_dir.is_dir():
        msg = f"Templates directory not found: {templates_dir}"
        raise FileNotFoundError(msg)

    if job is not None:
        out_dir = output_dir / "jobs" / job.id
        selection = select_for_job(candidate, job, match)
    else:
        out_dir = output_dir / "base"
        selection = select_for_base_cv(candidate)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if job is not None:
        selection_path = out_dir / "selection.json"
        write_selection_json(selection, selection_path)
        written.append(selection_path)
        if match is not None:
            match_path = out_dir / "match.json"
            match_path.write_text(
                json.dumps(match.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(match_path)

    if target == BuildTarget.ATS:
        ats_path = out_dir / "cv_ats.txt"
        ats_path.write_text(
            build_ats_text(candidate, templates_dir, selection, credit=credit),
            encoding="utf-8",
        )
        logger.info("Wrote ATS CV: %s", ats_path)
        written.append(ats_path)
        return written

    # For job builds, also emit ATS alongside PDF
    tex_path = out_dir / "cv.tex"
    log = None
    if application_log is not None and job is not None and match is not None:
        log = build_application_log(
            candidate,
            job,
            match,
            selection=selection,
            options=application_log,
            fallback_language=cv_language(style),
        )
    tex_content = render_cv_tex(
        candidate, templates_dir, selection, style=style, application_log=log, credit=credit
    )
    tex_path.write_text(tex_content, encoding="utf-8")
    logger.info("Wrote LaTeX CV: %s", tex_path)
    written.append(tex_path)

    pdf_path = _compile_xelatex(tex_path, out_dir)
    written.append(pdf_path)

    if job is not None:
        ats_path = out_dir / "cv_ats.txt"
        ats_path.write_text(
            render_cv_ats(candidate, templates_dir, selection, credit=credit),
            encoding="utf-8",
        )
        written.append(ats_path)

    return written


def should_rebuild_job_cv(job_dir: Path, profile_path: Path) -> bool:
    """True when the adapted CV is missing or older than profile.yaml.

    An existing `cv.pdf` / `cv_ats.txt` is not evidence that it still matches
    the profile. If the profile file is newer than either artifact, rebuild.
    """
    artifacts = [path for path in (job_dir / "cv.pdf", job_dir / "cv_ats.txt") if path.is_file()]
    if not artifacts:
        return True
    if not profile_path.is_file():
        return False
    profile_mtime = profile_path.stat().st_mtime
    return any(profile_mtime > artifact.stat().st_mtime for artifact in artifacts)


APPLICATION_LOG_MARKER = "% jobbot:application-log"


def application_log_options(cv: CvConfig) -> ApplicationLogOptions | None:
    if not cv.jobbot_signature:
        return None
    return ApplicationLogOptions(project_url=cv.jobbot_project_url)


def _log_page_mismatch(job_dir: Path, wanted: bool) -> bool:
    tex = job_dir / "cv.tex"
    if not tex.is_file():
        return False
    return (APPLICATION_LOG_MARKER in tex.read_text(encoding="utf-8")) != wanted


def build_adapted_cv(
    config: JobbotConfig,
    candidate: Candidate,
    job: JobPosting,
    match: JobMatch,
) -> list[Path]:
    """The job CV every command builds: skipped while current, log page when opted in.

    Without XeLaTeX the ATS text is still written. Returns [] when nothing was rebuilt.
    """
    options = application_log_options(config.cv)
    job_dir = config.output_dir / "jobs" / job.id
    if not should_rebuild_job_cv(job_dir, config.profile_path) and not _log_page_mismatch(
        job_dir, wanted=options is not None
    ):
        return []
    try:
        return build_cv(
            candidate,
            config.templates_dir,
            config.output_dir,
            target=BuildTarget.CV,
            job=job,
            match=match,
            application_log=options,
            credit=config.cv.jobbot_signature,
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)
        return build_cv(
            candidate,
            config.templates_dir,
            config.output_dir,
            target=BuildTarget.ATS,
            job=job,
            match=match,
            credit=config.cv.jobbot_signature,
        )


def build_job_cv_bundle(
    candidate: Candidate,
    job: JobPosting,
    match: JobMatch,
    templates_dir: Path,
    output_dir: Path,
) -> list[Path]:
    return build_cv(
        candidate,
        templates_dir,
        output_dir,
        target=BuildTarget.CV,
        job=job,
        match=match,
    )


def _compile_xelatex(tex_path: Path, work_dir: Path) -> Path:
    xelatex = shutil.which("xelatex")
    if xelatex is None:
        raise RuntimeError(XELATEX_INSTALL_HINT)

    log_path = work_dir / "build.log"
    cmd = [
        xelatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={work_dir}",
        str(tex_path),
    ]
    logger.info("Running XeLaTeX: %s", " ".join(cmd))

    combined_log: list[str] = []
    for pass_no in (1, 2):
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=work_dir,
        )
        combined_log.append(f"=== pass {pass_no} ===\n")
        combined_log.append(proc.stdout or "")
        combined_log.append(proc.stderr or "")
        if proc.returncode != 0:
            log_path.write_text("".join(combined_log), encoding="utf-8")
            msg = f"XeLaTeX failed with exit code {proc.returncode}.\nSee build log: {log_path}"
            raise RuntimeError(msg)

    log_path.write_text("".join(combined_log), encoding="utf-8")
    pdf_path = work_dir / "cv.pdf"
    if not pdf_path.is_file():
        msg = f"XeLaTeX reported success but PDF missing: {pdf_path}\nSee {log_path}"
        raise RuntimeError(msg)
    logger.info("Wrote PDF CV: %s", pdf_path)
    return pdf_path
