"""Jinja2 rendering of CV templates."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from babel.units import format_unit
from jinja2 import Environment, FileSystemLoader, select_autoescape

from jobbot.branding import CV_CREDIT, MARK, REPO_URL
from jobbot.cv.application_log import ApplicationLog
from jobbot.cv.latex import escape_latex, escape_latex_multiline, escape_latex_url
from jobbot.cv.selection import SelectionResult, filter_experiences, select_for_base_cv
from jobbot.jobs.language import Language
from jobbot.models.candidate import Candidate
from jobbot.models.education import Education
from jobbot.models.experience import format_metric
from jobbot.models.targets import ProfileTarget


class CvStyle(StrEnum):
    """LaTeX look of the generated CV."""

    MODERNCV = "moderncv"  # mirrors the candidate's own moderncv/banking CV
    PLAIN = "plain"  # portable article layout, no moderncv class needed


_TEMPLATE_BY_STYLE = {
    CvStyle.MODERNCV: "cv_moderncv.tex.j2",
    CvStyle.PLAIN: "cv.tex.j2",
}

_LANGUAGE_BY_STYLE: dict[CvStyle, Language] = {
    CvStyle.MODERNCV: "es",
    CvStyle.PLAIN: "en",
}


def cv_language(style: CvStyle) -> Language:
    """The language the template's own headings are written in."""
    return _LANGUAGE_BY_STYLE[style]


def split_name(full_name: str) -> tuple[str, str]:
    """Split into moderncv's given / family parts (Spanish: two of each)."""
    parts = full_name.split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    given = 2 if len(parts) >= 4 else 1
    return " ".join(parts[:given]), " ".join(parts[given:])


def social_handle(url: str | None) -> str | None:
    """moderncv wants the handle, profile.yaml stores the full URL."""
    if not url:
        return None
    tail = str(url).rstrip("/").rsplit("/", 1)[-1]
    return tail or None


# CLDR abbreviates September as "sept"; this CV uses three letters everywhere,
# so month names stay a design decision of the CV, not locale data.
_ES_MONTHS = (
    "Ene",
    "Feb",
    "Mar",
    "Abr",
    "May",
    "Jun",
    "Jul",
    "Ago",
    "Sep",
    "Oct",
    "Nov",
    "Dic",
)


def _parse_ym(value: str | None) -> tuple[int, int | None] | None:
    if not value:
        return None
    parts = str(value).strip().split("-")
    try:
        year = int(parts[0])
    except (ValueError, IndexError):
        return None
    month = None
    if len(parts) > 1:
        try:
            month = int(parts[1])
        except ValueError:
            month = None
    return year, month


def _ym_label(parsed: tuple[int, int | None]) -> str:
    year, month = parsed
    if month is None or not 1 <= month <= 12:
        return str(year)
    return f"{_ES_MONTHS[month - 1]} {year}"


def _duration_label(start: tuple[int, int | None], end: tuple[int, int | None]) -> str:
    if start[1] is None or end[1] is None:
        return ""
    total = (end[0] - start[0]) * 12 + (end[1] - start[1]) + 1
    if total <= 0:
        return ""
    years, months = divmod(total, 12)
    chunks = [
        format_unit(amount, f"duration-{unit}", locale="es")
        for amount, unit in ((years, "year"), (months, "month"))
        if amount
    ]
    return f" ({', '.join(chunks)})" if chunks else ""


def es_date_range(start: str | None, end: str | None) -> str:
    """Spanish date range as the candidate writes it in his own CV."""
    parsed_start = _parse_ym(start)
    parsed_end = _parse_ym(end)
    if parsed_start is None and parsed_end is None:
        return ""
    if parsed_start is None and parsed_end is not None:
        return _ym_label(parsed_end)
    assert parsed_start is not None
    if parsed_end is None:
        return f"{_ym_label(parsed_start)} – Presente"
    duration = _duration_label(parsed_start, parsed_end)
    return f"{_ym_label(parsed_start)} – {_ym_label(parsed_end)}{duration}"


def es_year_range(start: str | None, end: str | None) -> str:
    """Education keeps years only, like the real CV ('2017 – 2019')."""
    parsed_start = _parse_ym(start)
    parsed_end = _parse_ym(end)
    if parsed_start is None and parsed_end is None:
        return ""
    if parsed_start is None and parsed_end is not None:
        return str(parsed_end[0])
    assert parsed_start is not None
    if parsed_end is None:
        return f"{parsed_start[0]} – Presente"
    if parsed_start[0] == parsed_end[0]:
        return str(parsed_start[0])
    return f"{parsed_start[0]} – {parsed_end[0]}"


def _build_env(templates_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["latex"] = escape_latex
    env.filters["latex_para"] = escape_latex_multiline
    env.filters["latex_url"] = escape_latex_url
    return env


def _bold_latex(text: str) -> str:
    return rf"\textbf{{{escape_latex(text)}}}"


def _group_label(name: str) -> str:
    return name.replace("_", " ").title()


def _edu_dates(edu: Education) -> str:
    if edu.start_date and edu.end_date:
        return f"{edu.start_date} – {edu.end_date}"
    return edu.end_date or edu.start_date or ""


def _metrics_line(metrics: Mapping[str, int | float | str]) -> str:
    return "; ".join(format_metric(k, v) for k, v in metrics.items())


def _metrics_line_latex(metrics: Mapping[str, int | float | str]) -> str:
    return escape_latex(_metrics_line(metrics))


def _contact_line_latex(candidate: Candidate) -> str:
    personal = candidate.personal
    parts: list[str] = []
    loc = personal.location_line()
    if loc:
        parts.append(escape_latex(loc))
    if personal.email:
        email = str(personal.email)
        parts.append(rf"\href{{mailto:{email}}}{{{escape_latex(email)}}}")
    if personal.phone:
        parts.append(escape_latex(personal.phone))
    if personal.linkedin:
        parts.append(rf"\href{{{personal.linkedin}}}{{LinkedIn}}")
    if personal.github:
        parts.append(rf"\href{{{personal.github}}}{{GitHub}}")
    if personal.orcid_url:
        parts.append(_href_latex(personal.orcid_url))
    return r" $\cdot$ ".join(parts)


def _specialties_line_latex(candidate: Candidate) -> str:
    return r" $\cdot$ ".join(escape_latex(s) for s in candidate.specialties)


def url_label(url: str) -> str:
    """What a reader sees for a link: no scheme, no trailing slash."""
    return url.split("://", 1)[-1].rstrip("/")


def _href_latex(url: str) -> str:
    return rf"\href{{{escape_latex_url(url)}}}{{{escape_latex(url_label(url))}}}"


def render_cv_tex(
    candidate: Candidate,
    templates_dir: Path,
    selection: SelectionResult | None = None,
    *,
    style: CvStyle = CvStyle.MODERNCV,
    application_log: ApplicationLog | None = None,
    credit: bool = False,
) -> str:
    selection = selection or select_for_base_cv(candidate)
    env = _build_env(templates_dir)
    template = env.get_template(_TEMPLATE_BY_STYLE[style])
    experiences = filter_experiences(candidate, selection)
    name_first, name_last = split_name(candidate.personal.name)
    return template.render(
        candidate=candidate,
        personal=candidate.personal,
        experiences=experiences,
        skills=candidate.skills.as_dict(),
        publications=candidate.publications if selection.include_publications else [],
        projects=candidate.projects,
        href=_href_latex,
        application_log=application_log,
        contact_line=_contact_line_latex(candidate),
        specialties_line=_specialties_line_latex(candidate),
        metrics_line=_metrics_line_latex,
        edu_dates=_edu_dates,
        group_label=_group_label,
        bold=_bold_latex,
        name_first=name_first,
        name_last=name_last,
        exp_dates=lambda exp: es_date_range(exp.start_date, exp.end_date),
        edu_years=lambda edu: es_year_range(edu.start_date, edu.end_date),
        linkedin_handle=social_handle(candidate.personal.linkedin),
        github_handle=social_handle(candidate.personal.github),
        repo_url=REPO_URL,
        mark=MARK,
        cv_credit=CV_CREDIT if credit else None,
        target=ProfileTarget.CV,
    )


def render_cv_ats(
    candidate: Candidate,
    templates_dir: Path,
    selection: SelectionResult | None = None,
    *,
    credit: bool = False,
) -> str:
    selection = selection or select_for_base_cv(candidate)
    env = _build_env(templates_dir)
    template = env.get_template("cv_ats.txt.j2")
    experiences = filter_experiences(candidate, selection)
    return template.render(
        candidate=candidate,
        personal=candidate.personal,
        experiences=experiences,
        skills=candidate.skills.as_dict(),
        publications=candidate.publications if selection.include_publications else [],
        projects=candidate.projects,
        metrics_line=_metrics_line,
        edu_dates=_edu_dates,
        group_label=_group_label,
        cv_credit=CV_CREDIT if credit else None,
        target=ProfileTarget.ATS,
    )
