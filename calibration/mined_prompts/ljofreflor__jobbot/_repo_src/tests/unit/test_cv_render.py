"""CV rendering tests (no XeLaTeX required)."""

from __future__ import annotations

from pathlib import Path

from jobbot.cv.build import BuildTarget, build_cv
from jobbot.cv.renderer import CvStyle, render_cv_ats, render_cv_tex
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import sample_profile_dict


def test_render_tex_escapes_and_includes_name(project_root: Path) -> None:
    data = sample_profile_dict()
    data["personal"]["name"] = "Ana & Co"
    candidate = Candidate.model_validate(data)
    tex = render_cv_tex(candidate, project_root / "templates")
    # moderncv splits the name: \name{given}{family}, each escaped on its own
    assert r"\name{Ana}{\& Co}" in tex
    assert "Mercado Libre" in tex
    assert "Share of Wallet" in tex
    assert "github.com/ljofreflor/jobbot" not in tex
    assert "powered by Jobbot" not in tex


def test_render_ats_plain_text(project_root: Path) -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    ats = render_cv_ats(candidate, project_root / "templates")
    assert "SUMMARY" in ats
    assert "EXPERIENCE" in ats
    assert "SKILLS" in ats
    assert "Ana Ejemplo" in ats
    assert "Python" in ats
    assert "github.com/ljofreflor/jobbot" not in ats
    assert "powered by Jobbot" not in ats
    assert "\\section" not in ats
    assert "\\textbf" not in ats


def test_jobbot_credit_is_opt_in(project_root: Path) -> None:
    """The repository credit is printed only when the candidate turned it on."""
    candidate = Candidate.model_validate(sample_profile_dict())
    templates = project_root / "templates"
    for style in (CvStyle.MODERNCV, CvStyle.PLAIN):
        assert "powered by Jobbot" not in render_cv_tex(candidate, templates, style=style)
        tex = render_cv_tex(candidate, templates, style=style, credit=True)
        assert r"\url{https://github.com/ljofreflor/jobbot}" in tex
        assert "powered by Jobbot sync CV" in tex
    assert "powered by Jobbot" not in render_cv_ats(candidate, templates)
    ats = render_cv_ats(candidate, templates, credit=True)
    assert "https://github.com/ljofreflor/jobbot" in ats


def test_render_ats_keeps_every_field_on_its_own_line(project_root: Path) -> None:
    """Jinja trim_blocks drops the newline after a block tag: fields glued together."""
    data = sample_profile_dict()
    data["experience"][0]["achievements"].append(
        {
            "id": "meli-experimentation",
            "text": "Framework de experimentación A/B.",
            "tags": [],
            "metrics": {},
        }
    )
    data["experience"].append(
        {
            "id": "retail-co",
            "company": "Retail Co",
            "title": "Data Scientist",
            "location": "Santiago, Chile",
            "start_date": "2019-01",
            "end_date": "2022-03",
            "current": False,
            "description": "Forecasting y pricing para retail omnicanal.",
            "achievements": [],
        }
    )
    candidate = Candidate.model_validate(data)

    lines = render_cv_ats(candidate, project_root / "templates").splitlines()

    assert "2022-04 – 2025-05 | Remote" in lines
    assert "Customer analytics & credit risk." in lines
    assert "- Framework de experimentación A/B." in lines
    assert "Data Scientist — Retail Co" in lines
    assert "Forecasting y pricing para retail omnicanal." in lines
    assert "EDUCATION" in lines
    assert "SKILLS" in lines
    assert "PUBLICATIONS" in lines


def test_build_ats_writes_file(project_root: Path, tmp_path: Path) -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    outputs = build_cv(
        candidate=candidate,
        templates_dir=project_root / "templates",
        output_dir=tmp_path / "output",
        target=BuildTarget.ATS,
    )
    assert len(outputs) == 1
    assert outputs[0].name == "cv_ats.txt"
    assert outputs[0].is_file()
    assert "EXPERIENCE" in outputs[0].read_text(encoding="utf-8")


def test_split_name_spanish_two_given_two_family() -> None:
    from jobbot.cv.renderer import split_name

    assert split_name("Leonardo Andrés Jofré Flor") == ("Leonardo Andrés", "Jofré Flor")
    assert split_name("Ana Pérez Soto") == ("Ana", "Pérez Soto")
    assert split_name("Ana Ejemplo") == ("Ana", "Ejemplo")
    assert split_name("Ana") == ("Ana", "")


def test_social_handle_from_url() -> None:
    from jobbot.cv.renderer import social_handle

    assert social_handle("https://www.linkedin.com/in/leonardojofre") == "leonardojofre"
    assert social_handle("https://github.com/ljofreflor/") == "ljofreflor"
    assert social_handle("") is None


def test_render_moderncv_matches_the_real_cv_design(project_root: Path) -> None:
    """The derived CV must look like the candidate's own LaTeX CV (moderncv banking)."""
    from jobbot.cv.renderer import CvStyle

    candidate = Candidate.model_validate(sample_profile_dict())
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.MODERNCV)

    assert r"\documentclass[11pt,a4paper,sans]{moderncv}" in tex
    assert r"\moderncvstyle{banking}" in tex
    assert r"\setmainlanguage{spanish}" in tex
    assert r"\makecvtitle" in tex
    assert r"\cventry{" in tex
    assert r"\cvitem{" in tex
    assert r"\section{Resumen Profesional}" in tex
    assert r"\section{Experiencia Profesional}" in tex
    assert r"\section{Habilidades}" in tex
    assert "Mercado Libre" in tex
    # article-only markup must not leak into the moderncv build
    assert r"\documentclass[11pt,a4paper]{article}" not in tex
    assert r"\section*{Summary}" not in tex


def test_render_plain_style_still_available(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    candidate = Candidate.model_validate(sample_profile_dict())
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.PLAIN)
    assert r"\documentclass[11pt,a4paper]{article}" in tex
    assert "github.com/ljofreflor/jobbot" not in tex


def test_build_cv_moderncv_writes_tex_for_job(project_root: Path, tmp_path: Path) -> None:
    """Job builds go through the same style as the base CV."""
    from jobbot.cv.renderer import CvStyle, render_cv_tex

    candidate = Candidate.model_validate(sample_profile_dict())
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.MODERNCV)
    assert candidate.personal.headline is None or candidate.personal.headline in tex


def test_es_date_range_matches_the_real_cv_format() -> None:
    """The real CV writes 'Abr 2022 – May 2025 (3 años, 2 meses)'."""
    from jobbot.cv.renderer import es_date_range

    assert es_date_range("2022-04", "2025-05") == "Abr 2022 – May 2025 (3 años, 2 meses)"
    assert es_date_range("2025-10", None) == "Oct 2025 – Presente"
    assert es_date_range("2008-01", "2014-12") == "Ene 2008 – Dic 2014 (7 años)"
    assert es_date_range("2019-05", "2019-08") == "May 2019 – Ago 2019 (4 meses)"
    assert es_date_range(None, None) == ""


def test_moderncv_renders_address_and_spanish_dates(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    data = sample_profile_dict()
    data["personal"]["city"] = "Santiago"
    data["personal"]["country"] = "Chile"
    candidate = Candidate.model_validate(data)
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.MODERNCV)

    assert r"\address{Santiago}{Chile}{}" in tex
    # ISO months must not reach the PDF
    assert "2022-04" not in tex
    assert "Abr 2022" in tex


def _with_project(url: str | None = "https://example.org/huerto#inicio") -> Candidate:
    data = sample_profile_dict()
    data["projects"] = [
        {
            "id": "huerto",
            "name": "Huerto Comunitario",
            "description": "Coordinación de un huerto vecinal & su compostaje.",
            "url": url,
            "tags": ["Compostaje", "Riego"],
        }
    ]
    return Candidate.model_validate(data)


def test_moderncv_renders_projects(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    tex = render_cv_tex(_with_project(), project_root / "templates", style=CvStyle.MODERNCV)

    assert r"\section{Proyectos Personales}" in tex
    # banking prints the third \cventry argument as the bold heading: the name, not the link
    href = r"\href{https://example.org/huerto\#inicio}{example.org/huerto\#inicio}"
    assert rf"\cventry{{}}{{{href}}}{{Huerto Comunitario}}{{}}{{}}{{}}" in tex
    assert r"huerto vecinal \& su compostaje" in tex
    assert "Compostaje, Riego" in tex


def test_plain_style_renders_projects(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    tex = render_cv_tex(_with_project(), project_root / "templates", style=CvStyle.PLAIN)

    assert r"\section*{Projects}" in tex
    assert r"\textbf{Huerto Comunitario}" in tex
    assert r"\href{https://example.org/huerto\#inicio}" in tex


def test_ats_renders_projects(project_root: Path) -> None:
    lines = render_cv_ats(_with_project(), project_root / "templates").splitlines()

    assert "PROJECTS" in lines
    assert "Huerto Comunitario — https://example.org/huerto#inicio" in lines
    assert "Coordinación de un huerto vecinal & su compostaje." in lines
    assert "Compostaje, Riego" in lines


def test_project_without_url_renders_without_link(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    candidate = _with_project(url=None)
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.MODERNCV)
    ats = render_cv_ats(candidate, project_root / "templates")

    assert r"\cventry{}{}{Huerto Comunitario}{}{}{}" in tex
    assert r"\href{None" not in tex
    assert "Huerto Comunitario" in ats.splitlines()


def test_no_projects_no_section(project_root: Path) -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    tex = render_cv_tex(candidate, project_root / "templates")
    ats = render_cv_ats(candidate, project_root / "templates")

    assert "Proyectos" not in tex
    assert "PROJECTS" not in ats


def _with_orcid() -> Candidate:
    data = sample_profile_dict()
    data["personal"]["orcid"] = "0000-0002-1825-0097"
    return Candidate.model_validate(data)


_ORCID_HREF = r"\href{https://orcid.org/0000-0002-1825-0097}{orcid.org/0000-0002-1825-0097}"


def test_moderncv_renders_orcid_as_linked_text(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    tex = render_cv_tex(_with_orcid(), project_root / "templates", style=CvStyle.MODERNCV)

    # \extrainfo joins the header contact line without a symbol glyph
    assert rf"\extrainfo{{{_ORCID_HREF}}}" in tex
    assert r"\social[orcid]" not in tex


def test_plain_style_renders_orcid_in_contact_line(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    tex = render_cv_tex(_with_orcid(), project_root / "templates", style=CvStyle.PLAIN)

    assert rf"$\cdot$ {_ORCID_HREF}" in tex


def test_ats_renders_orcid_on_its_own_line(project_root: Path) -> None:
    lines = render_cv_ats(_with_orcid(), project_root / "templates").splitlines()

    assert "ORCID: https://orcid.org/0000-0002-1825-0097" in lines


def test_no_orcid_renders_nothing(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    candidate = Candidate.model_validate(sample_profile_dict())
    templates = project_root / "templates"
    for style in CvStyle:
        tex = render_cv_tex(candidate, templates, style=style)
        assert "orcid" not in tex.lower()
        assert r"\extrainfo" not in tex
    assert "ORCID" not in render_cv_ats(candidate, templates)


def test_moderncv_education_puts_degree_before_institution(project_root: Path) -> None:
    from jobbot.cv.renderer import CvStyle

    candidate = Candidate.model_validate(sample_profile_dict())
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.MODERNCV)
    edu = candidate.education[0]
    marker = rf"\textbf{{{edu.degree}}}}}{{{edu.institution}}}"
    assert marker in tex


def test_moderncv_null_city_and_location_do_not_crash(project_root: Path) -> None:
    """Null optional strings must not hit escape_latex via the Jinja filter (#112)."""
    from jobbot.cv.renderer import CvStyle

    data = sample_profile_dict()
    data["personal"]["city"] = None
    data["personal"]["country"] = "Chile"
    data["experience"][0]["location"] = None
    candidate = Candidate.model_validate(data)
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.MODERNCV)
    assert r"\address{}{Chile}{}" in tex
    assert "Mercado Libre" in tex


def test_moderncv_cventry_wraps_long_left_column(project_root: Path) -> None:
    """Banking tabular* uses a wrapping p-column so long org names do not overflow (#112)."""
    from jobbot.cv.renderer import CvStyle

    data = sample_profile_dict()
    long_org = (
        "Ministerio de Ciencia, Tecnología, Conocimiento e Innovación — "
        "Subsecretaría de Ciencia — Departamento de Estudios y Estadísticas"
    )
    data["experience"][0]["company"] = long_org
    data["experience"][0]["location"] = None
    data["education"][0]["degree"] = (
        "Doctorado en Estadística con mención en Inferencia Bayesiana "
        "No Paramétrica y Modelos Jerárquicos"
    )
    candidate = Candidate.model_validate(data)
    tex = render_cv_tex(candidate, project_root / "templates", style=CvStyle.MODERNCV)

    assert r"p{\dimexpr\maincolumnwidth-20em\relax}" in tex
    assert long_org in tex
    assert "Doctorado en Estadística" in tex
    # moderncv owns hyperref; a second \usepackage{hyperref} option-clashes
    assert r"\usepackage{hyperref}" not in tex
