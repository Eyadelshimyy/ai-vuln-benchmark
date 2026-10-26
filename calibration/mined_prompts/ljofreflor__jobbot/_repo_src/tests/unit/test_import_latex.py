"""Tests for moderncv LaTeX importer."""

from __future__ import annotations

from pathlib import Path

from jobbot.models.candidate import Candidate
from jobbot.profile.importer_latex import import_latex_cv, write_generated_profile
from jobbot.profile.loader import load_profile
from jobbot.profile.validator import validate_candidate


def test_import_moderncv_fixture(project_root: Path, tmp_path: Path) -> None:
    tex = project_root / "tests" / "fixtures" / "moderncv_sample.tex"
    result = import_latex_cv(tex)

    assert result.data["personal"]["name"] == "Ana Ejemplo"
    assert "Senior Data Scientist" in result.data["personal"]["headline"]
    assert result.data["personal"]["email"] == "ana.ejemplo@example.com"
    assert result.data["personal"]["linkedin"].endswith("/ana-ejemplo")
    assert result.data["summary"]
    assert "causal inference" in result.data["summary"].lower()

    experiences = result.data["experience"]
    assert len(experiences) == 2
    thoughtworks = next(e for e in experiences if "Thoughtworks" in e["company"])
    assert thoughtworks["current"] is True
    assert thoughtworks["start_date"] == "2025-10"
    assert "end_date" not in thoughtworks or thoughtworks.get("end_date") is None

    meli = next(e for e in experiences if "Mercado Libre" in e["company"])
    assert meli["start_date"] == "2022-04"
    assert meli["end_date"] == "2025-05"
    assert len(meli["achievements"]) == 2
    assert any("Share of Wallet" in a["text"] for a in meli["achievements"])

    education = result.data["education"]
    assert len(education) == 2
    assert any("Magíster" in e["degree"] for e in education)

    skills = result.data["skills"]
    assert "Python" in skills["lenguajes"]
    assert "XGBoost" in skills["ml_frameworks"]
    assert "GCP" in skills["big_data_cloud"]
    assert "Causal Inference" in skills["metodologias"]

    pubs = result.data["publications"]
    assert len(pubs) == 2
    assert pubs[0]["year"] == 2022
    assert pubs[0]["doi"] == "10.1038/s41598-022-13743-8"
    assert "doi" not in pubs[1]

    out = tmp_path / "profile.generated.yaml"
    write_generated_profile(result, out)
    candidate = load_profile(out)
    assert validate_candidate(candidate).ok
    assert isinstance(candidate, Candidate)


def test_parse_present_and_year_only_dates(project_root: Path) -> None:
    tex = project_root / "tests" / "fixtures" / "moderncv_sample.tex"
    result = import_latex_cv(tex)
    phd = next(e for e in result.data["education"] if "Doctorado" in e["degree"])
    assert phd["start_date"] == "2020-01"
    assert "end_date" not in phd


def test_nested_itemize_does_not_leak_the_environment_name(tmp_path: Path) -> None:
    """The closing tag search stopped at the inner list, leaving a bare \\begin.

    The orphan command then became the word 'itemize' at the end of a bullet, and
    that word travelled all the way into the exported CV.
    """
    tex = tmp_path / "cv.tex"
    tex.write_text(
        r"""
\documentclass{moderncv}
\begin{document}
\section{Experiencia}
\cventry{2022--2025}{Analista}{Empresa Neutra}{Ciudad}{}{}
\begin{itemize}
  \item Construí modelos de propensión para varios canales:
  \begin{itemize}
    \item Canal directo: mejora sostenida.
    \item Canal socios: mejora sostenida.
  \end{itemize}
  \item Documenté el proceso de despliegue.
\end{itemize}
\end{document}
""",
        encoding="utf-8",
    )

    result = import_latex_cv(tex)

    bullets = [a["text"] for a in result.data["experience"][0]["achievements"]]
    assert bullets, "the itemize block after the cventry was not read at all"
    for text in bullets:
        assert "itemize" not in text, f"LaTeX environment leaked into: {text!r}"
    # The bullet that introduced the nested list keeps its own sentence.
    assert any(text.rstrip().endswith("canales:") for text in bullets)
    # And the bullet after the nested list is not lost.
    assert any("Documenté el proceso" in text for text in bullets)


def test_spanish_skill_section_labels_are_kept(tmp_path: Path) -> None:
    """A Spanish CV must not be rewritten into English group keys on import."""
    tex = tmp_path / "cv.tex"
    tex.write_text(
        r"""
\documentclass{moderncv}
\begin{document}
\section{Habilidades}
\cvitem{Lenguajes}{Python, SQL}
\cvitem{ML Frameworks}{PyTorch}
\cvitem{Big Data \& Cloud}{GCP}
\cvitem{Metodologías}{Experimentación}
\end{document}
""",
        encoding="utf-8",
    )
    result = import_latex_cv(tex)
    skills = result.data["skills"]
    assert "lenguajes" in skills
    assert "programming" not in skills
    assert "ml_frameworks" in skills
    assert "machine_learning" not in skills
    assert "big_data_cloud" in skills
    assert "metodologias" in skills
    assert "statistics" not in skills
