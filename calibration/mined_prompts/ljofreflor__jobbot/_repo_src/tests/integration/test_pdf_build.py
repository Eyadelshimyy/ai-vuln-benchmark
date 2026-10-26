"""Optional PDF build when XeLaTeX is available."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from jobbot.cv.build import BuildTarget, build_cv
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import sample_profile_dict

pytestmark = pytest.mark.integration

_OVERFULL_HBOX = re.compile(r"Overfull \\hbox")


@pytest.mark.skipif(shutil.which("xelatex") is None, reason="xelatex not installed")
def test_build_pdf(project_root: Path, tmp_path: Path) -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    outputs = build_cv(
        candidate=candidate,
        templates_dir=project_root / "templates",
        output_dir=tmp_path / "output",
        target=BuildTarget.CV,
    )
    names = {p.name for p in outputs}
    assert "cv.tex" in names
    assert "cv.pdf" in names
    assert (tmp_path / "output" / "base" / "cv.pdf").is_file()


@pytest.mark.skipif(shutil.which("xelatex") is None, reason="xelatex not installed")
def test_build_pdf_long_spanish_org_has_no_overfull_hbox(
    project_root: Path, tmp_path: Path
) -> None:
    """Long institutional names must wrap inside banking \\cventry (#112)."""
    data = sample_profile_dict()
    data["experience"][0]["company"] = (
        "Ministerio de Ciencia, Tecnología, Conocimiento e Innovación — "
        "Subsecretaría de Ciencia — Departamento de Estudios y Estadísticas "
        "de la División de Políticas Públicas"
    )
    data["experience"][0]["location"] = None
    data["education"][0]["degree"] = (
        "Doctorado en Estadística con mención en Inferencia Bayesiana "
        "No Paramétrica aplicada a modelos jerárquicos espaciales"
    )
    candidate = Candidate.model_validate(data)
    outputs = build_cv(
        candidate=candidate,
        templates_dir=project_root / "templates",
        output_dir=tmp_path / "output",
        target=BuildTarget.CV,
    )
    out_dir = tmp_path / "output" / "base"
    assert (out_dir / "cv.pdf").is_file()
    log = (out_dir / "cv.log").read_text(encoding="utf-8", errors="replace")
    assert not _OVERFULL_HBOX.search(log), (
        "expected no Overfull \\hbox with long Spanish institution names; "
        f"log excerpt:\n{log[-2000:]}"
    )
    assert any(p.name == "cv.tex" for p in outputs)
