"""Publication.url: canonical link when there is no DOI (#253)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jobbot.adapters.linkedin.package import build_linkedin_publication_items
from jobbot.cv.renderer import CvStyle, render_cv_ats, render_cv_tex
from jobbot.models.candidate import Candidate, PersonalInfo
from jobbot.models.skill import Publication
from tests.fixtures.profile import sample_profile_dict


def _pub(**kwargs: object) -> Publication:
    base: dict[str, object] = {
        "id": "pub-note",
        "title": "Technical note on retail demand",
        "journal": "Institutional repository",
        "year": 2020,
    }
    base.update(kwargs)
    return Publication.model_validate(base)


def _candidate_with(pub: Publication) -> Candidate:
    data = sample_profile_dict()
    data["publications"] = [pub.model_dump(mode="json")]
    return Candidate.model_validate(data)


def test_publication_accepts_url_without_doi() -> None:
    pub = _pub(url="https://example.org/repos/tech-note")
    assert pub.doi is None
    assert str(pub.url) == "https://example.org/repos/tech-note"


def test_publication_rejects_non_url() -> None:
    with pytest.raises(ValidationError):
        _pub(url="no-es-url")


def test_cv_templates_use_url_when_no_doi(project_root: Path) -> None:
    candidate = _candidate_with(_pub(url="https://example.org/repos/tech-note"))
    templates = project_root / "templates"

    plain = render_cv_tex(candidate, templates, style=CvStyle.PLAIN)
    modern = render_cv_tex(candidate, templates, style=CvStyle.MODERNCV)
    ats = render_cv_ats(candidate, templates)

    for text in (plain, modern, ats):
        assert "https://example.org/repos/tech-note" in text
        assert "doi.org" not in text


def test_cv_templates_prefer_doi_over_url(project_root: Path) -> None:
    candidate = _candidate_with(
        _pub(doi="10.1234/example.note", url="https://example.org/repos/tech-note")
    )
    templates = project_root / "templates"

    plain = render_cv_tex(candidate, templates, style=CvStyle.PLAIN)
    ats = render_cv_ats(candidate, templates)

    assert "doi:10.1234/example.note" in plain
    assert "doi.org/10.1234/example.note" in plain
    assert "example.org/repos" not in plain
    assert "doi:10.1234/example.note" in ats
    assert "example.org/repos" not in ats


def test_linkedin_items_use_url_when_no_doi() -> None:
    cand = Candidate(
        personal=PersonalInfo(name="Ana Ejemplo", headline="DS"),
        publications=[_pub(url="https://example.org/repos/tech-note")],
    )
    items = build_linkedin_publication_items(cand)
    assert items[0].url == "https://example.org/repos/tech-note"


def test_linkedin_items_prefer_doi_over_url() -> None:
    cand = Candidate(
        personal=PersonalInfo(name="Ana Ejemplo", headline="DS"),
        publications=[
            _pub(doi="10.1234/example.note", url="https://example.org/repos/tech-note")
        ],
    )
    items = build_linkedin_publication_items(cand)
    assert items[0].url == "https://doi.org/10.1234/example.note"
