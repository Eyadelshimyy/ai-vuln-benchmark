"""HTML fixture parsing for portal profiles."""

from pathlib import Path

from jobbot.adapters.html_parse import parse_indeed_profile_html, parse_linkedin_profile_html


def test_parse_indeed_fixture(project_root: Path) -> None:
    html = (project_root / "tests/fixtures/indeed_profile.html").read_text(encoding="utf-8")
    profile = parse_indeed_profile_html(html)
    assert profile.headline == "Data Scientist"
    assert profile.experience[0].company == "Mercado Libre"
    assert "Python" in profile.skills


def test_parse_indeed_resume_sections(project_root: Path) -> None:
    """Live resume DOM has no per-item testids: entries hang off the section container."""
    html = (project_root / "tests/fixtures/indeed_resume_sections.html").read_text(encoding="utf-8")
    profile = parse_indeed_profile_html(html)
    assert [(e.company, e.title) for e in profile.experience] == [
        ("Acme Analytics", "Senior Data Scientist"),
        # Entities must be unescaped or the diff keeps re-adding roles that are already there.
        ("Nimbus Labs", "Investigador & Data Engineer"),
        # Entries without an edit button (repeated roles) still name their employer.
        ("Instituto Ejemplo", "Profesor de Estadística"),
    ]
    assert profile.experience[0].location == "Santiago de Chile, Región Metropolitana"
    assert profile.experience[0].description
    assert profile.experience[1].location is None
    assert [(e.institution, e.degree) for e in profile.education] == [
        ("Universidad Ejemplo de Chile", "Magíster en Estadística"),
    ]
    # Single-letter skills are real (R, C) and must survive, or the diff re-adds them forever.
    assert profile.skills == ["Python", "SQL", "R"]


def test_parse_linkedin_fixture(project_root: Path) -> None:
    html = (project_root / "tests/fixtures/linkedin_profile.html").read_text(encoding="utf-8")
    profile = parse_linkedin_profile_html(html)
    assert profile.source == "linkedin"
    assert profile.headline == "Data Scientist"
    assert profile.summary
