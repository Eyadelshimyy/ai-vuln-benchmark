"""LinkedIn publications package tests."""

from jobbot.adapters.linkedin.package import (
    build_linkedin_publication_items,
    doi_url,
    publications_missing_from_remote,
)
from jobbot.adapters.linkedin.publications import (
    profile_publications_new_url,
    vanity_from_linkedin_url,
)
from jobbot.models.candidate import Candidate, PersonalInfo
from jobbot.models.skill import Publication


def test_doi_url() -> None:
    assert doi_url("10.3390/e28020207") == "https://doi.org/10.3390/e28020207"
    assert doi_url("https://doi.org/10.1/x") == "https://doi.org/10.1/x"
    assert doi_url(None) is None


def test_build_linkedin_publication_items_coauthors() -> None:
    cand = Candidate(
        personal=PersonalInfo(name="Leonardo Andrés Jofré Flor", headline="DS"),
        publications=[
            Publication(
                id="p1",
                title="Efficient EM Estimation for the Pogit Model via Polya-Gamma Augmentation",
                journal="Entropy",
                year=2026,
                doi="10.3390/e28020207",
                authors=["Iván Gutiérrez", "Sandra Ramírez", "Leonardo Jofré"],
            )
        ],
    )
    items = build_linkedin_publication_items(cand)
    assert len(items) == 1
    assert items[0].url == "https://doi.org/10.3390/e28020207"
    assert items[0].publisher == "Entropy"
    assert items[0].coauthors == ("Iván Gutiérrez", "Sandra Ramírez")


def test_publications_missing_from_remote() -> None:
    cand = Candidate(
        personal=PersonalInfo(name="Leonardo Jofré", headline="DS"),
        publications=[
            Publication(id="a", title="Alpha Paper", doi="10.1/a", authors=["X", "Leonardo Jofré"]),
            Publication(id="b", title="Beta Paper", doi="10.1/b", authors=["Leonardo Jofré"]),
        ],
    )
    items = build_linkedin_publication_items(cand)
    missing = publications_missing_from_remote(items, ["alpha paper"])
    assert [m.id for m in missing] == ["b"]


def test_vanity_and_new_url() -> None:
    assert vanity_from_linkedin_url("https://www.linkedin.com/in/leonardojofre") == "leonardojofre"
    assert "leonardojofre/edit/forms/publication/new/" in profile_publications_new_url(
        "leonardojofre"
    )
