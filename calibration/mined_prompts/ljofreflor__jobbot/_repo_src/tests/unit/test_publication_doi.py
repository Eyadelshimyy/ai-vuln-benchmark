"""Publication DOI enrichment helpers."""

from jobbot.models.skill import Publication
from jobbot.publications.doi import DoiMetadata, _parse_work


def test_parse_crossref_work_extracts_authors_and_doi() -> None:
    message = {
        "DOI": "10.3390/e28020207",
        "title": ["Efficient EM Estimation for the Pogit Model via Polya-Gamma Augmentation"],
        "container-title": ["Entropy"],
        "published-print": {"date-parts": [[2026, 2, 11]]},
        "author": [
            {"given": "Iván", "family": "Gutiérrez"},
            {"given": "Sandra", "family": "Ramírez"},
            {"given": "Leonardo", "family": "Jofré"},
        ],
    }
    meta = _parse_work(message)
    assert isinstance(meta, DoiMetadata)
    assert meta.doi == "10.3390/e28020207"
    assert meta.authors == ["Iván Gutiérrez", "Sandra Ramírez", "Leonardo Jofré"]
    assert meta.year == 2026


def test_publication_coauthors_exclude_self() -> None:
    pub = Publication(
        id="p1",
        title="T",
        authors=["Iván Gutiérrez", "Leonardo Jofré", "Sandra Ramírez"],
    )
    assert pub.coauthors("Leonardo Andrés Jofré Flor") == [
        "Iván Gutiérrez",
        "Sandra Ramírez",
    ]


def test_publication_coauthors_work_for_any_surname() -> None:
    """Self-exclusion used to require one hardcoded surname; a namesake still stays."""
    pub = Publication(
        id="p2",
        title="T",
        authors=["Rocío Paredes Lagos", "Rocío Salinas", "Camila Núñez"],
    )

    assert pub.coauthors("Rocío Paredes Lagos") == ["Rocío Salinas", "Camila Núñez"]
