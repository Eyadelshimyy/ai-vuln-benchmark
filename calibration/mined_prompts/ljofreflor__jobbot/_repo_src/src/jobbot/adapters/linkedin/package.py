"""LinkedIn publications sync package from Candidate (facts only)."""

from __future__ import annotations

from dataclasses import dataclass

from jobbot.models.candidate import Candidate
from jobbot.models.skill import Publication


@dataclass(frozen=True)
class LinkedInPublicationItem:
    """One publication ready for LinkedIn Publications form."""

    id: str
    title: str
    publisher: str | None
    year: int | None
    url: str | None
    authors: tuple[str, ...]
    coauthors: tuple[str, ...]


def doi_url(doi: str | None) -> str | None:
    if not doi:
        return None
    cleaned = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/").strip()
    if not cleaned:
        return None
    return f"https://doi.org/{cleaned}"


def build_linkedin_publication_items(
    candidate: Candidate,
) -> list[LinkedInPublicationItem]:
    """Map profile publications → LinkedIn form fields (official DOI metadata)."""
    name = candidate.personal.name
    items: list[LinkedInPublicationItem] = []
    for pub in candidate.publications:
        items.append(_from_publication(pub, self_name=name))
    return items


def _from_publication(pub: Publication, *, self_name: str) -> LinkedInPublicationItem:
    authors = tuple(pub.authors)
    coauthors = tuple(pub.coauthors(self_name))
    return LinkedInPublicationItem(
        id=pub.id,
        title=pub.title,
        publisher=pub.journal,
        year=pub.year,
        url=doi_url(pub.doi) or (str(pub.url) if pub.url else None),
        authors=authors,
        coauthors=coauthors,
    )


def publications_missing_from_remote(
    local: list[LinkedInPublicationItem],
    remote_titles: list[str],
) -> list[LinkedInPublicationItem]:
    """Return local pubs whose title is not present on LinkedIn (casefold match)."""
    seen = {_norm_title(t) for t in remote_titles}
    return [item for item in local if _norm_title(item.title) not in seen]


def _norm_title(title: str) -> str:
    return " ".join(title.casefold().split())
