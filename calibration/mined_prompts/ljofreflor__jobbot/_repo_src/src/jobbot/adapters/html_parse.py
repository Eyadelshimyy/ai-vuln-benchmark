"""Parse ExternalProfile from HTML fixtures or live pages."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from jobbot.models.external_profile import (
    ExternalEducation,
    ExternalExperience,
    ExternalProfile,
)


def parse_indeed_profile_html(html: str) -> ExternalProfile:
    soup = _soup(html)
    return ExternalProfile(
        source="indeed",
        headline=_meta(soup, "headline") or _tag_text(soup, "h1"),
        summary=_meta(soup, "summary") or _section(soup, "summary"),
        location=_meta(soup, "location"),
        experience=_experiences(soup),
        education=_educations(soup),
        skills=_list_items(soup, "skills") or _skill_chips(soup),
        captured_at=datetime.now(UTC),
    )


def parse_linkedin_profile_html(html: str) -> ExternalProfile:
    soup = _soup(html)
    return ExternalProfile(
        source="linkedin",
        headline=_meta(soup, "headline") or _tag_text(soup, "h1"),
        summary=_meta(soup, "about") or _section(soup, "about"),
        location=_meta(soup, "location"),
        experience=_experiences(soup),
        education=_educations(soup),
        skills=_list_items(soup, "skills"),
        captured_at=datetime.now(UTC),
    )


def load_external_from_file(path: Path, source: str) -> ExternalProfile:
    html = path.read_text(encoding="utf-8")
    if source == "indeed":
        return parse_indeed_profile_html(html)
    return parse_linkedin_profile_html(html)


def _soup(html: str) -> BeautifulSoup:
    """stdlib parser: no compiled dependency, good enough for these pages."""
    return BeautifulSoup(html, "html.parser")


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    tagged = soup.select_one(f'[data-testid="{name}"]')
    if tagged is not None:
        return _clean(tagged.get_text(" ")) or None
    meta = soup.find("meta", attrs={"name": name})
    if isinstance(meta, Tag):
        content = meta.get("content")
        if isinstance(content, str):
            return content.strip() or None
    return None


def _tag_text(soup: BeautifulSoup, tag: str) -> str | None:
    found = soup.find(tag)
    return _clean(found.get_text(" ")) or None if isinstance(found, Tag) else None


def _section(soup: BeautifulSoup, section_id: str) -> str | None:
    found = soup.find(id=section_id)
    return _clean(found.get_text(" ")) or None if isinstance(found, Tag) else None


def _list_items(soup: BeautifulSoup, section_id: str) -> list[str]:
    found = soup.find(id=section_id)
    if not isinstance(found, Tag):
        return []
    items = [_clean(li.get_text(" ")) for li in found.find_all("li")]
    return [item for item in items if item]


def _experiences(soup: BeautifulSoup) -> list[ExternalExperience]:
    result = [
        ExternalExperience(
            title=_testid_text(item, "title"),
            company=_testid_text(item, "company"),
            location=_testid_text(item, "location"),
            description=_testid_text(item, "description"),
        )
        for item in soup.select('[data-testid="experience-item"]')
    ]
    if result:
        return result
    for title, lines in _section_entries(soup, "work-experience-section"):
        company, location = _split_bullet(lines[0] if lines else None)
        result.append(
            ExternalExperience(
                title=title,
                company=company,
                location=location,
                description=lines[2] if len(lines) > 2 else None,
            )
        )
    return result


def _educations(soup: BeautifulSoup) -> list[ExternalEducation]:
    result = [
        ExternalEducation(
            degree=_testid_text(item, "degree"),
            institution=_testid_text(item, "institution"),
        )
        for item in soup.select('[data-testid="education-item"]')
    ]
    if result:
        return result
    for degree, lines in _section_entries(soup, "education-section"):
        institution, _ = _split_bullet(lines[0] if lines else None)
        result.append(ExternalEducation(degree=degree, institution=institution))
    return result


def _section_entries(soup: BeautifulSoup, testid: str) -> list[tuple[str, list[str]]]:
    """Entries of a resume section: each h3 heading plus the leaf texts under it."""
    section = soup.select_one(f'[data-testid="{testid}"]')
    if section is None:
        return []
    entries: list[tuple[str, list[str]]] = []
    for heading in section.select("h3"):
        title = _clean(heading.get_text(" "))
        if not title:
            continue
        entries.append((title, _leaf_texts_after(heading)))
    return entries


def _skill_chips(soup: BeautifulSoup) -> list[str]:
    """Skills render as edit chips; keep them verbatim (single letters like R are real)."""
    section = soup.select_one('[data-testid="skills-section"]')
    if section is None:
        return []
    chips: list[str] = []
    for chip in section.select('[data-testid^="edit-chip-"]'):
        label = next((text for text in _leaf_texts(chip) if text), None)
        if label:
            chips.append(label)
    return chips


def _leaf_texts_after(heading: Tag) -> list[str]:
    """Leaf divs that follow a heading inside its own entry, in document order."""
    entry = _entry_container(heading)
    if entry is None:
        return []
    following = {id(div) for div in heading.find_all_next("div")}
    return [
        text
        for div in entry.find_all("div")
        if id(div) in following and (text := _leaf_text(div))
    ]


def _entry_container(heading: Tag) -> Tag | None:
    """Smallest ancestor holding the heading plus its detail lines.

    Editable entries are wrapped in a button, but repeated roles at one employer
    render as plain divs whose company line is a sibling of the heading's parent.
    """
    button = heading.find_parent("button")
    if button is not None:
        return button
    node = heading.parent
    while isinstance(node, Tag):
        if len(node.select("h3")) > 1:
            break
        if any(_leaf_text(div) for div in node.find_all("div") if div.find("h3") is None):
            return node
        node = node.parent
    return heading.parent


def _leaf_texts(scope: Tag) -> list[str]:
    return [text for div in scope.find_all("div") if (text := _leaf_text(div))]


def _leaf_text(div: Tag) -> str:
    """Text of a div that holds no further elements (a real leaf of the layout)."""
    if div.find(True) is not None:
        return ""
    return _clean(div.get_text(" "))


def _testid_text(scope: Tag, name: str) -> str | None:
    found = scope.select_one(f'[data-testid="{name}"]')
    return _clean(found.get_text(" ")) or None if found is not None else None


def _split_bullet(line: str | None) -> tuple[str | None, str | None]:
    if not line:
        return None, None
    head, _, tail = line.partition("•")
    head = head.strip()
    tail = tail.strip()
    return (head or None), (tail or None)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
