from __future__ import annotations

import copy
import re
from pathlib import Path

from rox_dox.model import Page
from rox_dox.render import render_page
from rox_dox.schema import extract_tables
from rox_dox.tree import build_tree

REPO_URL = "https://github.com/Rox-AI/rox-core"
SECTION_IDS = [
    "related",
    "tldr",
    "block",
    "schema",
    "sequences",
]
DIAGRAM_SECTION_IDS = ("block", "schema", "sequences")
EMPTY_MESSAGE = "Nothing to show for this module."


def _section_fragment(document: str, section_id: str) -> str:
    match = re.search(
        rf'<section id="{re.escape(section_id)}">.*?</section>',
        document,
        re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def test_complete_page_has_ordered_sections_and_inline_linked_diagrams(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["tldr"]["summary"][0]["text"] += (
        " More details at https://content.example.test/docs"
    )
    payload["block"]["nodes"][0]["label"] += " https://api.example.test"
    page = Page.model_validate(payload)
    tables = extract_tables(repo, commit)
    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables=tables,
        jar=plantuml_jar,
    )

    assert re.findall(r'<section id="([^"]+)">', document) == SECTION_IDS
    assert '</header><section id="related">' in document
    assert f"verified at <code>{page.commit[:10]}</code>" in document
    assert "https&#58;//content.example.test/docs" in document
    assert "https&#58;//api.example.test" in document
    related = _section_fragment(document, "related")
    assert re.search(
        r"<details open><summary><h2>Related</h2></summary>",
        related,
    )
    for section_id in DIAGRAM_SECTION_IDS:
        section = _section_fragment(document, section_id)
        assert "<svg" in section
        assert '<details class="sources">' in section
    assert "block node api" in _section_fragment(document, "block")
    assert "SQL table users" in _section_fragment(document, "schema")
    assert "NoSQL store cache" in _section_fragment(document, "schema")
    assert "sequence &#x27;send message&#x27; participant api" in _section_fragment(
        document,
        "sequences",
    )
    sidebar = re.search(r'<div id="sidebar">(.*?)</div>', document, re.DOTALL)
    assert sidebar is not None
    sidebar_html = sidebar.group(1)
    toc_position = sidebar_html.index('<nav id="toc" aria-label="On this page">')
    explorer_position = sidebar_html.index(
        '<nav id="site-nav" aria-label="Site navigation">'
    )
    assert toc_position < explorer_position
    toc = re.search(
        r'<nav id="toc" aria-label="On this page">(.*?)</nav>',
        sidebar_html,
        re.DOTALL,
    )
    assert toc is not None
    assert re.findall(r'href="#([^"]+)"', toc.group(1)) == [
        "related",
        "tldr",
        "block",
        "schema",
        "sequences",
    ]

    lowered = document.lower()
    assert "<script src" not in lowered
    assert "<link" not in lowered
    assert "@import" not in lowered
    assert '<img src="http' not in lowered
    anchor_tags = list(re.finditer(r"<a\b[^>]*>", document))
    for match in re.finditer(r"https?://", document):
        anchor = next(
            (tag for tag in anchor_tags if tag.start() <= match.start() < tag.end()),
            None,
        )
        assert anchor is not None
        assert re.search(r'\bhref=["\']', anchor.group(0))


def test_empty_sections_remain_with_explanatory_message(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"] = []
    payload["block"]["edges"] = []
    payload["data"]["sql_tables"] = []
    payload["data"]["nosql"] = []
    payload["sequences"] = []
    payload["related"] = []
    page = Page.model_validate(payload)

    document = render_page(
        page,
        tree=build_tree([page]),
        repo_url=REPO_URL,
        tables={},
        jar=plantuml_jar,
    )

    assert re.findall(r'<section id="([^"]+)">', document) == SECTION_IDS
    for section_id in ("block", "schema", "sequences", "related"):
        assert EMPTY_MESSAGE in _section_fragment(document, section_id)
