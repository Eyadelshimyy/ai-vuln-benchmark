"""Relative links in docs/ resolve inside docs/: the Pages build runs MkDocs --strict."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_LINK = re.compile(r"\]\(([^)\s]+)")
_SKIP = re.compile(r"^([a-z][a-z0-9+.\-]*:|#|/)", re.I)


def _doc_pages(project_root: Path) -> list[Path]:
    return sorted((project_root / "docs").rglob("*.md"))


def _broken_links(page: Path, docs: Path) -> list[str]:
    broken = []
    for target in _LINK.findall(page.read_text(encoding="utf-8")):
        if _SKIP.match(target):
            continue
        path = target.split("#", 1)[0]
        if not path:
            continue
        resolved = (page.parent / path).resolve()
        if not resolved.is_file() or docs.resolve() not in resolved.parents:
            broken.append(target)
    return broken


def test_every_relative_doc_link_points_at_a_published_file(project_root: Path) -> None:
    docs = project_root / "docs"
    broken = {
        str(page.relative_to(docs)): links
        for page in _doc_pages(project_root)
        if (links := _broken_links(page, docs))
    }
    assert broken == {}


def test_the_check_catches_links_that_leave_docs(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    (docs / "en").mkdir(parents=True)
    (docs / "instalacion.md").write_text("# ok\n", encoding="utf-8")
    page = docs / "en" / "index.md"
    page.write_text(
        "[a](../instalacion.md) [b](../instalacion/) [c](../../AGENTS.md) "
        "[d](https://example.com) [e](#top)\n",
        encoding="utf-8",
    )

    assert _broken_links(page, docs) == ["../instalacion/", "../../AGENTS.md"]


class _Loader(yaml.SafeLoader):
    """mkdocs.yml may carry ``!!python/name`` tags; their value is irrelevant here."""


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda *_: None)


def _nav_pages(node: object) -> set[str]:
    if isinstance(node, str):
        return {node} if node.endswith(".md") else set()
    if isinstance(node, dict):
        return set().union(*(_nav_pages(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_nav_pages(v) for v in node))
    return set()


def test_every_page_is_in_nav_or_declared_out_of_it(project_root: Path) -> None:
    config = yaml.load((project_root / "mkdocs.yml").read_text(encoding="utf-8"), _Loader)  # noqa: S506
    listed = _nav_pages(config["nav"]) | set(str(config.get("not_in_nav", "")).split())
    docs = project_root / "docs"
    pages = {
        str(p.relative_to(docs))
        for p in _doc_pages(project_root)
        if "blog" not in p.relative_to(docs).parts[:1]
        and "overrides" not in p.relative_to(docs).parts[:1]
    }
    assert pages - listed == set()
