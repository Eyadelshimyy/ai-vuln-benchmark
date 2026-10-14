from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlsplit

from rox_dox.cli import main


def test_root_domain_feature_tree_and_breadcrumbs_render_on_every_page(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    site_pages: list[dict[str, object]],
    plantuml_jar: Path,
) -> None:
    repo, _ = git_repo
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    output_dir = tmp_path / "site"
    filenames = ("root.json", "domain.json", "feature.json")
    for filename, payload in zip(filenames, site_pages, strict=True):
        (pages_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "build",
            str(pages_dir),
            "--repo",
            str(repo),
            "--repo-url",
            "https://github.com/Rox-AI/rox-core",
            "--out",
            str(output_dir),
            "--plantuml-jar",
            str(plantuml_jar),
        ]
    )

    assert exit_code == 0
    root = (output_dir / "rox-core.html").read_text(encoding="utf-8")
    domain_file = output_dir / "rox-core" / "pkg.html"
    feature_file = output_dir / "rox-core" / "pkg" / "sub.html"
    domain = domain_file.read_text(encoding="utf-8")
    feature = feature_file.read_text(encoding="utf-8")
    rendered_pages = (
        (
            output_dir / "rox-core.html",
            root,
            "rox-core",
            {"rox-core.html", "rox-core/pkg.html", "rox-core/pkg/sub.html"},
            1,
        ),
        (
            domain_file,
            domain,
            "pkg",
            {"../rox-core.html", "pkg.html", "pkg/sub.html"},
            2,
        ),
        (
            feature_file,
            feature,
            "sub",
            {"../../rox-core.html", "../pkg.html", "sub.html"},
            2,
        ),
    )
    for (
        html_file,
        document,
        current_title,
        expected_hrefs,
        expanded_levels,
    ) in rendered_pages:
        nav = re.search(
            r'<nav id="feature-tree" aria-label="Feature tree">(.*?)</nav>',
            document,
            re.DOTALL,
        )
        assert nav is not None
        rows = re.findall(
            r'<a class="row ([^"]+)" href="([^"]+)"[^>]*>(.*?)</a>',
            nav.group(1),
            re.DOTALL,
        )
        labels = [
            unescape(re.search(r"<span>(.*?)</span>", markup, re.DOTALL).group(1))
            for _classes, _href, markup in rows
        ]
        assert labels == ["rox-core", "pkg", "sub"]
        assert {href for _classes, href, _markup in rows} == expected_hrefs
        current_labels = [
            unescape(re.search(r"<span>(.*?)</span>", markup, re.DOTALL).group(1))
            for classes, _href, markup in rows
            if "current" in classes.split()
        ]
        assert current_labels == [current_title]
        assert nav.group(1).count("<details open>") == expanded_levels
        assert 'aria-current="page"' in nav.group(1)
        for href in re.findall(r'<a\b[^>]*href="([^"]+)"', document):
            parsed = urlsplit(unescape(href))
            if parsed.scheme or not parsed.path:
                continue
            target = (html_file.parent / unquote(parsed.path)).resolve()
            assert target.is_file(), f"{html_file}: relative href {href}"

    assert _breadcrumb_text(root) == "rox-core"
    assert _breadcrumb_text(domain) == "rox-core › pkg"
    assert _breadcrumb_text(feature) == "rox-core › pkg › sub"


def _breadcrumb_text(document: str) -> str:
    breadcrumb = re.search(r'<nav class="breadcrumbs".*?</nav>', document, re.DOTALL)
    assert breadcrumb is not None
    return unescape(re.sub(r"<[^>]+>", "", breadcrumb.group(0)))
