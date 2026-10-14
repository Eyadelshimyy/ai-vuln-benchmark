from __future__ import annotations

import json
import re
import subprocess
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlsplit

from rox_dox.cli import main
from rox_dox.feature_pages import feature_pages
from rox_dox.features import build_feature_map
from rox_dox.model import Page
from rox_dox.schema import extract_tables


def test_R7_folder_pages_use_local_explorer_links_and_show_coverage(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    repo, _ = git_repo
    additions = {
        "backend/src/pkg/sequence.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Sequence:\n"
            '    __tablename__ = "sequence"\n'
            "    id = Column(Integer, primary_key=True)\n"
        ),
        "backend/src/pkg/sequence_service.py": (
            "from .sequence import Sequence\n"
            "\n"
            "def get_sequence() -> Sequence:\n"
            "    return Sequence()\n"
        ),
        "backend/src/pkg/isolated.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Isolated:\n"
            '    __tablename__ = "isolated"\n'
            "    id = Column(Integer, primary_key=True)\n"
        ),
        "backend/src/pkg/isolated_service.py": (
            "from .isolated import Isolated\n"
            "\n"
            "def get_isolated() -> Isolated:\n"
            "    return Isolated()\n"
        ),
        "backend/src/uncovered/orphan.py": "value = 1\n",
        "backend/src/pkg/tests/ignored.py": "value = 2\n",
    }
    for relative_path, source in additions.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", *additions],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add folder page fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    feature_map = build_feature_map(
        repo,
        commit,
        {"seq": ["isolated", "sequence"]},
        "seq",
    )
    tables = extract_tables(repo, commit)
    generated_pages = feature_pages(
        feature_map,
        root_id="rox-core",
        domain_title="Sequences",
        names={"isolated": "Isolated", "sequence": "Sequences"},
        tables=tables,
    )
    root_payload = dict(page_data)
    root_payload["commit"] = commit
    root_payload["kind"] = "root"
    root_page = Page.model_validate(root_payload)

    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "root.json").write_text(
        json.dumps(root_page.model_dump()),
        encoding="utf-8",
    )
    for page in generated_pages:
        (pages_dir / f"{page.id}.json").write_text(
            json.dumps(page.model_dump(exclude_none=True)),
            encoding="utf-8",
        )
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    (features_dir / "seq.json").write_text(
        json.dumps(feature_map.model_dump(exclude_none=True)),
        encoding="utf-8",
    )

    output_dir = tmp_path / "site"
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
            "--features",
            str(features_dir),
        ]
    )

    assert exit_code == 0
    folder_pages = sorted((output_dir / "folders").rglob("*.html"))
    assert folder_pages
    for html_file in output_dir.rglob("*.html"):
        document = html_file.read_text(encoding="utf-8")
        explorer = re.search(
            r'<nav id="explorer".*?</nav>',
            document,
            re.DOTALL,
        )
        assert explorer is not None
        folder_hrefs = re.findall(
            r'<a class="row folder"[^>]*href="([^"]+)"',
            explorer.group(),
        )
        for href in folder_hrefs:
            parsed = urlsplit(unescape(href))
            assert parsed.netloc != "github.com"
            if parsed.scheme or not parsed.path:
                continue
            target = (html_file.parent / unquote(parsed.path)).resolve()
            assert target.is_file(), f"{html_file}: {href}"

    for folder_page in folder_pages:
        document = folder_page.read_text(encoding="utf-8")
        assert "github.com" not in document.lower()
        explorer = re.search(
            r'<nav id="explorer".*?</nav>',
            document,
            re.DOTALL,
        )
        assert explorer is not None
        for href in re.findall(r'<a\b[^>]*href="([^"]+)"', explorer.group()):
            parsed = urlsplit(unescape(href))
            assert parsed.netloc != "github.com"
            if parsed.scheme or not parsed.path:
                continue
            target = (folder_page.parent / unquote(parsed.path)).resolve()
            assert target.is_file(), f"{folder_page}: {href}"

    multi_feature_folder = output_dir / "folders/backend/src/pkg.html"
    document = multi_feature_folder.read_text(encoding="utf-8")
    coverage = re.search(
        r"<section><h2>Feature coverage</h2>(.*?)</section>",
        document,
        re.DOTALL,
    )
    assert coverage is not None
    feature_hrefs = set(
        re.findall(
            r'<tr><td><a href="([^"]+)">(?:Isolated|Sequences)</a></td>',
            coverage.group(),
        )
    )
    assert len(feature_hrefs) == 2
    assert {
        (multi_feature_folder.parent / unquote(href)).resolve()
        for href in feature_hrefs
    } == {
        output_dir / "feature-seq-isolated.html",
        output_dir / "feature-seq-sequence.html",
    }

    uncovered_folder = output_dir / "folders/backend/src/uncovered.html"
    uncovered_document = uncovered_folder.read_text(encoding="utf-8")
    uncovered_href = re.search(
        r'<a href="([^"]+)">listed in the uncovered inventory</a>',
        uncovered_document,
    )
    assert uncovered_href is not None
    assert (
        uncovered_folder.parent / unquote(uncovered_href.group(1))
    ).resolve() == output_dir / "uncovered.html"

    tests_folder = output_dir / "folders/backend/src/pkg/tests.html"
    assert "Tests and migrations are not documented." in tests_folder.read_text(
        encoding="utf-8"
    )
