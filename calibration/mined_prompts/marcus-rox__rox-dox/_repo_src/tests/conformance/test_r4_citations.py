import json
from pathlib import Path

import pytest

from rox_dox.cli import main


def test_R4_uncited_arrow_fails_and_names_the_edge(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    block = page_data["block"]
    assert isinstance(block, dict)
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    del edge["source"]
    (pages_dir / "root.json").write_text(json.dumps(page_data), encoding="utf-8")

    exit_code = main(["check", str(pages_dir), "--repo", str(git_repo[0])])

    assert exit_code == 1
    assert "block edge api->store" in capsys.readouterr().out


def test_R4_stale_source_fails_and_names_the_source(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    block = page_data["block"]
    assert isinstance(block, dict)
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["source"] = {"path": "pkg/a.py", "lines": [8, 20]}
    (pages_dir / "root.json").write_text(json.dumps(page_data), encoding="utf-8")

    exit_code = main(["check", str(pages_dir), "--repo", str(git_repo[0])])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "block edge api->store" in output
    assert "pkg/a.py" in output


def test_R4_stale_source_build_fails_without_writing_output(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    plantuml_jar: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    output_dir = tmp_path / "site"
    block = page_data["block"]
    assert isinstance(block, dict)
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["source"] = {"path": "pkg/a.py", "lines": [8, 20]}
    (pages_dir / "root.json").write_text(json.dumps(page_data), encoding="utf-8")

    exit_code = main(
        [
            "build",
            str(pages_dir),
            "--repo",
            str(git_repo[0]),
            "--repo-url",
            "https://github.com/Rox-AI/rox-core",
            "--out",
            str(output_dir),
            "--plantuml-jar",
            str(plantuml_jar),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "block edge api->store" in output
    assert "pkg/a.py" in output
    assert not output_dir.exists()


def test_R4_uncited_arrow_build_fails_without_writing_output(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    plantuml_jar: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    output_dir = tmp_path / "site"
    block = page_data["block"]
    assert isinstance(block, dict)
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    del edge["source"]
    (pages_dir / "root.json").write_text(json.dumps(page_data), encoding="utf-8")

    exit_code = main(
        [
            "build",
            str(pages_dir),
            "--repo",
            str(git_repo[0]),
            "--repo-url",
            "https://github.com/Rox-AI/rox-core",
            "--out",
            str(output_dir),
            "--plantuml-jar",
            str(plantuml_jar),
        ]
    )

    assert exit_code == 1
    assert "block edge api->store" in capsys.readouterr().out
    assert not output_dir.exists()
