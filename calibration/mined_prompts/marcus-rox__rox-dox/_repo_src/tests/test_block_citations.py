import json
from pathlib import Path

import pytest

from rox_dox.cli import main
from rox_dox.model import Page
from rox_dox.tree import build_tree, tree_problems


def test_stale_node_detail_source_fails_check_with_its_label(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["details"] = [
        {
            "text": "Handles requests",
            "sources": [{"path": "pkg/a.py", "lines": [8, 20]}],
        }
    ]
    (pages_dir / "root.json").write_text(json.dumps(page_data), encoding="utf-8")

    exit_code = main(["check", str(pages_dir), "--repo", str(git_repo[0])])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "block node api detail 1" in output
    assert "pkg/a.py" in output


def test_stale_group_source_fails_check_with_its_label(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    block = page_data["block"]
    assert isinstance(block, dict)
    block["groups"] = [
        {
            "id": "backend",
            "label": "Backend",
            "source": {"path": "pkg/a.py", "lines": [8, 20]},
        }
    ]
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["group"] = "backend"
    (pages_dir / "root.json").write_text(json.dumps(page_data), encoding="utf-8")

    exit_code = main(["check", str(pages_dir), "--repo", str(git_repo[0])])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "block group backend" in output
    assert "pkg/a.py" in output


def test_tree_supports_grouped_nodes_and_checks_their_page_links(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    block["groups"] = [
        {
            "id": "backend",
            "label": "Backend",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        }
    ]
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["group"] = "backend"
    node["details"] = [
        {
            "text": "Handles requests",
            "sources": [{"path": "pkg/a.py", "lines": [1, 3]}],
        }
    ]

    page = Page.model_validate(page_data)

    assert tree_problems([page]) == []
    tree = build_tree([page])
    assert tree.root == page.id
    assert tree.pages[page.id] is page

    node["link"] = "missing-page"
    page_with_missing_link = Page.model_validate(page_data)
    assert tree_problems([page_with_missing_link]) == [
        "block node api: link to missing page 'missing-page'"
    ]
