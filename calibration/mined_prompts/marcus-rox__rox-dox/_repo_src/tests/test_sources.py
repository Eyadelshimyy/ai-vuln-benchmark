from pathlib import Path

from rox_dox.model import Page
from rox_dox.sources import page_problems


def test_missing_commit_is_reported(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    page_data["commit"] = "missing-commit"
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert problems == ["commit 'missing-commit' not found in repository"]


def test_missing_source_file_is_reported(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["source"] = {"path": "pkg/missing.py", "lines": [1, 3]}
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert len(problems) == 1
    assert "block node api" in problems[0]
    assert "pkg/missing.py" in problems[0]


def test_page_path_can_be_a_git_blob(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    page_data["paths"] = ["pkg/a.py"]
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert problems == []


def test_membership_source_is_validated(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    page_data["membership"] = [
        {
            "layer": "Web screens",
            "rows": [
                {
                    "path": "web/src/page.tsx",
                    "primary": True,
                    "evidence": "route tag campaigns",
                    "sources": [{"path": "pkg/missing.py", "lines": [1, 1]}],
                }
            ],
        }
    ]
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert problems == [
        f"membership Web screens file web/src/page.tsx: "
        f"source file 'pkg/missing.py' not found at {page.commit[:8]}"
    ]


def test_missing_page_path_is_reported(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    page_data["paths"] = ["pkg/missing"]
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert problems == [f"path 'pkg/missing' not found at {page.commit[:8]}"]


def test_source_range_cannot_start_after_end(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    block = page_data["block"]
    assert isinstance(block, dict)
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["source"] = {"path": "pkg/a.py", "lines": [5, 3]}
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert len(problems) == 1
    assert "block edge api->store" in problems[0]
    assert "invalid line range 5-3" in problems[0]


def test_source_range_must_start_at_one(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    block = page_data["block"]
    assert isinstance(block, dict)
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["source"] = {"path": "pkg/a.py", "lines": [0, 3]}
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert len(problems) == 1
    assert "block edge api->store" in problems[0]
    assert "start before line 1" in problems[0]


def test_notion_source_requires_notion_host(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    tldr = page_data["tldr"]
    assert isinstance(tldr, dict)
    notes = tldr["notes"]
    assert isinstance(notes, list)
    note = notes[0]
    assert isinstance(note, dict)
    note["sources"] = [{"notion": "https://example.com/x"}]
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert len(problems) == 1
    assert "TLDR note: Design notes." in problems[0]
    assert "https://example.com/x" in problems[0]


def test_all_source_problems_are_reported(
    git_repo: tuple[Path, str], page_data: dict[str, object]
) -> None:
    repo, _ = git_repo
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["source"] = {"path": "pkg/missing.py", "lines": [1, 3]}
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["source"] = {"path": "pkg/a.py", "lines": [8, 20]}
    page = Page.model_validate(page_data)

    problems = page_problems(page, repo)

    assert len(problems) == 2
    assert any(
        "block node api" in problem and "pkg/missing.py" in problem
        for problem in problems
    )
    assert any(
        "block edge api->store" in problem and "pkg/a.py" in problem
        for problem in problems
    )
