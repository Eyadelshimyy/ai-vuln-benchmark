from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import rox_dox.cli as cli
from rox_dox.outline import (
    OutlineRow,
    format_markdown,
    outline_source,
    read_worktree,
)


def test_outline_source_reports_only_top_level_objects_in_source_order() -> None:
    source = (
        '"""Module docs."""\n'
        "import os\n"
        "\n"
        "MAX_RETRIES = 3\n"
        "logger = x()\n"
        "\n"
        "class Foo:\n"
        '    """Does foo.\n'
        "\n"
        '    More."""\n'
        "\n"
        "@decorate\n"
        "def bar():\n"
        "    pass\n"
        "\n"
        "async def baz():\n"
        '    """Runs baz."""\n'
    )

    outline = outline_source("example.py", source)

    assert outline.rows == (
        OutlineRow(name="MAX_RETRIES", kind="constant", line=4, doc=""),
        OutlineRow(name="Foo", kind="class", line=7, doc="Does foo."),
        OutlineRow(name="bar", kind="function", line=13, doc=""),
        OutlineRow(name="baz", kind="async_function", line=16, doc="Runs baz."),
    )
    assert outline.empty is False


def test_empty_and_imports_only_files_are_marked_empty() -> None:
    empty = outline_source("empty.py", "")
    imports_only = outline_source(
        "imports.py",
        '"""Imports for this module."""\nimport os\nfrom pathlib import Path\n',
    )

    assert empty.empty is True
    assert empty.to_json() == {"path": "empty.py", "empty": True}
    assert imports_only.empty is True
    assert imports_only.to_json() == {"path": "imports.py", "empty": True}


def test_read_worktree_recurses_in_order_and_skips_tests_and_migrations(
    tmp_path: Path,
) -> None:
    package = tmp_path / "pkg"
    files = {
        package / "a.py": "def a():\n    pass\n",
        package / "sub" / "b.py": "def b():\n    pass\n",
        package / "tests" / "test_x.py": "def test_x():\n    pass\n",
        package / "migrations" / "m.py": "def migration():\n    pass\n",
        package / "notes.txt": "not Python\n",
    }
    for path, source in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    sources = read_worktree([package, package / "a.py"])

    assert list(sources) == [
        (package / "a.py").as_posix(),
        (package / "sub" / "b.py").as_posix(),
    ]


def test_commit_outline_reads_selected_revision_as_json_and_markdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_file = repo / "pkg" / "m.py"
    source_file.parent.mkdir()
    source_file.write_text("def old():\n    pass\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test User"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "pkg/m.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add old function"],
        check=True,
        capture_output=True,
    )
    first_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    source_file.write_text("def new():\n    pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "pkg/m.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Rename function"],
        check=True,
        capture_output=True,
    )

    exit_code = cli.main(
        [
            "outline",
            "pkg",
            "--repo",
            str(repo),
            "--commit",
            first_commit,
            "--json",
        ]
    )
    json_output = capsys.readouterr()
    document = json.loads(json_output.out)

    assert exit_code == 0
    assert json_output.err == ""
    assert document["files"][0]["rows"][0]["name"] == "old"

    exit_code = cli.main(
        ["outline", "pkg", "--repo", str(repo), "--commit", first_commit]
    )
    markdown_output = capsys.readouterr()

    assert exit_code == 0
    assert "| `old` | — |" in markdown_output.out

    exit_code = cli.main(
        ["outline", ".", "--repo", str(repo), "--commit", first_commit, "--json"]
    )
    root_output = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(root_output.out)["files"][0]["path"] == "pkg/m.py"


def test_cli_reports_missing_paths_and_incomplete_commit_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    assert cli.main(["outline", "missing_dir"]) == 2
    missing_path_output = capsys.readouterr()
    assert missing_path_output.out == ""
    assert "path not found: missing_dir" in missing_path_output.err

    assert cli.main(["outline", "pkg", "--commit", "abc123"]) == 2
    option_output = capsys.readouterr()
    assert option_output.out == ""
    assert option_output.err.strip() == "--repo and --commit must be given together"


def test_outline_source_reports_syntax_errors_and_markdown_empty_rows() -> None:
    syntax_error = outline_source("invalid.py", "def invalid(:\n")
    no_rows = outline_source("logger.py", "logger = getLogger()\n")
    empty = outline_source("empty.py", "")

    assert syntax_error.error == "invalid.py:1: invalid syntax"
    assert syntax_error.to_json() == {
        "path": "invalid.py",
        "error": "invalid.py:1: invalid syntax",
    }
    assert format_markdown([syntax_error, no_rows, empty]) == (
        "## `invalid.py`\n\n"
        "Parse error: invalid.py:1: invalid syntax\n\n"
        "## `logger.py`\n\n"
        "No top-level classes, functions or constants.\n\n"
        "## `empty.py`\n\n"
        "Empty or imports only.\n"
    )
