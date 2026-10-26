"""The index only helps if it is true: keep it generated, complete and grouped."""

from __future__ import annotations

from pathlib import Path

from jobbot.cli import app
from jobbot.ops.capabilities import (
    INDEX_RELPATH,
    build_index,
    collect_commands,
    collect_modules,
    modules_without_summary,
    render_index,
)
from jobbot.workspace import repo_root

ROOT = repo_root()
SRC = ROOT / "src" / "jobbot"


def test_committed_index_is_up_to_date() -> None:
    committed = (ROOT / INDEX_RELPATH).read_text(encoding="utf-8")
    assert committed == build_index(ROOT, app), (
        "docs/capabilities.md is stale — run `make capabilities` and commit the result"
    )


def test_every_module_states_what_it_is_for() -> None:
    """A module with no docstring becomes a hole in the index."""
    assert modules_without_summary(SRC) == []


def test_index_lists_helpers_that_get_re_derived_otherwise() -> None:
    index = build_index(ROOT, app)
    for symbol in ("first_apply_email", "detect_country", "age_label", "escape_latex"):
        assert f"`{symbol}`" in index


def test_index_lists_commands_with_their_real_names() -> None:
    index = build_index(ROOT, app)
    assert "`jobbot ops capabilities`" in index
    assert "`jobbot linkedin sweep`" in index
    assert "`jobbot ops failure issue`" in index  # nested group, not a flat name


def test_hidden_commands_stay_out() -> None:
    assert "probe-exit" not in build_index(ROOT, app)


def test_cli_commands_are_not_repeated_as_module_symbols() -> None:
    """Typer command functions belong to the command list, not to `cli.py`'s exports."""
    cli_entry = next(entry for entry in collect_modules(SRC) if entry.relpath == "cli.py")
    assert "linkedin_sweep" not in cli_entry.symbols
    assert cli_entry.symbols == ("run_cli",)


def test_each_package_appears_once() -> None:
    """Sorting by full path used to interleave a package with its subpackages."""
    headings = [line for line in build_index(ROOT, app).splitlines() if line.startswith("### ")]
    assert len(headings) == len(set(headings))
    assert "### `adapters`" in headings
    assert "### `adapters/ats`" in headings


def test_index_is_deterministic() -> None:
    assert build_index(ROOT, app) == build_index(ROOT, app)


def test_render_reports_the_size_of_the_surface() -> None:
    modules = collect_modules(SRC)
    commands = collect_commands(app)
    symbols = sum(len(entry.symbols) for entry in modules)
    assert f"{len(commands)} commands, {len(modules)} modules, {symbols} public symbols" in (
        render_index(modules, commands)
    )


def test_collect_modules_skips_empty_package_inits(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""Just a package marker."""\n', encoding="utf-8")
    thing = pkg / "thing.py"
    thing.write_text('"""Does a thing."""\n\n\ndef go() -> None: ...\n', encoding="utf-8")

    entries = collect_modules(tmp_path)

    assert [entry.relpath for entry in entries] == ["pkg/thing.py"]
    assert entries[0].symbols == ("go",)
