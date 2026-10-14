"""Which build produced a run: the version's single definition, and the commit.

One case per acceptance criterion the mechanism can decide on its own — the
version agreeing with the project metadata, the version not being the schema
version, the three ways a commit is answered and the one way it is refused, and
the rule that binds them all: never inferred, never defaulted, never empty.

The git cases drive a real repository built in ``tmp_path`` rather than the one
the suite is running from, so they assert the resolver's behaviour instead of
this checkout's state.
"""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType

import pytest

from womblex import __version__
from womblex.store import build_info as bi
from womblex.store.build_info import UNAVAILABLE, BuildInfo, build_info, resolve_commit
from womblex.store.output import PARSER_VERSION


@pytest.fixture(autouse=True)
def _uncached():
    """The resolver caches per process; each case starts from a cold one."""
    build_info.cache_clear()
    yield
    build_info.cache_clear()


def _git_repo(root: Path) -> str:
    """A one-commit repository at *root*; returns its HEAD sha."""
    root.mkdir(parents=True, exist_ok=True)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        )

    run("init", "-q")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "Test")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    run("add", "tracked.txt")
    run("commit", "-q", "-m", "one")
    return run("rev-parse", "HEAD").stdout.strip()


class TestTheVersionHasOneDefinition:
    def test_the_package_version_and_the_project_metadata_agree(self):
        """The two spellings are hand-kept in step; this is what catches a drift."""
        try:
            installed = metadata.version("womblex")
        except metadata.PackageNotFoundError:  # pragma: no cover - source-tree run
            pytest.skip("womblex is not installed; nothing to compare __version__ against")
        assert installed == __version__

    def test_the_package_version_is_not_the_schema_version(self):
        """Two constants, two meanings.

        ``parser_version`` is the manifest's schema version and happens to look
        like an answer to "which Womblex wrote this". Conflating them would make
        a schema change read as a release.
        """
        assert PARSER_VERSION != __version__

    def test_the_resolved_build_reports_the_package_version(self):
        assert build_info().version == __version__


class TestTheCommitIsResolvedOrRefused:
    def test_a_work_tree_answers_with_its_head(self, tmp_path, monkeypatch):
        head = _git_repo(tmp_path / "repo")
        monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path / "repo")

        info = build_info()

        assert info.commit == head
        assert info.source == "git"
        assert info.commit_value == head

    def test_a_dirty_work_tree_says_so_rather_than_claiming_the_commit(
        self, tmp_path, monkeypatch
    ):
        """The bytes that ran are not the bytes at that commit."""
        root = tmp_path / "repo"
        head = _git_repo(root)
        (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
        monkeypatch.setattr(bi, "_PACKAGE_DIR", root)

        info = build_info()

        assert info.commit == head
        assert info.source == "git-dirty"
        assert info.commit_value == f"{head}-dirty"

    def test_a_build_stamp_answers_where_there_is_no_work_tree(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path / "not-a-repo")
        monkeypatch.setattr(bi, "_stamped_commit", lambda: "a" * 40)

        info = build_info()

        assert info.commit == "a" * 40
        assert info.source == "stamp"
        assert info.commit_value == "a" * 40

    def test_the_work_tree_wins_over_a_stamp_left_beside_it(self, tmp_path, monkeypatch):
        """A stamp from an earlier local build is stale against the tree running now."""
        head = _git_repo(tmp_path / "repo")
        monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path / "repo")
        monkeypatch.setattr(bi, "_stamped_commit", lambda: "b" * 40)

        assert build_info().commit == head

    def test_neither_source_gives_unavailable_with_a_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path / "not-a-repo")
        monkeypatch.setattr(bi, "_stamped_commit", lambda: None)

        info = build_info()

        assert info.commit == UNAVAILABLE
        assert info.reason == "not-a-git-work-tree"
        assert info.commit_value == "unavailable:not-a-git-work-tree"

    def test_a_missing_git_binary_is_a_different_reason(self, tmp_path, monkeypatch):
        """No git and no work tree are distinguishable situations in the record."""
        monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path / "not-a-repo")
        monkeypatch.setattr(bi, "_stamped_commit", lambda: None)
        monkeypatch.setattr(
            bi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError)
        )

        assert build_info().reason == "git-not-installed"

    @pytest.mark.parametrize("stamped,expected", [("  ", None), (" d" + "e" * 39, "d" + "e" * 39)])
    def test_the_stamp_reader_strips_and_treats_an_empty_stamp_as_absent(
        self, stamped, expected, monkeypatch
    ):
        """A stamp written with an empty value is absent, not an answer.

        Injected as a module rather than by patching the reader, so the reader's
        own handling is what is under test.
        """
        module = ModuleType("womblex._build_stamp")
        module.COMMIT = stamped  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "womblex._build_stamp", module)

        assert bi._stamped_commit() == expected


class TestTheValueIsNeverEmptyOrInvented:
    @pytest.mark.parametrize(
        "info",
        [
            BuildInfo("0.0.0", "c" * 40, "git", ""),
            BuildInfo("0.0.0", "c" * 40, "git-dirty", ""),
            BuildInfo("0.0.0", "c" * 40, "stamp", ""),
            BuildInfo("0.0.0", UNAVAILABLE, UNAVAILABLE, "not-a-git-work-tree"),
        ],
    )
    def test_every_state_renders_a_non_empty_value(self, info):
        assert info.commit_value.strip()

    def test_the_resolver_never_returns_an_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path / "not-a-repo")
        monkeypatch.setattr(bi, "_stamped_commit", lambda: None)

        assert resolve_commit().startswith(f"{UNAVAILABLE}:")

    def test_a_git_failure_does_not_propagate_to_the_caller(self, tmp_path, monkeypatch):
        """A run is not failed for a fact it can honestly do without."""
        monkeypatch.setattr(bi, "_PACKAGE_DIR", tmp_path / "not-a-repo")
        monkeypatch.setattr(bi, "_stamped_commit", lambda: None)
        monkeypatch.setattr(
            bi.subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 5)),
        )

        assert resolve_commit() == "unavailable:not-a-git-work-tree"
