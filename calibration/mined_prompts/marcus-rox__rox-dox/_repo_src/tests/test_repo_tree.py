from pathlib import Path

from rox_dox.repo_tree import list_entries


def test_list_entries_identifies_root_directories_and_files(
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo

    entries = {entry.path: entry for entry in list_entries(repo, commit, ".")}

    assert entries["pkg"].is_dir
    assert entries["models"].is_dir
    assert not entries["README.md"].is_dir


def test_list_entries_returns_subdirectory_children_with_full_paths(
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo

    entries = {entry.path: entry for entry in list_entries(repo, commit, "pkg")}

    assert entries["pkg/a.py"].is_dir is False
    assert entries["pkg/sub"].is_dir
