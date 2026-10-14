from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


DIRECTORY_OBJECT_TYPES = {"tree", "commit"}


@dataclass(frozen=True)
class RepoEntry:
    path: str
    is_dir: bool

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


def list_entries(repo: Path, commit: str, directory: str) -> list[RepoEntry]:
    """Immediate children of `directory` at `commit`; "." is the repository root."""
    command = ["git", "-C", str(repo), "ls-tree", commit]
    if directory != ".":
        command += ["--", f"{directory.rstrip('/')}/"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    entries = []
    for line in result.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        object_type = metadata.split()[1]
        entries.append(
            RepoEntry(path=path, is_dir=object_type in DIRECTORY_OBJECT_TYPES)
        )
    return entries


def list_tree_paths(repo: Path, commit: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            "--full-tree",
            commit,
        ],
        check=True,
        capture_output=True,
    )
    return [path.decode() for path in result.stdout.split(b"\0") if path]
