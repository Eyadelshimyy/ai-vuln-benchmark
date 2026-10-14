"""Download the object files the world names, and check their bytes.

The scanned-object and YCB collections used to be git submodules. Almost all
of those bytes are packages this line never spawns. A manifest pins the files
that are spawned, at the commits the submodules pinned, and this module
fetches those files the same way the conveyor module is fetched: by URL and
SHA-256.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import yaml

from clave.corpus.artifacts import digest_of
from clave.errors import ClaveError

RAW_HOST = "https://raw.githubusercontent.com"


class AssetError(ClaveError):
    """A declared object file is missing from the manifest or failed its digest."""


@dataclass(frozen=True)
class ObjectFile:
    """One file the world or a fixture reads.

    Attributes:
        path: Path relative to the repository root.
        source: Key into the manifest's sources.
        sha256: Expected digest.
        url: Where the bytes are fetched from.
    """

    path: str
    source: str
    sha256: str
    url: str


def load_manifest(path: Path) -> tuple[ObjectFile, ...]:
    """Read the object manifest.

    Args:
        path: The YAML file.

    Returns:
        One entry per file, with its fetch URL filled in.

    Raises:
        AssetError: If a source or a file entry is incomplete.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise AssetError(f"{path} is not a mapping")
    sources = raw.get("sources")
    files = raw.get("files")
    if not isinstance(sources, dict) or not isinstance(files, list):
        raise AssetError(f"{path} needs 'sources' and 'files'")
    loaded: list[ObjectFile] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise AssetError(f"{path} has a file entry that is not a mapping")
        source_name = str(_need(entry, "source"))
        source = sources.get(source_name)
        if not isinstance(source, dict):
            raise AssetError(f"{path} names unknown source {source_name!r}")
        relative = str(_need(entry, "path"))
        prefix = f"third_party/{source_name}/"
        if not relative.startswith(prefix):
            raise AssetError(f"{relative} is not under {prefix}")
        repository = str(_need(source, "repository"))
        commit = str(_need(source, "commit"))
        url = f"{RAW_HOST}/{repository}/{commit}/{relative.removeprefix(prefix)}"
        loaded.append(
            ObjectFile(
                path=relative,
                source=source_name,
                sha256=str(_need(entry, "sha256")),
                url=url,
            )
        )
    return tuple(loaded)


def fetch_objects(
    root: Path,
    manifest: Path,
    opener: Callable[..., Any] | None = None,
) -> tuple[int, int]:
    """Download every declared file that is absent or does not match.

    A file already on disk whose digest matches is left untouched. A download
    is written only after its digest matches, so a failed fetch cannot replace
    a good file with a bad one.

    Args:
        root: Repository root.
        manifest: Path to the object manifest.
        opener: Callable with the `urlopen` signature, for tests.

    Returns:
        A pair of (downloaded, already present).

    Raises:
        AssetError: If a download's digest does not match the manifest.
    """
    read = opener or urlopen
    downloaded = 0
    present = 0
    for item in load_manifest(manifest):
        destination = root / item.path
        if destination.is_file() and digest_of(destination) == item.sha256:
            present += 1
            continue
        with read(item.url, timeout=120) as response:
            payload = response.read()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != item.sha256:
            raise AssetError(
                f"{item.path} downloaded as {observed}, and the manifest "
                f"records {item.sha256}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        downloaded += 1
    return downloaded, present


def _need(mapping: dict[str, Any], key: str) -> Any:
    """Read a key, naming it when it is absent."""
    if key not in mapping:
        raise AssetError(f"required asset key {key!r} is missing")
    return mapping[key]
