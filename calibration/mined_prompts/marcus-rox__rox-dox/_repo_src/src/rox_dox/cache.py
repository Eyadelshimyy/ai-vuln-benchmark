from __future__ import annotations

import hashlib
import os
import pickle
import string
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache" / "rox-dox"
# Bump this whenever a source extractor changes.
CACHE_VERSION = 6
T = TypeVar("T")


def _commit_key(commit: str) -> str:
    if len(commit) >= 7 and all(character in string.hexdigits for character in commit):
        return commit
    return hashlib.sha256(commit.encode()).hexdigest()


def get_or_compute(commit: str, key: str, compute: Callable[[], T]) -> T:
    if not key or Path(key).name != key:
        raise ValueError(f"invalid cache key '{key}'")

    cache_dir = CACHE_ROOT / _commit_key(commit) / f"v{CACHE_VERSION}"
    cache_path = cache_dir / f"{key}.pickle"
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as cache_file:
                return pickle.load(cache_file)
        except (
            AttributeError,
            EOFError,
            ImportError,
            OSError,
            pickle.PickleError,
            TypeError,
            ValueError,
        ):
            cache_path.unlink(missing_ok=True)

    value = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{key}.",
            dir=cache_dir,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            pickle.dump(value, temporary_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return value
