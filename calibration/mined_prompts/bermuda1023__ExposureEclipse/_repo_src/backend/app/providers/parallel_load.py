"""Parallel multi-dataset fact loading with bounded workers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_many(
    keys: Sequence[str],
    loader: Callable[[str], T],
    *,
    max_workers: int = 16,
    empty: T | None = None,
    on_error: str = "empty",
) -> dict[str, T]:
    """Load ``keys`` in parallel via ``loader(key)``.

    ``on_error``: ``empty`` | ``raise`` | ``skip``.
    Failed loads under ``empty`` store ``empty`` (default None / []).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for k in keys:
        if not k or k in seen:
            continue
        seen.add(k)
        ordered.append(k)

    if not ordered:
        return {}

    if len(ordered) == 1:
        k = ordered[0]
        try:
            return {k: loader(k)}
        except Exception:
            if on_error == "raise":
                raise
            if on_error == "skip":
                return {}
            return {k: empty}  # type: ignore[dict-item]

    workers = max(1, min(max_workers, len(ordered)))
    out: dict[str, T] = {}

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edm-load") as pool:
        futures = {pool.submit(loader, k): k for k in ordered}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                out[k] = fut.result()
            except Exception as exc:
                logger.warning("Fact load failed for %s: %s", k, exc)
                if on_error == "empty":
                    out[k] = empty  # type: ignore[assignment]
                elif on_error == "raise":
                    for f in futures:
                        f.cancel()
                    raise
                # skip: omit key

    return out
