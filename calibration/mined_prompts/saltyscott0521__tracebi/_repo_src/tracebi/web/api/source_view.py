"""Read-only views of the files that define an object: a model, a pipeline, a
connector's models. The files come from where discovery found them, never from
a path the caller names, and each is cut at a size cap."""

from __future__ import annotations

import os
from typing import Optional

SOURCE_LANG = {".json": "json", ".html": "html", ".css": "css", ".js": "javascript", ".py": "python", ".ipynb": "json"}
SOURCE_MAX_BYTES = 256 * 1024


def source_file(path: str, label: Optional[str] = None) -> dict:
    """One file as the Code view shows it: where it is, what it says, and
    whether it was cut. Relative to the working directory when it sits inside it."""
    with open(path, "rb") as fh:
        raw = fh.read(SOURCE_MAX_BYTES + 1)
    cwd = os.getcwd()
    shown = os.path.relpath(path, cwd) if os.path.commonpath([os.path.abspath(path), cwd]) == cwd else path
    return {
        "path": shown,
        "label": label,
        "language": SOURCE_LANG.get(os.path.splitext(path)[1], "text"),
        "content": raw[:SOURCE_MAX_BYTES].decode("utf-8", errors="replace"),
        "truncated": len(raw) > SOURCE_MAX_BYTES,
    }


def source_payload(found: list[tuple[Optional[str], Optional[str]]], missing_hint: str, hint: str = "") -> dict:
    """``found`` is ``(label, path)`` pairs; paths that are not files are skipped.
    With nothing to show, ``missing_hint`` says why."""
    seen, files = set(), []
    for label, path in found:
        if path and os.path.isfile(path) and os.path.abspath(path) not in seen:
            seen.add(os.path.abspath(path))
            files.append(source_file(path, label))
    return {"files": files, "hint": hint if files else missing_hint}
