from __future__ import annotations

import posixpath
from pathlib import PurePosixPath
from urllib.parse import quote

from rox_dox.model import CodeSource, Source


def page_href(from_id: str, to_id: str) -> str:
    from_path = f"{PurePosixPath(from_id).as_posix()}.html"
    to_path = f"{PurePosixPath(to_id).as_posix()}.html"
    relative_path = posixpath.relpath(
        to_path,
        start=posixpath.dirname(from_path) or ".",
    )
    return quote(relative_path, safe="/")


def source_url(source: Source, *, repo_url: str, commit: str) -> str:
    if isinstance(source, CodeSource):
        path = quote(source.path, safe="/")
        return (
            f"{repo_url.rstrip('/')}/blob/{quote(commit, safe='')}/{path}"
            f"#L{source.lines[0]}-L{source.lines[1]}"
        )
    return source.notion
