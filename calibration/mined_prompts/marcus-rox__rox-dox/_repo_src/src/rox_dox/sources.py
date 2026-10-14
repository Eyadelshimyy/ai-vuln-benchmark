from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from rox_dox.model import (
    CodeSource,
    NotionSource,
    Page,
    Source,
    page_sources,
)
from rox_dox.schema import Table


def _valid_notion_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
    except ValueError:
        return False

    if parsed.scheme != "https" or host is None:
        return False
    return host in {"notion.so", "www.notion.so"} or (
        host.endswith(".notion.site") and host != "notion.site"
    )


def _notion_problem(label: str, url: str) -> str | None:
    if _valid_notion_url(url):
        return None
    return (
        f"{label}: invalid Notion URL '{url}' "
        "(expected https://notion.so, www.notion.so, or *.notion.site)"
    )


def _line_count(
    commit: str, path: str, repo: Path, cache: dict[tuple[str, str], int | None]
) -> int | None:
    key = (commit, path)
    if key not in cache:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{path}"],
            check=False,
            capture_output=True,
        )
        cache[key] = len(result.stdout.splitlines()) if result.returncode == 0 else None
    return cache[key]


def _code_source_problems(
    label: str,
    source: CodeSource,
    commit: str,
    repo: Path,
    cache: dict[tuple[str, str], int | None],
) -> list[str]:
    path = PurePosixPath(source.path)
    if path.is_absolute() or ".." in path.parts or not source.path:
        return [f"{label}: source path '{source.path}' is not repo-relative"]

    line_count = _line_count(commit, source.path, repo, cache)
    if line_count is None:
        return [f"{label}: source file '{source.path}' not found at {commit[:8]}"]

    start, end = source.lines
    problems = []
    if start < 1:
        problems.append(
            f"{label}: {source.path} lines {start}-{end} start before line 1 "
            f"at {commit[:8]}"
        )
    if start > end:
        problems.append(
            f"{label}: invalid line range {start}-{end} in {source.path} "
            f"at {commit[:8]}"
        )
    elif end > line_count:
        problems.append(
            f"{label}: {source.path} lines {start}-{end} beyond end of file "
            f"({line_count} lines) at {commit[:8]}"
        )
    return problems


def _source_problem(
    label: str,
    source: Source,
    page: Page,
    repo: Path,
    line_count_cache: dict[tuple[str, str], int | None],
    commit_exists: bool,
) -> list[str]:
    if isinstance(source, NotionSource):
        problem = _notion_problem(label, source.notion)
        return [problem] if problem is not None else []
    if not commit_exists:
        return []
    return _code_source_problems(label, source, page.commit, repo, line_count_cache)


def page_problems(
    page: Page,
    repo: Path,
    *,
    tables: Mapping[str, Table] | None = None,
) -> list[str]:
    commit_result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "cat-file",
            "-e",
            "--end-of-options",
            f"{page.commit}^{{commit}}",
        ],
        check=False,
        capture_output=True,
    )
    commit_exists = commit_result.returncode == 0
    problems = []
    if not commit_exists:
        problems.append(f"commit '{page.commit}' not found in repository")

    line_count_cache: dict[tuple[str, str], int | None] = {}
    for label, source in page_sources(page, tables=tables):
        problems.extend(
            _source_problem(label, source, page, repo, line_count_cache, commit_exists)
        )

    if commit_exists:
        for path in page.paths:
            if path == ".":
                continue
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "cat-file",
                    "-t",
                    "--end-of-options",
                    f"{page.commit}:{path}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or result.stdout.strip() not in {"blob", "tree"}:
                problems.append(f"path '{path}' not found at {page.commit[:8]}")
    return problems
