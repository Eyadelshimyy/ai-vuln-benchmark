from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from rox_dox.features import _git_blobs, _git_tree

PARALLEL_MIN_FILES = 50
_SKIPPED_FOLDERS = frozenset({"tests", "migrations"})
_CONSTANT_NAME = re.compile(r"^_*[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class OutlineRow:
    name: str
    kind: Literal["class", "function", "async_function", "constant"]
    line: int
    doc: str


@dataclass(frozen=True)
class FileOutline:
    path: str
    rows: tuple[OutlineRow, ...]
    empty: bool
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        if self.empty:
            return {"path": self.path, "empty": True}
        if self.error is not None:
            return {"path": self.path, "error": self.error}
        return {
            "path": self.path,
            "empty": False,
            "rows": [asdict(row) for row in self.rows],
        }


class OutlineInputError(ValueError):
    pass


def _module_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _doc_first_line(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    doc = ast.get_docstring(node)
    return doc.splitlines()[0].strip() if doc else ""


def _assignment_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple):
        return [name for element in target.elts for name in _assignment_names(element)]
    return []


def outline_source(path: str, source: str) -> FileOutline:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        return FileOutline(
            path=path,
            rows=(),
            empty=False,
            error=f"{path}:{error.lineno}: {error.msg}",
        )

    rows = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            rows.append(
                OutlineRow(
                    name=node.name,
                    kind="class",
                    line=node.lineno,
                    doc=_doc_first_line(node),
                )
            )
        elif isinstance(node, ast.FunctionDef):
            rows.append(
                OutlineRow(
                    name=node.name,
                    kind="function",
                    line=node.lineno,
                    doc=_doc_first_line(node),
                )
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            rows.append(
                OutlineRow(
                    name=node.name,
                    kind="async_function",
                    line=node.lineno,
                    doc=_doc_first_line(node),
                )
            )
        elif isinstance(node, ast.Assign):
            names = [
                name
                for target in node.targets
                for name in _assignment_names(target)
                if _CONSTANT_NAME.fullmatch(name)
            ]
            rows.extend(
                OutlineRow(name=name, kind="constant", line=node.lineno, doc="")
                for name in names
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if _CONSTANT_NAME.fullmatch(node.target.id):
                rows.append(
                    OutlineRow(
                        name=node.target.id,
                        kind="constant",
                        line=node.lineno,
                        doc="",
                    )
                )

    empty = not tree.body or all(
        isinstance(statement, (ast.Import, ast.ImportFrom))
        or (index == 0 and _module_docstring(statement))
        for index, statement in enumerate(tree.body)
    )
    return FileOutline(path=path, rows=tuple(rows), empty=empty)


def outline_sources(sources: Mapping[str, str]) -> list[FileOutline]:
    paths = list(sources)
    texts = list(sources.values())
    if len(paths) > PARALLEL_MIN_FILES:
        with ProcessPoolExecutor() as executor:
            return list(executor.map(outline_source, paths, texts, chunksize=64))
    return [outline_source(path, text) for path, text in zip(paths, texts)]


def _has_skipped_folder(relative_path: str) -> bool:
    return any(
        component in _SKIPPED_FOLDERS
        for component in PurePosixPath(relative_path).parts[:-1]
    )


def read_worktree(paths: Sequence[Path]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            if path.suffix != ".py":
                raise OutlineInputError(f"not a Python file: {path}")
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("*.py"), key=Path.as_posix)
        else:
            raise OutlineInputError(f"path not found: {path}")

        for candidate in candidates:
            if path.is_dir() and _has_skipped_folder(
                candidate.relative_to(path).as_posix()
            ):
                continue
            key = candidate.as_posix()
            if key not in sources:
                sources[key] = candidate.read_text(encoding="utf-8", errors="replace")
    return sources


def _normalise_commit_path(path: str) -> str:
    while path.startswith("./"):
        path = path[2:]
    path = path.rstrip("/")
    return "" if path in {"", "."} else path


def read_commit(repo: Path, commit: str, paths: Sequence[str]) -> dict[str, str]:
    tree = _git_tree(repo, commit)
    tree_by_path = dict(tree)
    selected: list[tuple[str, str]] = []
    selected_paths: set[str] = set()

    for requested_path in paths:
        path = _normalise_commit_path(requested_path)
        if path in tree_by_path:
            if not path.endswith(".py"):
                raise OutlineInputError(f"not a Python file: {requested_path}")
            candidates = [path]
        else:
            prefix = f"{path}/" if path else ""
            matched_paths = sorted(
                tree_path for tree_path in tree_by_path if tree_path.startswith(prefix)
            )
            if not matched_paths:
                raise OutlineInputError(f"path not found at {commit[:10]}: {path}")
            candidates = [
                tree_path
                for tree_path in matched_paths
                if tree_path.endswith(".py")
                and not _has_skipped_folder(
                    tree_path[len(prefix) :] if prefix else tree_path
                )
            ]

        for candidate in candidates:
            if candidate in selected_paths:
                continue
            selected_paths.add(candidate)
            selected.append((candidate, tree_by_path[candidate]))

    blobs = _git_blobs(repo, selected)
    return {path: blobs[path] for path, _object_id in selected}


def _table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def format_markdown(outlines: Sequence[FileOutline]) -> str:
    sections = []
    for outline in outlines:
        section = [f"## `{outline.path}`", ""]
        if outline.error is not None:
            section.append(f"Parse error: {outline.error}")
        elif outline.empty:
            section.append("Empty or imports only.")
        elif not outline.rows:
            section.append("No top-level classes, functions or constants.")
        else:
            section.extend(
                [
                    "| Object/Function | What it does |",
                    "|---|---|",
                    *(
                        f"| `{row.name}` | {_table_cell(row.doc) or '—'} |"
                        for row in outline.rows
                    ),
                ]
            )
        sections.append("\n".join(section))
    return "\n\n".join(sections) + ("\n" if sections else "")
