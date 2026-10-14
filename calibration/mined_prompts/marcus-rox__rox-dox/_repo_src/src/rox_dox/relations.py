from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

from rox_dox.cache import get_or_compute
from rox_dox.schema import Table, extract_tables

COMMENTED_FOREIGN_KEY = re.compile(r"""#\s*ForeignKey\s*\(\s*["']([^"']+)["']\s*\)""")


@dataclass(frozen=True)
class RelationCandidate:
    src: str
    dst: str
    signal: str
    path: str
    line: int

    def to_tsv(self) -> str:
        return f"{self.src}\t{self.dst}\t{self.signal}\t{self.path}:{self.line}"


def _git_output(repo: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _candidate_paths(repo: Path, commit: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "grep",
            "--full-name",
            "-I",
            "-l",
            "-E",
            "-e",
            "==|primaryjoin|ForeignKey",
            commit,
            "--",
            "backend/src",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "git grep failed")
    paths = [
        line.partition(":")[2] for line in result.stdout.splitlines() if ":" in line
    ]
    return sorted(
        path
        for path in paths
        if path.startswith("backend/src/") and path.endswith(".py")
    )


def _call_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return None


def _column_reference(expression: ast.expr) -> tuple[str, str] | None:
    while (
        isinstance(expression, ast.Call)
        and _call_name(expression.func) in {"foreign", "remote"}
        and expression.args
    ):
        expression = expression.args[0]
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        return expression.value.id, expression.attr
    return None


def _column_assignment(statement: ast.stmt) -> str | None:
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        name = statement.target.id
        value = statement.value
    elif isinstance(statement, ast.Assign) and isinstance(
        statement.targets[0], ast.Name
    ):
        name = statement.targets[0].id
        value = statement.value
    else:
        return None
    if isinstance(value, ast.Call) and _call_name(value.func) in {
        "Column",
        "mapped_column",
    }:
        return name
    return None


def _candidate_for_columns(
    left: tuple[str, str],
    right: tuple[str, str],
    *,
    class_to_tables: dict[str, tuple[str, ...]],
    imported_tables: dict[str, tuple[str, ...]],
    selected_tables: set[str],
    signal: str,
    path: str,
    line: int,
) -> list[RelationCandidate]:
    src_tables = imported_tables.get(left[0], class_to_tables.get(left[0], ()))
    dst_tables = imported_tables.get(right[0], class_to_tables.get(right[0], ()))
    if not src_tables or not dst_tables:
        return []
    ambiguous = len(src_tables) > 1 or len(dst_tables) > 1
    candidate_signal = f"{signal}:ambiguous" if ambiguous else signal
    return [
        RelationCandidate(
            src=f"{src_table}.{left[1]}",
            dst=f"{dst_table}.{right[1]}",
            signal=candidate_signal,
            path=path,
            line=line,
        )
        for src_table in src_tables
        for dst_table in dst_tables
        if src_table in selected_tables and dst_table in selected_tables
    ]


def _comparison_candidate(
    expression: ast.expr,
    *,
    class_to_tables: dict[str, tuple[str, ...]],
    imported_tables: dict[str, tuple[str, ...]],
    selected_tables: set[str],
    signal: str,
    path: str,
    line: int,
) -> list[RelationCandidate]:
    if (
        not isinstance(expression, ast.Compare)
        or len(expression.ops) != 1
        or not isinstance(expression.ops[0], ast.Eq)
        or len(expression.comparators) != 1
    ):
        return []
    left = _column_reference(expression.left)
    right = _column_reference(expression.comparators[0])
    if left is None or right is None:
        return []
    return _candidate_for_columns(
        left,
        right,
        class_to_tables=class_to_tables,
        imported_tables=imported_tables,
        selected_tables=selected_tables,
        signal=signal,
        path=path,
        line=line,
    )


def _commented_foreign_key_candidates(
    module: ast.Module,
    source_lines: list[str],
    *,
    path: str,
    class_to_table: dict[str, str],
    selected_tables: set[str],
) -> list[RelationCandidate]:
    candidates = []
    for class_node in ast.walk(module):
        if not isinstance(class_node, ast.ClassDef):
            continue
        table_name = class_to_table.get(class_node.name)
        if table_name not in selected_tables:
            continue
        for statement in class_node.body:
            column_name = _column_assignment(statement)
            if column_name is None:
                continue
            end_line = getattr(statement, "end_lineno", statement.lineno)
            for line_number in range(statement.lineno, end_line + 1):
                for match in COMMENTED_FOREIGN_KEY.finditer(
                    source_lines[line_number - 1]
                ):
                    target = match.group(1)
                    if "." not in target:
                        continue
                    target_table, target_column = target.rsplit(".", 1)
                    if target_table not in selected_tables:
                        continue
                    candidates.append(
                        RelationCandidate(
                            src=f"{table_name}.{column_name}",
                            dst=f"{target_table}.{target_column}",
                            signal="commented_fk",
                            path=path,
                            line=line_number,
                        )
                    )
    return candidates


def _primaryjoin_candidates(
    module: ast.Module,
    *,
    path: str,
    class_to_tables: dict[str, tuple[str, ...]],
    imported_tables: dict[str, tuple[str, ...]],
    selected_tables: set[str],
) -> list[RelationCandidate]:
    candidates = []
    for call in ast.walk(module):
        if not isinstance(call, ast.Call):
            continue
        for keyword in call.keywords:
            if (
                keyword.arg != "primaryjoin"
                or not isinstance(keyword.value, ast.Constant)
                or not isinstance(keyword.value.value, str)
            ):
                continue
            try:
                expression = ast.parse(keyword.value.value, mode="eval")
            except SyntaxError:
                continue
            for node in ast.walk(expression):
                candidates.extend(
                    _comparison_candidate(
                        node,
                        class_to_tables=class_to_tables,
                        imported_tables=imported_tables,
                        selected_tables=selected_tables,
                        signal="primaryjoin",
                        path=path,
                        line=keyword.value.lineno,
                    )
                )
    return candidates


def _comparison_candidates(
    module: ast.Module,
    *,
    path: str,
    class_to_tables: dict[str, tuple[str, ...]],
    imported_tables: dict[str, tuple[str, ...]],
    selected_tables: set[str],
) -> list[RelationCandidate]:
    candidates = []
    for node in ast.walk(module):
        candidates.extend(
            _comparison_candidate(
                node,
                class_to_tables=class_to_tables,
                imported_tables=imported_tables,
                selected_tables=selected_tables,
                signal="comparison",
                path=path,
                line=getattr(node, "lineno", 1),
            )
        )
    return candidates


def _imported_class_tables(
    module: ast.Module,
    *,
    path: str,
    tables_by_path: dict[str, list[tuple[str, str]]],
) -> dict[str, tuple[str, ...]]:
    source_path = PurePosixPath(path)
    source_parts = source_path.with_suffix("").parts
    try:
        source_root = source_parts.index("src")
    except ValueError:
        return {}
    package_parts = source_parts[source_root + 1 : -1]
    imported: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_parts = (node.module or "").split(".") if node.module else []
        if node.level:
            base_parts = package_parts[: len(package_parts) - node.level + 1]
            module_parts = [*base_parts, *module_parts]
        if not module_parts:
            continue
        candidate_paths = [
            f"backend/src/{'/'.join(module_parts)}.py",
            f"backend/src/{'/'.join(module_parts)}/__init__.py",
        ]
        imported_tables = [
            (class_name, table_name)
            for candidate_path in candidate_paths
            for class_name, table_name in tables_by_path.get(candidate_path, [])
        ]
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            matches = tuple(
                table_name
                for class_name, table_name in imported_tables
                if class_name == alias.name
            )
            imported[local_name] = tuple(sorted(set(matches)))
    return imported


def _file_candidates(
    path: str,
    source: str,
    *,
    class_to_tables: dict[str, tuple[str, ...]],
    tables_by_path: dict[str, list[tuple[str, str]]],
    selected_tables: set[str],
) -> list[RelationCandidate]:
    try:
        module = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise RuntimeError(
            f"could not parse {path}: line {error.lineno}: {error.msg}"
        ) from error
    source_lines = source.splitlines()
    imported_tables = _imported_class_tables(
        module,
        path=path,
        tables_by_path=tables_by_path,
    )
    local_class_tables = dict(tables_by_path.get(path, []))
    return [
        *_commented_foreign_key_candidates(
            module,
            source_lines,
            path=path,
            class_to_table=local_class_tables,
            selected_tables=selected_tables,
        ),
        *_primaryjoin_candidates(
            module,
            path=path,
            class_to_tables=class_to_tables,
            imported_tables=imported_tables,
            selected_tables=selected_tables,
        ),
        *_comparison_candidates(
            module,
            path=path,
            class_to_tables=class_to_tables,
            imported_tables=imported_tables,
            selected_tables=selected_tables,
        ),
    ]


def _id_column_candidates(
    tables: Mapping[str, Table],
    selected_tables: set[str],
) -> list[RelationCandidate]:
    candidates = []
    for table_name in sorted(selected_tables & tables.keys()):
        table = tables[table_name]
        for column in table.columns:
            if column.name in {"id", "public_id", "rox_org_id"}:
                continue
            suffix = next(
                (
                    suffix
                    for suffix in ("_public_id", "_id")
                    if column.name.endswith(suffix)
                ),
                None,
            )
            if suffix is None:
                continue
            tokens = column.name[: -len(suffix)].split("_")
            if not tokens:
                continue
            target_table = None
            for index in range(len(tokens)):
                candidate_table = "_".join(tokens[index:])
                if candidate_table in tables:
                    target_table = candidate_table
                    break
            if target_table is None or target_table == table_name:
                continue
            if target_table not in selected_tables:
                continue
            target_columns = {item.name for item in tables[target_table].columns}
            possible_columns = [
                column.name,
                *(
                    f"{'_'.join(tokens[index:])}{suffix}"
                    for index in range(len(tokens))
                ),
                "public_id",
                "id",
            ]
            target_column = next(
                (name for name in possible_columns if name in target_columns),
                None,
            )
            if target_column is None:
                continue
            candidates.append(
                RelationCandidate(
                    src=f"{table_name}.{column.name}",
                    dst=f"{target_table}.{target_column}",
                    signal="id_column",
                    path=table.source.path,
                    line=column.line,
                )
            )
    return candidates


def _find_relation_candidates(
    repo: Path,
    commit: str,
    tables: Sequence[str],
) -> list[RelationCandidate]:
    extracted_tables = extract_tables(repo, commit)
    class_to_tables: dict[str, tuple[str, ...]] = {}
    tables_by_path: dict[str, list[tuple[str, str]]] = {}
    for table in extracted_tables.values():
        class_to_tables.setdefault(table.class_name, ())
        class_to_tables[table.class_name] = tuple(
            sorted({*class_to_tables[table.class_name], table.name})
        )
        tables_by_path.setdefault(table.source.path, []).append(
            (table.class_name, table.name)
        )
    selected_tables = set(tables)
    candidates = []
    for path in _candidate_paths(repo, commit):
        source = _git_output(repo, ["show", f"{commit}:{path}"])
        candidates.extend(
            _file_candidates(
                path,
                source,
                class_to_tables=class_to_tables,
                tables_by_path=tables_by_path,
                selected_tables=selected_tables,
            )
        )
    candidates.extend(_id_column_candidates(extracted_tables, selected_tables))

    candidates.sort(key=lambda candidate: (candidate.path, candidate.line))
    deduplicated: dict[tuple[str, str, str], RelationCandidate] = {}
    for candidate in candidates:
        deduplicated.setdefault(
            (candidate.src, candidate.dst, candidate.signal),
            candidate,
        )
    return sorted(
        deduplicated.values(),
        key=lambda candidate: (
            candidate.src,
            candidate.dst,
            candidate.signal,
            candidate.path,
            candidate.line,
        ),
    )


def find_relation_candidates(
    repo: Path,
    commit: str,
    tables: Sequence[str],
) -> list[RelationCandidate]:
    table_names = sorted(set(tables))
    table_digest = hashlib.sha256(
        json.dumps(table_names, separators=(",", ":")).encode()
    ).hexdigest()
    return get_or_compute(
        commit,
        f"relation-candidates-{table_digest}",
        lambda: _find_relation_candidates(repo, commit, table_names),
    )
