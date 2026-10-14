from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rox_dox.cache import get_or_compute
from rox_dox.model import CodeSource


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Column(SchemaModel):
    name: str
    type: str
    primary_key: bool = False
    foreign_key: str | None = None
    line: int = 1


class Table(SchemaModel):
    name: str
    class_name: str
    columns: list[Column]
    source: CodeSource
    duplicate_paths: list[str] = []


def _call_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return None


def _assignment(
    statement: ast.stmt,
) -> tuple[str, ast.expr | None, ast.expr | None] | None:
    if isinstance(statement, ast.Assign):
        value = statement.value
        for target in statement.targets:
            if isinstance(target, ast.Name):
                return target.id, value, None
    elif isinstance(statement, ast.AnnAssign) and isinstance(
        statement.target, ast.Name
    ):
        return statement.target.id, statement.value, statement.annotation
    return None


def _table_name(class_node: ast.ClassDef) -> str | None:
    for statement in class_node.body:
        assignment = _assignment(statement)
        if assignment is None:
            continue
        name, value, _ = assignment
        if (
            name == "__tablename__"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    return None


def _mapped_annotation_type(annotation: ast.expr | None) -> str | None:
    if (
        isinstance(annotation, ast.Subscript)
        and _call_name(annotation.value) == "Mapped"
    ):
        return ast.unparse(annotation.slice)
    return None


def _column_type(call: ast.Call, annotation: ast.expr | None) -> str:
    for argument in call.args:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            continue
        if isinstance(argument, ast.Call) and _call_name(argument.func) == "ForeignKey":
            continue
        return ast.unparse(argument)
    return _mapped_annotation_type(annotation) or "?"


def _foreign_key_target(call: ast.Call) -> str | None:
    arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
    for argument in arguments:
        for node in ast.walk(argument):
            if not isinstance(node, ast.Call) or _call_name(node.func) != "ForeignKey":
                continue
            for target in node.args:
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    return target.value
    return None


def _column(class_statement: ast.stmt) -> Column | None:
    assignment = _assignment(class_statement)
    if assignment is None:
        return None
    attribute_name, value, annotation = assignment
    if not isinstance(value, ast.Call) or _call_name(value.func) not in {
        "Column",
        "mapped_column",
    }:
        return None

    explicit_name = next(
        (
            argument.value
            for argument in value.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ),
        attribute_name,
    )
    primary_key = any(
        keyword.arg == "primary_key"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in value.keywords
    )
    return Column(
        name=explicit_name,
        type=_column_type(value, annotation),
        primary_key=primary_key,
        foreign_key=_foreign_key_target(value),
        line=class_statement.lineno,
    )


def _table_columns(
    class_node: ast.ClassDef,
    classes_by_name: dict[str, ast.ClassDef],
    cache: dict[str, list[Column]],
    visiting: set[str],
) -> list[Column]:
    if class_node.name in cache:
        return cache[class_node.name]
    if class_node.name in visiting:
        return []

    visiting.add(class_node.name)
    columns_by_name: dict[str, Column] = {}
    for base in reversed(class_node.bases):
        base_name = _call_name(base)
        base_node = classes_by_name.get(base_name or "")
        if base_node is not None:
            columns_by_name.update(
                (
                    column.name,
                    column,
                )
                for column in _table_columns(
                    base_node,
                    classes_by_name,
                    cache,
                    visiting,
                )
            )
    for statement in class_node.body:
        column = _column(statement)
        if column is not None:
            columns_by_name[column.name] = column
    visiting.remove(class_node.name)
    cache[class_node.name] = list(columns_by_name.values())
    return cache[class_node.name]


def _tables_in_file(source: str, path: str) -> list[Table]:
    module = ast.parse(source, filename=path)
    classes = sorted(
        (node for node in ast.walk(module) if isinstance(node, ast.ClassDef)),
        key=lambda node: node.lineno,
    )
    classes_by_name = {class_node.name: class_node for class_node in classes}
    column_cache: dict[str, list[Column]] = {}
    tables = []
    for class_node in classes:
        name = _table_name(class_node)
        if name is None:
            continue
        tables.append(
            Table(
                name=name,
                class_name=class_node.name,
                columns=_table_columns(
                    class_node, classes_by_name, column_cache, set()
                ),
                source=CodeSource(
                    path=path,
                    lines=(
                        class_node.lineno,
                        class_node.end_lineno or class_node.lineno,
                    ),
                ),
            )
        )
    return tables


def _extract_tables(repo: Path, commit: str) -> dict[str, Table]:
    candidates = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "grep",
            "-l",
            "__tablename__",
            commit,
            "--",
            "*.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if candidates.returncode not in (0, 1):
        raise RuntimeError(
            f"could not search Python files at {commit}: {candidates.stderr.strip()}"
        )

    paths = sorted(
        line.partition(":")[2] for line in candidates.stdout.splitlines() if ":" in line
    )
    tables: dict[str, Table] = {}
    for path in paths:
        file_result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{path}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if file_result.returncode != 0:
            raise RuntimeError(
                f"could not read {path} at {commit}: {file_result.stderr.strip()}"
            )
        try:
            file_tables = _tables_in_file(file_result.stdout, path)
        except SyntaxError as error:
            raise RuntimeError(
                f"could not parse {path} at {commit}: {error}"
            ) from error

        for table in file_tables:
            if table.name in tables:
                kept_table = tables[table.name]
                if path != kept_table.source.path:
                    kept_table.duplicate_paths.append(path)
                continue
            tables[table.name] = table
    return tables


def extract_tables(repo: Path, commit: str) -> dict[str, Table]:
    return get_or_compute(commit, "tables", lambda: _extract_tables(repo, commit))
