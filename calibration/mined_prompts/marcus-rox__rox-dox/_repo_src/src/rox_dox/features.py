from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from rox_dox.cache import get_or_compute
from rox_dox.relations import RelationCandidate, find_relation_candidates
from rox_dox.schema import Table, extract_tables


DomainReason = Literal["tables", "calls", "called_by"]
PARALLEL_MIN_FILES = 50
PARALLEL_FILE_CHUNKSIZE = 64
_FORK_CONTEXT = multiprocessing.get_context("fork")
_worker_shared: object = None
SharedT = TypeVar("SharedT")
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


class FeatureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["table", "call", "route", "reference"]
    path: str
    line: int
    table: str | None = None
    to: str | None = None
    tag: str | None = None
    target: str | None = None


class FeatureFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    primary: bool
    reason: Literal["tables", "calls", "called_by", "route", "reference"]
    evidence: list[FeatureEvidence]


class TableLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: str
    b: str
    signal: str
    path: str
    line: int


class Feature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tables: list[str]
    files: list[FeatureFile]
    table_links: list[TableLink]


class UncoveredFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    reason: Literal["no shared table or call in its feature", "no feature reached"]


class FeatureCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_files: int
    primary_placed: int
    shared: int
    uncovered: int
    web: int
    deployment: int
    skills: int
    unparseable: int


class FeatureMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str
    domain: str
    threshold: float
    tables: list[str]
    features: list[Feature]
    cross_links: list[TableLink] = Field(default_factory=list)
    external_links: list[TableLink] = Field(default_factory=list)
    uncovered: list[UncoveredFile]
    unmapped_tags: list[str]
    counts: FeatureCounts


@dataclass
class _Snapshot:
    files: dict[str, str]
    all_paths: set[str]


@dataclass
class _GraphFile:
    path: str
    module: str
    source: str
    tree: ast.Module
    imports: list[str] = field(default_factory=list)
    import_lines: dict[str, int] = field(default_factory=dict)
    tables: set[str] = field(default_factory=set)
    table_lines: dict[str, int] = field(default_factory=dict)
    table_accesses: dict[str, tuple[int | None, int | None]] = field(
        default_factory=dict
    )
    defines: set[str] = field(default_factory=set)
    reexport_only: bool = False
    domains: list[str] = field(default_factory=list)
    domain_reason: str | None = None


@dataclass(frozen=True)
class _RawLink:
    a: str
    b: str
    signal: str
    path: str
    line: int


@dataclass
class _Placement:
    feature: int
    primary: bool
    reason: Literal["tables", "calls", "called_by", "route", "reference"]
    evidence: list[FeatureEvidence]


def _most_common_lowest(counter: Counter[int]) -> int:
    return min(counter, key=lambda feature: (-counter[feature], feature))


_GENERATED_ROUTE = re.compile(r"generated/(?:core/)?([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)")
_TS_IMPORT = re.compile(
    r"""(?:\bfrom\s*["']([^"']+)["']|\bfrom\s+["']([^"']+)["']|"""
    r"""\bimport\s*\(\s*["']([^"']+)["']\s*\)|"""
    r"""\brequire\s*\(\s*["']([^"']+)["']\s*\))"""
)
_NAMESPACE_CALL = "FlaskRestxNamespace"
_SCORE_TIE_EPSILON = 1e-9
_DEPLOYMENT_ROOTS = (
    "k8s/",
    "cicd/",
    ".circleci/",
    ".github/workflows/",
)


def _lowest_score_feature(scores: list[float]) -> int:
    highest = max(scores)
    return min(
        feature
        for feature, score in enumerate(scores)
        if score >= highest - _SCORE_TIE_EPSILON
    )


def _is_excluded(path: str) -> bool:
    pure_path = PurePosixPath(path)
    excluded_segments = {"tests", "test", "__tests__", "migrations", "alembic"}
    if any(
        segment in excluded_segments or segment.startswith("tests_")
        for segment in pure_path.parts
    ):
        return True
    filename = pure_path.name
    return (
        filename == "conftest.py"
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or ".test." in filename
        or ".spec." in filename
    )


def _is_deployment_path(path: str) -> bool:
    return (
        path.startswith(_DEPLOYMENT_ROOTS)
        or "/Dockerfile" in path
        or path.startswith("Dockerfile")
        or ("/" not in path and path.endswith((".yml", ".yaml")))
    )


def is_excluded_path(path: str) -> bool:
    return _is_excluded(path)


def is_feature_path(path: str) -> bool:
    return not _is_excluded(path) and (
        (path.startswith("backend/src/") and path.endswith(".py"))
        or (path.startswith("web/") and path.endswith((".ts", ".tsx")))
        or _is_deployment_path(path)
        or path.startswith(".agents/skills/")
    )


def _git_tree(repo: Path, commit: str) -> list[tuple[str, str]]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "-z",
            "--end-of-options",
            commit,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not list git tree at {commit}: {result.stderr.decode().strip()}"
        )
    entries = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode().split(" ", 2)
        if object_type != "blob" or mode == "120000":
            continue
        entries.append((path_bytes.decode(), object_id))
    return entries


def _git_blobs(repo: Path, objects: list[tuple[str, str]]) -> dict[str, str]:
    if not objects:
        return {}
    request = "".join(f"{object_id}\n" for _, object_id in objects).encode()
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        check=True,
        input=request,
        capture_output=True,
    )
    output = result.stdout
    offset = 0
    contents = {}
    for path, object_id in objects:
        header_end = output.index(b"\n", offset)
        header = output[offset:header_end].decode()
        found_id, object_type, size_text = header.split(" ", 2)
        if found_id != object_id or object_type != "blob":
            raise RuntimeError(f"could not read {path} at {object_id}")
        offset = header_end + 1
        size = int(size_text)
        contents[path] = output[offset : offset + size].decode(
            "utf-8", errors="replace"
        )
        offset += size + 1
    return contents


def _set_worker_shared(shared: object) -> None:
    global _worker_shared
    _worker_shared = shared


def _call_with_worker_shared(
    function: Callable[[object, ItemT], ResultT], item: ItemT
) -> ResultT:
    return function(_worker_shared, item)


def parallel_map(
    function: Callable[[SharedT, ItemT], ResultT],
    shared: SharedT,
    items: Sequence[ItemT],
    *,
    min_items: int = 1,
    chunksize: int = 1,
    desc: str,
    unit: str,
) -> list[ResultT]:
    """Return [function(shared, item) for item in items], in forked workers past min_items.

    shared reaches the workers through fork, so it is never pickled; items and results
    are, so keep them plain data. function must be defined at module level.
    """
    if len(items) <= min_items:
        return [function(shared, item) for item in tqdm(items, desc=desc, unit=unit)]
    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        mp_context=_FORK_CONTEXT,
        initializer=_set_worker_shared,
        initargs=(shared,),
    ) as executor:
        results = executor.map(
            partial(_call_with_worker_shared, function),
            items,
            chunksize=chunksize,
        )
        return list(tqdm(results, total=len(items), desc=desc, unit=unit))


def _snapshot_from_git(repo: Path, commit: str) -> _Snapshot:
    tree = _git_tree(repo, commit)
    selected = []
    all_paths = {path for path, _ in tree}
    for path, object_id in tree:
        if is_feature_path(path):
            selected.append((path, object_id))
    return _Snapshot(_git_blobs(repo, selected), all_paths)


def _snapshot(repo: Path, commit: str) -> _Snapshot:
    return get_or_compute(
        commit,
        "snapshot",
        lambda: _snapshot_from_git(repo, commit),
    )


def _module_name(path: str) -> str:
    relative = PurePosixPath(path).relative_to("backend/src")
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _call_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return None


def _attribute_parts(expression: ast.expr) -> list[str] | None:
    parts = []
    while isinstance(expression, ast.Attribute):
        parts.append(expression.attr)
        expression = expression.value
    if not isinstance(expression, ast.Name):
        return None
    return [expression.id, *reversed(parts)]


def _relative_module(
    module: str,
    node: ast.ImportFrom,
    *,
    is_package: bool,
) -> str:
    if not node.level:
        return node.module or ""
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    base = package_parts[: len(package_parts) - node.level + 1]
    return ".".join([*base, *([node.module] if node.module else [])])


_WRITE_CALLS = frozenset({"insert", "update", "delete"})
_BULK_WRITE_CALLS = frozenset({"bulk_insert_mappings", "bulk_update_mappings"})
_CHAINED_WRITE_CALLS = frozenset({"update", "delete"})
_READ_CALLS = frozenset({"query", "select"})
_READ_METHOD_PREFIXES = ("find", "get", "list", "fetch", "load")


def _table_references(
    expression: ast.expr,
    class_tables: Mapping[str, list[str]],
    class_aliases: Mapping[str, list[str]],
    module_aliases: Mapping[str, str],
) -> set[str]:
    tables = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name):
            tables.update(class_aliases.get(node.id, ()))
            tables.update(class_tables.get(node.id, ()))
        if not isinstance(node, ast.Attribute):
            continue
        parts = _attribute_parts(node)
        if not parts:
            continue
        tables.update(class_aliases.get(parts[0], ()))
        tables.update(class_tables.get(parts[0], ()))
        if len(parts) > 1 and parts[0] in module_aliases:
            tables.update(class_tables.get(parts[-1], ()))
    return tables


def _model_constructor(
    expression: ast.expr,
    class_tables: Mapping[str, list[str]],
    class_aliases: Mapping[str, list[str]],
    module_aliases: Mapping[str, str],
) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id in class_tables or expression.id in class_aliases
    if not isinstance(expression, ast.Attribute):
        return False
    parts = _attribute_parts(expression)
    if not parts:
        return False
    class_name = parts[-1] if parts[0] in module_aliases else expression.attr
    return class_name in class_tables or class_name in class_aliases


def _scope_nodes(statements: Sequence[ast.stmt]) -> list[ast.AST]:
    nodes = []
    pending = list(reversed(statements))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        nodes.append(node)
        pending.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes


def _table_accesses(
    tree: ast.Module,
    class_tables: Mapping[str, list[str]],
    class_aliases: Mapping[str, list[str]],
    module_aliases: Mapping[str, str],
) -> dict[str, tuple[int | None, int | None]]:
    writes: defaultdict[str, list[int]] = defaultdict(list)
    reads: defaultdict[str, list[int]] = defaultdict(list)
    model_aliases = {name: list(tables) for name, tables in class_aliases.items()}
    module_assignments = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    while True:
        changed = False
        for assignment in module_assignments:
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            value = assignment.value
            if value is None or any(
                isinstance(node, ast.Call) for node in ast.walk(value)
            ):
                continue
            tables = _table_references(
                value,
                class_tables,
                model_aliases,
                module_aliases,
            )
            if not tables:
                continue
            for target in targets:
                names = {
                    node.id
                    for node in ast.walk(target)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                }
                for name in names:
                    combined = set(model_aliases.get(name, ())) | tables
                    if combined != set(model_aliases.get(name, ())):
                        model_aliases[name] = sorted(combined)
                        changed = True
        if not changed:
            break

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return_tables_by_name: defaultdict[str, list[set[str]]] = defaultdict(list)
    for function in functions:
        tables = (
            _table_references(
                function.returns,
                class_tables,
                model_aliases,
                module_aliases,
            )
            if function.returns is not None
            else set()
        )
        for node in _scope_nodes(function.body):
            if isinstance(node, ast.Return) and node.value is not None:
                tables.update(
                    _table_references(
                        node.value,
                        class_tables,
                        model_aliases,
                        module_aliases,
                    )
                )
        return_tables_by_name[function.name].append(tables)
    function_returns = {
        name: variants[0]
        for name, variants in return_tables_by_name.items()
        if all(variant == variants[0] for variant in variants)
    }

    module_nodes = _scope_nodes(tree.body)
    scoped_calls = [
        (node, model_aliases) for node in module_nodes if isinstance(node, ast.Call)
    ]
    scoped_attributes = [(node, model_aliases) for node in module_nodes]
    for function in functions:
        function_aliases = {
            name: list(tables) for name, tables in model_aliases.items()
        }
        positional = [*function.args.posonlyargs, *function.args.args]
        default_offset = len(positional) - len(function.args.defaults)
        arguments = [
            (
                argument,
                function.args.defaults[index - default_offset]
                if index >= default_offset
                else None,
            )
            for index, argument in enumerate(positional)
        ]
        arguments.extend(
            zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True)
        )
        arguments.extend(
            (argument, None)
            for argument in (function.args.vararg, function.args.kwarg)
            if argument is not None
        )
        for argument, default in arguments:
            tables = (
                _table_references(
                    argument.annotation,
                    class_tables,
                    function_aliases,
                    module_aliases,
                )
                if argument.annotation is not None
                else set()
            )
            if default is not None:
                tables.update(
                    _table_references(
                        default,
                        class_tables,
                        function_aliases,
                        module_aliases,
                    )
                )
            if tables:
                function_aliases[argument.arg] = sorted(
                    set(function_aliases.get(argument.arg, ())) | tables
                )

        function_nodes = _scope_nodes(function.body)
        while True:
            changed = False
            for node in function_nodes:
                if isinstance(node, ast.Assign):
                    targets = node.targets
                    value = node.value
                    tables = _table_references(
                        value,
                        class_tables,
                        function_aliases,
                        module_aliases,
                    )
                    if isinstance(value, ast.Call):
                        tables.update(function_returns.get(_call_name(value.func), ()))
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                    tables = (
                        _table_references(
                            node.annotation,
                            class_tables,
                            function_aliases,
                            module_aliases,
                        )
                        if node.annotation is not None
                        else set()
                    )
                    if node.value is not None:
                        tables.update(
                            _table_references(
                                node.value,
                                class_tables,
                                function_aliases,
                                module_aliases,
                            )
                        )
                        if isinstance(node.value, ast.Call):
                            tables.update(
                                function_returns.get(
                                    _call_name(node.value.func),
                                    (),
                                )
                            )
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    targets = [node.target]
                    value = node.iter
                    tables = _table_references(
                        value,
                        class_tables,
                        function_aliases,
                        module_aliases,
                    )
                    if isinstance(value, ast.Call):
                        tables.update(function_returns.get(_call_name(value.func), ()))
                else:
                    continue
                if not tables:
                    continue
                for target in targets:
                    names = {
                        target_node.id
                        for target_node in ast.walk(target)
                        if isinstance(target_node, ast.Name)
                        and isinstance(target_node.ctx, ast.Store)
                    }
                    for name in names:
                        combined = set(function_aliases.get(name, ())) | tables
                        if combined != set(function_aliases.get(name, ())):
                            function_aliases[name] = sorted(combined)
                            changed = True
            if not changed:
                break
        scoped_calls.extend(
            (node, function_aliases)
            for node in function_nodes
            if isinstance(node, ast.Call)
        )
        scoped_attributes.extend((node, function_aliases) for node in function_nodes)

    query_calls_used_for_writes = set()
    for call, aliases in scoped_calls:
        if _call_name(call.func) not in _CHAINED_WRITE_CALLS or not isinstance(
            call.func, ast.Attribute
        ):
            continue
        receiver = call.func.value
        for nested in ast.walk(receiver):
            if isinstance(nested, ast.Call) and _call_name(nested.func) in _READ_CALLS:
                query_calls_used_for_writes.add(id(nested))
                expression = ast.Tuple(elts=nested.args, ctx=ast.Load())
                writes_for_query = _table_references(
                    expression,
                    class_tables,
                    aliases,
                    module_aliases,
                )
                for table in writes_for_query:
                    writes[table].append(call.lineno)

    for call, aliases in scoped_calls:
        name = _call_name(call.func)
        if name in _BULK_WRITE_CALLS | _WRITE_CALLS:
            if call.args:
                for table in _table_references(
                    call.args[0],
                    class_tables,
                    aliases,
                    module_aliases,
                ):
                    writes[table].append(call.lineno)
            continue
        if name in _READ_CALLS:
            if id(call) in query_calls_used_for_writes:
                continue
            expression = ast.Tuple(elts=call.args, ctx=ast.Load())
            for table in _table_references(
                expression,
                class_tables,
                aliases,
                module_aliases,
            ):
                reads[table].append(call.lineno)
            continue
        if isinstance(call.func, ast.Attribute) and call.func.attr.startswith(
            _READ_METHOD_PREFIXES
        ):
            for table in _table_references(
                call.func.value,
                class_tables,
                aliases,
                module_aliases,
            ):
                reads[table].append(call.lineno)
            continue
        if _model_constructor(
            call.func,
            class_tables,
            aliases,
            module_aliases,
        ):
            for table in _table_references(
                call.func,
                class_tables,
                aliases,
                module_aliases,
            ):
                writes[table].append(call.lineno)

    for node, aliases in scoped_attributes:
        if isinstance(node, ast.Attribute) and node.attr == "query":
            for table in _table_references(
                node.value,
                class_tables,
                aliases,
                module_aliases,
            ):
                reads[table].append(node.lineno)
    return {
        table: (
            min(writes[table]) if writes[table] else None,
            min(reads[table]) if reads[table] else None,
        )
        for table in sorted(writes.keys() | reads.keys())
    }


def _reexport_only(tree: ast.Module) -> bool:
    return all(
        isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr, ast.Assign))
        for node in tree.body
    )


def _parse_source(files: Mapping[str, str], path: str) -> ast.Module | None:
    try:
        return ast.parse(files[path], filename=path)
    except SyntaxError:
        return None


def _parse_graph(
    snapshot: _Snapshot,
    tables: Mapping[str, Table],
    domain_table_names: set[str],
) -> tuple[dict[str, _GraphFile], int]:
    paths = sorted(
        path
        for path in snapshot.files
        if path.startswith("backend/src/")
        and path.endswith(".py")
        and not _is_excluded(path)
    )
    module_to_path = {}
    parsed: dict[str, _GraphFile] = {}
    unparseable = 0
    trees = parallel_map(
        _parse_source,
        snapshot.files,
        paths,
        min_items=PARALLEL_MIN_FILES,
        chunksize=PARALLEL_FILE_CHUNKSIZE,
        desc="Parsing backend files",
        unit="file",
    )
    for path, tree in zip(paths, trees, strict=True):
        if tree is None:
            unparseable += 1
            continue
        module = _module_name(path)
        module_to_path[module] = path
        parsed[path] = _GraphFile(
            path=path,
            module=module,
            source=snapshot.files[path],
            tree=tree,
            reexport_only=_reexport_only(tree),
        )

    packages = {module.split(".")[0] for module in module_to_path if module}
    class_tables: defaultdict[str, list[str]] = defaultdict(list)
    tables_by_path: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for table in tables.values():
        if table.name not in domain_table_names:
            continue
        class_tables[table.class_name].append(table.name)
        tables_by_path[table.source.path].append((table.class_name, table.name))
    for names in class_tables.values():
        names.sort()

    for path in sorted(parsed):
        graph_file = parsed[path]
        aliases: dict[str, str] = {}
        class_aliases: dict[str, list[str]] = {}
        imports: set[str] = set()
        touched: set[str] = set()
        for node in ast.walk(graph_file.tree):
            if isinstance(node, ast.ImportFrom):
                module = _relative_module(
                    graph_file.module,
                    node,
                    is_package=path.endswith("/__init__.py"),
                )
                if not module or module.split(".")[0] not in packages:
                    continue
                for alias in node.names:
                    full = f"{module}.{alias.name}"
                    if alias.name in class_tables:
                        class_aliases[alias.asname or alias.name] = class_tables[
                            alias.name
                        ]
                    if full in module_to_path:
                        imports.add(full)
                        aliases[alias.asname or alias.name] = full
                    else:
                        if module in module_to_path:
                            imports.add(module)
                        touched.update(class_tables.get(alias.name, ()))
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in packages:
                        continue
                    imports.add(alias.name)
                    aliases[alias.asname or alias.name] = alias.name
                continue
            if (
                isinstance(node, ast.Attribute)
                and node.attr in class_tables
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                touched.update(class_tables[node.attr])

        defined = {
            table.name
            for table in tables.values()
            if table.name in domain_table_names and table.source.path == path
        }
        touched.update(defined)
        graph_file.imports = sorted(
            module_to_path[module]
            for module in imports
            if module in module_to_path and module_to_path[module] != path
        )
        graph_file.tables = touched
        graph_file.defines = defined
        for module in graph_file.imports:
            graph_file.import_lines[module] = _first_import_line(
                graph_file.tree,
                graph_file.module,
                path.endswith("/__init__.py"),
                module,
                module_to_path,
            )
        for table_name in sorted(touched):
            graph_file.table_lines[table_name] = _table_line(
                path, table_name, tables, graph_file.tree
            )
        graph_file.table_accesses = _table_accesses(
            graph_file.tree,
            class_tables,
            class_aliases,
            aliases,
        )
    return parsed, unparseable


def _cached_parse_graph(
    repo: Path,
    commit: str,
    snapshot: _Snapshot,
    tables: Mapping[str, Table],
    domain_table_names: set[str],
) -> tuple[dict[str, _GraphFile], int]:
    signature = json.dumps(
        {
            "domain_tables": sorted(domain_table_names),
            "tables": sorted(
                (table.name, table.class_name, table.source.path)
                for table in tables.values()
            ),
        },
        separators=(",", ":"),
    )
    graph_key = hashlib.sha256(signature.encode()).hexdigest()
    return get_or_compute(
        commit,
        f"parse-graph-{graph_key}",
        lambda: _parse_graph(snapshot, tables, domain_table_names),
    )


def _first_import_line(
    tree: ast.Module,
    importing_module: str,
    is_package: bool,
    target: str,
    module_to_path: Mapping[str, str],
) -> int:
    target_module = next(
        (module for module, path in module_to_path.items() if path == target), target
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == target_module for alias in node.names):
                return node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = _relative_module(
                importing_module,
                node,
                is_package=is_package,
            )
            if module == target_module or any(
                f"{module}.{alias.name}" == target_module for alias in node.names
            ):
                return node.lineno
    return 1


def _table_line(
    path: str,
    table_name: str,
    tables: Mapping[str, Table],
    tree: ast.Module,
) -> int:
    table = tables[table_name]
    if table.source.path != path:
        return 1
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == table.class_name:
            return node.lineno
    return table.source.lines[0]


def _domain_scope(
    graph: Mapping[str, _GraphFile],
    domain_tables: Mapping[str, Collection[str]],
) -> tuple[dict[str, list[str]], dict[str, DomainReason]]:
    table_domain = {
        table: domain_id
        for domain_id, tables in domain_tables.items()
        for table in tables
    }
    table_counts = Counter(
        table for graph_file in graph.values() for table in graph_file.tables
    )
    file_count = len(graph)
    idf = {
        table: math.log(file_count / count)
        for table, count in table_counts.items()
        if count
    }
    domains: dict[str, list[str]] = {}
    reasons: dict[str, DomainReason] = {}
    for path in sorted(graph):
        graph_file = graph[path]
        if not graph_file.tables or (
            path.endswith("__init__.py") and graph_file.reexport_only
        ):
            continue
        scores = Counter()
        for table in sorted(graph_file.tables):
            domain_id = table_domain.get(table)
            if domain_id is not None and table in idf:
                scores[domain_id] += idf[table]
        if not scores:
            continue
        highest = max(scores.values())
        domains[path] = sorted(
            domain_id for domain_id, score in scores.items() if score >= highest - 1e-9
        )
        reasons[path] = "tables"

    importers: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(graph):
        for imported in graph[path].imports:
            importers[imported].append(path)

    def vote(neighbors: Collection[str], share: float) -> str | None:
        counter = Counter(
            domain_id
            for neighbor in neighbors
            if neighbor in domains
            for domain_id in domains[neighbor]
        )
        total = sum(1 for neighbor in neighbors if neighbor in domains)
        if not total or not counter:
            return None
        domain_id = min(counter, key=lambda item: (-counter[item], item))
        count = counter[domain_id]
        return domain_id if count / total > share else None

    for _ in range(2):
        new: dict[str, str] = {}
        for path in sorted(graph):
            if path in domains:
                continue
            domain_id = vote(graph[path].imports, 0.5)
            if domain_id is not None:
                new[path] = domain_id
        for path, domain_id in new.items():
            domains[path] = [domain_id]
            reasons[path] = "calls"

    new: dict[str, str] = {}
    for path in sorted(graph):
        if path in domains:
            continue
        domain_id = vote(importers[path], 0.999)
        if domain_id is not None:
            new[path] = domain_id
    for path, domain_id in new.items():
        domains[path] = [domain_id]
        reasons[path] = "called_by"
    for path, domain_ids in sorted(domains.items()):
        graph[path].domains = domain_ids
        graph[path].domain_reason = reasons[path]
    return domains, reasons


def _normalise_link(a: str, b: str, signal: str, path: str, line: int) -> _RawLink:
    left, right = sorted((a, b))
    return _RawLink(left, right, signal, path, line)


def _foreign_key_line(
    graph: Mapping[str, _GraphFile],
    tables: Mapping[str, Table],
    table_name: str,
    target: str,
) -> int:
    table = tables[table_name]
    graph_file = graph.get(table.source.path)
    if graph_file is not None:
        for node in ast.walk(graph_file.tree):
            if not isinstance(node, ast.ClassDef) or node.name != table.class_name:
                continue
            for call in ast.walk(node):
                if (
                    not isinstance(call, ast.Call)
                    or _call_name(call.func) != "ForeignKey"
                ):
                    continue
                if any(
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.split(".", 1)[0] == target
                    for argument in call.args
                ):
                    return call.lineno
    return table.source.lines[0]


def _table_links(
    repo: Path,
    commit: str,
    graph: Mapping[str, _GraphFile],
    tables: Mapping[str, Table],
    domain_table_names: list[str],
    relation_candidates: Collection[RelationCandidate] | None = None,
) -> dict[tuple[str, str], list[_RawLink]]:
    table_set = set(domain_table_names)
    links: defaultdict[tuple[str, str], dict[str, _RawLink]] = defaultdict(dict)
    for table_name in domain_table_names:
        table = tables[table_name]
        for column in table.columns:
            if not column.foreign_key:
                continue
            target = column.foreign_key.split(".", 1)[0]
            if target not in table_set or target == table_name:
                continue
            link = _normalise_link(
                table_name,
                target,
                "fk",
                table.source.path,
                _foreign_key_line(graph, tables, table_name, target),
            )
            links[(link.a, link.b)][link.signal] = link
        for other_name in domain_table_names:
            if other_name == table_name:
                continue
            other = tables[other_name]
            if other.source.path == table.source.path:
                link = _normalise_link(
                    table_name,
                    other_name,
                    "same_file",
                    table.source.path,
                    table.source.lines[0],
                )
                links[(link.a, link.b)][link.signal] = link

    candidates = (
        relation_candidates
        if relation_candidates is not None
        else find_relation_candidates(repo, commit, domain_table_names)
    )
    for candidate in tqdm(candidates, desc="Relation candidates", unit="candidate"):
        source_table = candidate.src.split(".", 1)[0]
        target_table = candidate.dst.split(".", 1)[0]
        if source_table not in table_set or target_table not in table_set:
            continue
        if source_table == target_table:
            continue
        link = _normalise_link(
            source_table,
            target_table,
            candidate.signal,
            candidate.path,
            candidate.line,
        )
        links[(link.a, link.b)].setdefault(link.signal, link)
    return {pair: list(signals.values()) for pair, signals in links.items()}


def _external_table_links(
    domain: str,
    domain_tables: Mapping[str, Collection[str]],
    tables: Mapping[str, Table],
    graph: Mapping[str, _GraphFile],
    candidates: Collection[RelationCandidate],
) -> list[TableLink]:
    table_domains = {
        table_name: domain_id
        for domain_id, table_names in domain_tables.items()
        for table_name in table_names
    }
    raw_links = []
    for table_name in sorted(tables):
        source_domain = table_domains.get(table_name)
        if source_domain is None:
            continue
        for column in tables[table_name].columns:
            if not column.foreign_key:
                continue
            target_table = column.foreign_key.split(".", 1)[0]
            target_domain = table_domains.get(target_table)
            if source_domain == target_domain or domain not in {
                source_domain,
                target_domain,
            }:
                continue
            raw_links.append(
                _normalise_link(
                    table_name,
                    target_table,
                    "fk",
                    tables[table_name].source.path,
                    _foreign_key_line(graph, tables, table_name, target_table),
                )
            )
    for candidate in candidates:
        if candidate.signal == "same_file" or candidate.signal.endswith(":ambiguous"):
            continue
        source_table = candidate.src.split(".", 1)[0]
        target_table = candidate.dst.split(".", 1)[0]
        source_domain = table_domains.get(source_table)
        target_domain = table_domains.get(target_table)
        if (
            source_domain is None
            or target_domain is None
            or source_domain == target_domain
            or domain not in {source_domain, target_domain}
        ):
            continue
        raw_links.append(
            _normalise_link(
                source_table,
                target_table,
                candidate.signal,
                candidate.path,
                candidate.line,
            )
        )
    links: dict[tuple[str, str, str], _RawLink] = {}
    for link in sorted(
        raw_links,
        key=lambda item: (item.a, item.b, item.signal, item.path, item.line),
    ):
        links.setdefault((link.a, link.b, link.signal), link)
    return [
        TableLink(
            a=link.a,
            b=link.b,
            signal=link.signal,
            path=link.path,
            line=link.line,
        )
        for link in links.values()
    ]


def _cross_feature_links(
    links: Mapping[tuple[str, str], list[_RawLink]],
    feature_by_table: Mapping[str, int],
) -> list[TableLink]:
    return [
        TableLink(
            a=link.a,
            b=link.b,
            signal=link.signal,
            path=link.path,
            line=link.line,
        )
        for pair in sorted(links)
        for link in sorted(
            links[pair],
            key=lambda item: (item.a, item.b, item.signal, item.path, item.line),
        )
        if feature_by_table[link.a] != feature_by_table[link.b]
        and link.signal != "same_file"
        and not link.signal.endswith(":ambiguous")
    ]


def _clusters(
    tables: list[str],
    users: Mapping[str, set[str]],
    links: Mapping[tuple[str, str], list[_RawLink]],
    threshold: float,
) -> list[list[str]]:
    linked = set(links)

    def similarity(left: str, right: str) -> float:
        left_users = users[left]
        right_users = users[right]
        union = left_users | right_users
        jaccard = len(left_users & right_users) / len(union) if union else 0
        pair = tuple(sorted((left, right)))
        return jaccard + (0.5 if pair in linked else 0)

    clusters = [[table] for table in tables]
    while True:
        best: tuple[float, int, int] | None = None
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                score = sum(
                    similarity(left, right)
                    for left in clusters[left_index]
                    for right in clusters[right_index]
                ) / (len(clusters[left_index]) * len(clusters[right_index]))
                if best is None or score > best[0]:
                    best = (score, left_index, right_index)
        if best is None or best[0] < threshold:
            return clusters
        _, left_index, right_index = best
        clusters[left_index] = sorted([*clusters[left_index], *clusters[right_index]])
        del clusters[right_index]


def _evidence_sort_key(evidence: FeatureEvidence) -> tuple[str, str, int, str]:
    value = evidence.table or evidence.to or evidence.tag or evidence.target or ""
    return evidence.kind, evidence.path, evidence.line, value


def _bounded_evidence(evidence: Collection[FeatureEvidence]) -> list[FeatureEvidence]:
    unique = {
        (
            item.kind,
            item.path,
            item.line,
            item.table,
            item.to,
            item.tag,
            item.target,
        ): item
        for item in evidence
    }
    return sorted(unique.values(), key=_evidence_sort_key)[:5]


def _backend_evidence(
    path: str,
    feature: int,
    owner: Mapping[str, int],
    graph: Mapping[str, _GraphFile],
    touch: Mapping[str, set[str]],
    primary_files: Mapping[int, set[str]],
) -> list[FeatureEvidence]:
    evidence: list[FeatureEvidence] = []
    graph_file = graph[path]
    for table in sorted(touch[path]):
        peers = [
            peer
            for peer in primary_files[feature]
            if peer != path and table in touch.get(peer, set())
        ]
        if peers:
            evidence.append(
                FeatureEvidence(
                    kind="table",
                    table=table,
                    path=path,
                    line=graph_file.table_lines.get(table, 1),
                )
            )
    for imported in sorted(graph_file.imports):
        if owner.get(imported) == feature and imported != path:
            evidence.append(
                FeatureEvidence(
                    kind="call",
                    to=imported,
                    path=path,
                    line=graph_file.import_lines.get(imported, 1),
                )
            )
    for importer in sorted(graph):
        importer_file = graph[importer]
        if owner.get(importer) != feature or importer == path:
            continue
        if path not in importer_file.imports:
            continue
        evidence.append(
            FeatureEvidence(
                kind="call",
                to=importer,
                path=importer,
                line=importer_file.import_lines.get(path, 1),
            )
        )
    return _bounded_evidence(evidence)


def _has_backend_evidence(
    path: str,
    feature: int,
    owner: Mapping[str, int],
    graph: Mapping[str, _GraphFile],
    touch: Mapping[str, set[str]],
    primary_files: Mapping[int, set[str]],
) -> bool:
    return bool(_backend_evidence(path, feature, owner, graph, touch, primary_files))


def _resolve_web_import(
    path: str,
    imported: str,
    web_files: set[str],
) -> str | None:
    imported = imported.split("?", 1)[0]
    if imported.startswith("#/"):
        base = PurePosixPath("web/apps/web/src") / imported[2:]
    elif imported.startswith("."):
        base = PurePosixPath(path).parent / imported
    else:
        return None
    candidate = str(base)
    options = [candidate]
    if not candidate.endswith((".ts", ".tsx")):
        options.extend(
            [
                f"{candidate}.ts",
                f"{candidate}.tsx",
                f"{candidate}/index.ts",
                f"{candidate}/index.tsx",
            ]
        )
    return next((option for option in options if option in web_files), None)


def _line_number(source: str, position: int) -> int:
    return source.count("\n", 0, position) + 1


def _namespace_map(
    graph: Mapping[str, _GraphFile],
) -> dict[str, tuple[str, int]]:
    namespaces: dict[str, tuple[str, int]] = {}
    for path in sorted(graph):
        graph_file = graph[path]
        for node in ast.walk(graph_file.tree):
            if (
                not isinstance(node, ast.Call)
                or _call_name(node.func) != _NAMESPACE_CALL
            ):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    namespaces.setdefault(
                        keyword.value.value.replace("_", "-"),
                        (path, keyword.value.lineno),
                    )
    return namespaces


def _route_tags(path: str, source: str) -> dict[str, int]:
    tags = {}
    for match in _GENERATED_ROUTE.finditer(path + "\n" + source):
        if match.group(1) != match.group(2):
            continue
        position = match.start()
        if position < len(path) + 1:
            line = 1
        else:
            line = _line_number(source, position - len(path) - 1)
        tags.setdefault(match.group(1), line)
    return tags


def _web_imports(path: str, source: str) -> list[tuple[str, int]]:
    imports = []
    for match in _TS_IMPORT.finditer(source):
        imported = next((group for group in match.groups() if group), None)
        if imported is not None:
            imports.append((imported, _line_number(source, match.start())))
    return imports


def _primary_feature_for_path(
    path: str,
    owner: Mapping[str, int],
) -> int | None:
    return owner.get(path)


def _external_placements(
    snapshot: _Snapshot,
    primary_owner: Mapping[str, int],
    graph: Mapping[str, _GraphFile],
    category_paths: Collection[str],
) -> dict[str, list[_Placement]]:
    token_targets: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, feature in sorted(primary_owner.items()):
        relative = path.removeprefix("backend/src/")
        if len(PurePosixPath(relative).parts) >= 2:
            token_targets[path].append((path, feature))
            token_targets[relative].append((path, feature))
        module = graph[path].module
        if len(module.split(".")) >= 2:
            token_targets[module].append((path, feature))
    reference_pattern = re.compile(
        r"(?<![\w])(?:"
        + "|".join(
            re.escape(token)
            for token in sorted(token_targets, key=lambda value: (-len(value), value))
        )
        + r")(?![\w])"
    )

    results: dict[str, list[_Placement]] = {}
    for path in sorted(category_paths):
        source = snapshot.files[path]
        by_feature: Counter[int] = Counter()
        evidence_by_feature: defaultdict[int, list[FeatureEvidence]] = defaultdict(list)
        for line_number, line in enumerate(source.splitlines(), 1):
            seen_tokens = set()
            for match in reference_pattern.finditer(line):
                token = match.group()
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                for _, feature in sorted(token_targets[token]):
                    by_feature[feature] += 1
                    evidence_by_feature[feature].append(
                        FeatureEvidence(
                            kind="reference",
                            target=token,
                            path=path,
                            line=line_number,
                        )
                    )
        if not by_feature:
            continue
        primary = _most_common_lowest(by_feature)
        results[path] = [
            _Placement(
                feature=feature,
                primary=feature == primary,
                reason="reference",
                evidence=_bounded_evidence(evidence_by_feature[feature]),
            )
            for feature in sorted(by_feature)
        ]
    return results


def _validate_evidence(
    snapshot: _Snapshot,
    feature_map: FeatureMap,
) -> None:
    for feature in feature_map.features:
        for file in feature.files:
            for evidence in file.evidence:
                source = snapshot.files.get(evidence.path)
                if source is None:
                    raise ValueError(
                        f"evidence path '{evidence.path}' is not in {feature.id}"
                    )
                if not 1 <= evidence.line <= len(source.splitlines()):
                    raise ValueError(
                        f"evidence line {evidence.line} for '{evidence.path}' "
                        "is outside the pinned source"
                    )


def _prepare_feature_maps(
    repo: Path,
    commit: str,
    domain_tables: Mapping[str, Collection[str]],
) -> tuple[
    _Snapshot,
    dict[str, Table],
    dict[str, _GraphFile],
    int,
    dict[str, list[str]],
    dict[str, DomainReason],
]:
    snapshot = _snapshot(repo, commit)
    tables = extract_tables(repo, commit)
    all_domain_tables = {table for names in domain_tables.values() for table in names}
    graph, parse_counts = _cached_parse_graph(
        repo,
        commit,
        snapshot,
        tables,
        all_domain_tables,
    )
    domains, domain_reasons = _domain_scope(graph, domain_tables)
    return snapshot, tables, graph, parse_counts, domains, domain_reasons


def build_feature_maps(
    repo: Path,
    commit: str,
    domain_tables: Mapping[str, Collection[str]],
    *,
    threshold: float = 0.25,
) -> dict[str, FeatureMap]:
    snapshot, tables, graph, parse_counts, domains, domain_reasons = (
        _prepare_feature_maps(repo, commit, domain_tables)
    )
    inputs = _FeatureMapInputs(
        repo=repo,
        commit=commit,
        domain_tables=domain_tables,
        snapshot=snapshot,
        tables=tables,
        graph=graph,
        parse_counts=parse_counts,
        domains=domains,
        domain_reasons=domain_reasons,
        threshold=threshold,
    )
    domain_ids = sorted(domain_tables)
    dumps = parallel_map(
        _feature_map_dump,
        inputs,
        domain_ids,
        desc="Building domain feature maps",
        unit="domain",
    )
    return {
        domain: FeatureMap.model_validate(dump)
        for domain, dump in zip(domain_ids, dumps, strict=True)
    }


@dataclass(frozen=True)
class _FeatureMapInputs:
    repo: Path
    commit: str
    domain_tables: Mapping[str, Collection[str]]
    snapshot: _Snapshot
    tables: dict[str, Table]
    graph: dict[str, _GraphFile]
    parse_counts: int
    domains: dict[str, list[str]]
    domain_reasons: dict[str, DomainReason]
    threshold: float


def _feature_map_dump(inputs: _FeatureMapInputs, domain: str) -> dict:
    return _build_feature_map(
        inputs.repo,
        inputs.commit,
        inputs.domain_tables,
        inputs.snapshot,
        inputs.tables,
        inputs.graph,
        inputs.parse_counts,
        inputs.domains,
        inputs.domain_reasons,
        domain,
        threshold=inputs.threshold,
    ).model_dump()


def build_feature_map(
    repo: Path,
    commit: str,
    domain_tables: Mapping[str, Collection[str]],
    domain: str,
    *,
    threshold: float = 0.25,
) -> FeatureMap:
    if domain not in domain_tables:
        raise ValueError(f"unknown domain '{domain}'")
    snapshot, tables, graph, parse_counts, domains, domain_reasons = (
        _prepare_feature_maps(repo, commit, domain_tables)
    )
    return _build_feature_map(
        repo,
        commit,
        domain_tables,
        snapshot,
        tables,
        graph,
        parse_counts,
        domains,
        domain_reasons,
        domain,
        threshold=threshold,
    )


def _build_feature_map(
    repo: Path,
    commit: str,
    domain_tables: Mapping[str, Collection[str]],
    snapshot: _Snapshot,
    tables: dict[str, Table],
    graph: dict[str, _GraphFile],
    parse_counts: int,
    domains: dict[str, list[str]],
    domain_reasons: dict[str, DomainReason],
    domain: str,
    *,
    threshold: float,
) -> FeatureMap:
    domain_table_names = sorted(set(domain_tables[domain]))
    domain_files = sorted(path for path in domains if domain in domains[path])
    touch = {
        path: graph[path].tables & set(domain_table_names) for path in domain_files
    }
    users = {
        table: {
            path
            for path in domain_files
            if table in touch[path]
            and not path.endswith("__init__.py")
            and table not in graph[path].defines
        }
        for table in domain_table_names
    }
    all_table_names = sorted(tables)
    relation_candidates = find_relation_candidates(repo, commit, all_table_names)
    links = _table_links(
        repo,
        commit,
        graph,
        tables,
        domain_table_names,
        relation_candidates,
    )
    external_links = _external_table_links(
        domain,
        domain_tables,
        tables,
        graph,
        relation_candidates,
    )
    clustering_links = {
        pair: pair_links
        for pair, pair_links in links.items()
        if any(link.signal != "id_column" for link in pair_links)
    }
    clusters = _clusters(
        domain_table_names,
        users,
        clustering_links,
        threshold,
    )
    cluster_by_table = {
        table: index for index, cluster in enumerate(clusters) for table in cluster
    }
    cross_links = _cross_feature_links(links, cluster_by_table)
    local_idf = {
        table: math.log(len(domain_files) / max(1, len(users[table])))
        for table in domain_table_names
    }
    owner: dict[str, int] = {}
    for path in domain_files:
        if not touch[path]:
            continue
        scores = [
            sum(
                local_idf[table]
                for table in sorted(touch[path])
                if cluster_by_table[table] == feature
            )
            for feature in range(len(clusters))
        ]
        owner[path] = _lowest_score_feature(scores)
    imports = {
        path: set(graph[path].imports) & set(domain_files) for path in domain_files
    }
    for _ in range(3):
        for path in domain_files:
            if path in owner:
                continue
            neighbors = [
                owner[neighbor]
                for neighbor in sorted(imports[path])
                if neighbor in owner
            ]
            neighbors.extend(
                owner[neighbor]
                for neighbor in domain_files
                if path in imports[neighbor] and neighbor in owner
            )
            if neighbors:
                owner[path] = _most_common_lowest(Counter(neighbors))

    primary_files = {
        feature: {
            path
            for path, assigned_feature in owner.items()
            if assigned_feature == feature
        }
        for feature in range(len(clusters))
    }
    valid_owner: dict[str, int] = {}
    uncovered: dict[str, str] = {}
    for path, feature in sorted(owner.items()):
        if _has_backend_evidence(
            path,
            feature,
            owner,
            graph,
            touch,
            primary_files,
        ):
            valid_owner[path] = feature
        else:
            uncovered[path] = "no shared table or call in its feature"

    placements: defaultdict[str, list[_Placement]] = defaultdict(list)
    for path, feature in sorted(valid_owner.items()):
        reason = domain_reasons[path]
        placements[path].append(
            _Placement(
                feature=feature,
                primary=True,
                reason=reason,
                evidence=_backend_evidence(
                    path,
                    feature,
                    valid_owner,
                    graph,
                    touch,
                    {
                        feature_id: {
                            item
                            for item, assigned in valid_owner.items()
                            if assigned == feature_id
                        }
                        for feature_id in range(len(clusters))
                    },
                ),
            )
        )
        if touch[path]:
            primary_score = sum(
                local_idf[table]
                for table in sorted(touch[path])
                if cluster_by_table[table] == feature
            )
            for other_feature in range(len(clusters)):
                if other_feature == feature:
                    continue
                other_score = sum(
                    local_idf[table]
                    for table in sorted(touch[path])
                    if cluster_by_table[table] == other_feature
                )
                if other_score == 0 or other_score < 0.5 * primary_score:
                    continue
                evidence = _backend_evidence(
                    path,
                    other_feature,
                    valid_owner,
                    graph,
                    touch,
                    {
                        feature_id: {
                            item
                            for item, assigned in valid_owner.items()
                            if assigned == feature_id
                        }
                        for feature_id in range(len(clusters))
                    },
                )
                if evidence:
                    placements[path].append(
                        _Placement(
                            feature=other_feature,
                            primary=False,
                            reason=reason,
                            evidence=evidence,
                        )
                    )

    namespace_map = _namespace_map(graph)
    web_paths = sorted(
        path
        for path in snapshot.files
        if path.startswith("web/")
        and path.endswith((".ts", ".tsx"))
        and not _is_excluded(path)
    )
    web_files = set(web_paths)
    web_placements: dict[str, list[_Placement]] = {}
    unmapped_tags: set[str] = set()
    for path in web_paths:
        tags = _route_tags(path, snapshot.files[path])
        feature_tags: defaultdict[int, list[str]] = defaultdict(list)
        route_evidence: defaultdict[int, list[FeatureEvidence]] = defaultdict(list)
        for tag, line in tags.items():
            endpoint = namespace_map.get(tag)
            if endpoint is None:
                unmapped_tags.add(tag)
                continue
            endpoint_path, endpoint_line = endpoint
            endpoint_feature = valid_owner.get(endpoint_path)
            if endpoint_feature is None:
                continue
            feature_tags[endpoint_feature].append(tag)
            route_evidence[endpoint_feature].extend(
                [
                    FeatureEvidence(
                        kind="route",
                        tag=tag,
                        path=path,
                        line=line,
                    ),
                    FeatureEvidence(
                        kind="route",
                        tag=tag,
                        path=endpoint_path,
                        line=endpoint_line,
                    ),
                ]
            )
        if not feature_tags:
            continue
        primary = _most_common_lowest(
            Counter({feature: len(tagged) for feature, tagged in feature_tags.items()})
        )
        web_placements[path] = [
            _Placement(
                feature=feature,
                primary=feature == primary,
                reason="route",
                evidence=_bounded_evidence(route_evidence[feature]),
            )
            for feature in sorted(feature_tags)
        ]

    for _ in range(2):
        placed_snapshot = web_placements.copy()
        new: dict[str, list[_Placement]] = {}
        for path in web_paths:
            if path in placed_snapshot or _route_tags(path, snapshot.files[path]):
                continue
            resolved_imports = [
                (resolved, line)
                for imported, line in _web_imports(path, snapshot.files[path])
                if (resolved := _resolve_web_import(path, imported, web_files))
            ]
            placed_imports = [
                (imported_path, line)
                for imported_path, line in resolved_imports
                if imported_path in placed_snapshot
            ]
            imported_features = [
                placement.feature
                for imported_path, _ in placed_imports
                for placement in placed_snapshot[imported_path]
                if placement.primary
            ]
            if not imported_features or not resolved_imports:
                continue
            counts = Counter(imported_features)
            feature = _most_common_lowest(counts)
            count = counts[feature]
            if count / len(resolved_imports) <= 0.5:
                continue
            evidence = [
                FeatureEvidence(
                    kind="call",
                    to=imported_path,
                    path=path,
                    line=line,
                )
                for imported_path, line in placed_imports
            ]
            new[path] = [
                _Placement(
                    feature=feature,
                    primary=True,
                    reason="calls",
                    evidence=_bounded_evidence(evidence),
                )
            ]
        web_placements.update(new)

    deployment_paths = {
        path
        for path in snapshot.files
        if not _is_excluded(path) and _is_deployment_path(path)
    }
    skill_paths = {
        path
        for path in snapshot.files
        if path.startswith(".agents/skills/") and not _is_excluded(path)
    }
    external_placements = _external_placements(
        snapshot,
        valid_owner,
        graph,
        deployment_paths | skill_paths,
    )
    feature_files: defaultdict[int, list[FeatureFile]] = defaultdict(list)
    for path, file_placements in sorted(placements.items()):
        for placement in file_placements:
            feature_files[placement.feature].append(
                FeatureFile(
                    path=path,
                    primary=placement.primary,
                    reason=placement.reason,
                    evidence=placement.evidence,
                )
            )
    for path, file_placements in sorted(web_placements.items()):
        for placement in file_placements:
            feature_files[placement.feature].append(
                FeatureFile(
                    path=path,
                    primary=placement.primary,
                    reason=placement.reason,
                    evidence=placement.evidence,
                )
            )
    for path, file_placements in sorted(external_placements.items()):
        for placement in file_placements:
            feature_files[placement.feature].append(
                FeatureFile(
                    path=path,
                    primary=placement.primary,
                    reason=placement.reason,
                    evidence=placement.evidence,
                )
            )

    table_features = []
    for feature_index, cluster in enumerate(clusters):
        feature_id = min(
            cluster,
            key=lambda table: (-len(users[table]), table),
        )
        raw_files = sorted(
            feature_files[feature_index],
            key=lambda file: file.path,
        )
        raw_links = [
            link
            for pair, pair_links in links.items()
            if set(pair).issubset(cluster)
            for link in pair_links
        ]
        table_features.append(
            Feature(
                id=feature_id,
                tables=sorted(cluster),
                files=raw_files,
                table_links=[
                    TableLink(
                        a=link.a,
                        b=link.b,
                        signal=link.signal,
                        path=link.path,
                        line=link.line,
                    )
                    for link in sorted(
                        raw_links,
                        key=lambda link: (
                            link.a,
                            link.b,
                            link.signal,
                            link.path,
                            link.line,
                        ),
                    )
                ],
            )
        )

    table_features.sort(
        key=lambda feature: (
            -sum(file.primary for file in feature.files),
            feature.id,
        )
    )
    for path in domain_files:
        if path not in owner:
            uncovered[path] = "no feature reached"
    uncovered_files = [
        UncoveredFile(path=path, reason=reason)
        for path, reason in sorted(uncovered.items())
    ]
    result = FeatureMap(
        commit=commit,
        domain=domain,
        threshold=threshold,
        tables=domain_table_names,
        features=table_features,
        cross_links=cross_links,
        external_links=external_links,
        uncovered=uncovered_files,
        unmapped_tags=sorted(unmapped_tags),
        counts=FeatureCounts(
            domain_files=len(domain_files),
            primary_placed=len(
                {
                    file.path
                    for feature in table_features
                    for file in feature.files
                    if file.primary
                }
            ),
            shared=sum(
                1
                for feature in table_features
                for file in feature.files
                if not file.primary
            ),
            uncovered=len(uncovered_files),
            web=len(web_placements),
            deployment=sum(
                1 for path in external_placements if path in deployment_paths
            ),
            skills=sum(1 for path in external_placements if path in skill_paths),
            unparseable=parse_counts,
        ),
    )
    _validate_evidence(snapshot, result)
    return result
