from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from rox_dox.features import _git_blobs, _git_tree

DEFAULT_PREFIXES = ("backend/src/",)
STATE_ENUM_NAME = re.compile(r"(Status|State|Stage|Phase)$")
STATE_COLUMN_SUFFIXES = ("status", "state", "stage")
ENUM_BASES = frozenset({"Enum", "StrEnum", "IntEnum"})
COLUMN_CALLS = frozenset({"Column", "mapped_column"})
SQL_ENUM_CALL = "Enum"
MAPPED_ANNOTATION = "Mapped"
ENUM_VALUE_ATTRIBUTE = "value"
PARSE_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _EnumRef:
    name: str
    module_path: str | None


@dataclass(frozen=True)
class _EnumDef:
    name: str
    line: int
    members: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _ColumnFact:
    table: str
    column: str
    line: int
    ref: _EnumRef


@dataclass(frozen=True)
class _SetFact:
    ref: _EnumRef
    member: str
    line: int


@dataclass(frozen=True)
class _TransitionFact:
    from_ref: _EnumRef
    from_member: str
    to_ref: _EnumRef
    to_member: str
    line: int


@dataclass
class _FileFacts:
    path: str
    parse_error: str | None = None
    enums: list[_EnumDef] = field(default_factory=list)
    columns: list[_ColumnFact] = field(default_factory=list)
    sets: list[_SetFact] = field(default_factory=list)
    transitions: list[_TransitionFact] = field(default_factory=list)


@dataclass(frozen=True)
class Member:
    name: str
    line: int


@dataclass(frozen=True)
class StateColumn:
    table: str
    column: str
    path: str
    line: int


@dataclass(frozen=True)
class StateSet:
    member: str
    path: str
    line: int


@dataclass(frozen=True)
class Transition:
    from_member: str
    to_member: str
    path: str
    line: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "from": self.from_member,
            "to": self.to_member,
            "path": self.path,
            "line": self.line,
        }


@dataclass
class StateEnum:
    enum: str
    path: str
    line: int
    members: list[Member]
    columns: list[StateColumn] = field(default_factory=list)
    sets: list[StateSet] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "enum": self.enum,
            "path": self.path,
            "line": self.line,
            "members": [asdict(member) for member in self.members],
            "columns": [asdict(column) for column in self.columns],
            "sets": [asdict(state_set) for state_set in self.sets],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


@dataclass(frozen=True)
class ParseError:
    path: str
    message: str


@dataclass(frozen=True)
class StatesReport:
    commit: str
    enums: list[StateEnum]
    parse_errors: list[ParseError]

    def to_dict(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "enums": [state_enum.to_dict() for state_enum in self.enums],
            "parse_errors": [asdict(error) for error in self.parse_errors],
        }

    def to_text(self) -> str:
        lines = []
        for state_enum in self.enums:
            members = ", ".join(member.name for member in state_enum.members)
            lines.append(
                f"{state_enum.enum}  {state_enum.path}:{state_enum.line}  "
                f"members: {members}"
            )
            lines.extend(
                f"  column {column.table}.{column.column}  {column.path}:{column.line}"
                for column in state_enum.columns
            )
            lines.extend(
                f"  set {state_set.member}  {state_set.path}:{state_set.line}"
                for state_set in state_enum.sets
            )
            lines.extend(
                f"  transition {transition.from_member} -> {transition.to_member}  "
                f"{transition.path}:{transition.line}"
                for transition in state_enum.transitions
            )
        lines.extend(
            f"parse error: {error.path}: {error.message}" for error in self.parse_errors
        )
        lines.append(
            f"{len(self.enums)} enums, "
            f"{sum(len(item.columns) for item in self.enums)} columns, "
            f"{sum(len(item.sets) for item in self.enums)} sets, "
            f"{sum(len(item.transitions) for item in self.enums)} transitions"
        )
        return "\n".join(lines)


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _import_module_path(path: str, node: ast.ImportFrom) -> str:
    module_parts = node.module.split(".") if node.module else []
    if node.level == 0:
        return "/".join(module_parts)
    package_parts = list(PurePosixPath(path).parent.parts)
    base_parts = package_parts[: len(package_parts) - (node.level - 1)]
    return "/".join([*base_parts, *module_parts])


def _imports(path: str, tree: ast.Module) -> dict[str, _EnumRef]:
    imported = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_path = _import_module_path(path, node)
        for alias in node.names:
            imported[alias.asname or alias.name] = _EnumRef(alias.name, module_path)
    return imported


class _FileScanner:
    def __init__(self, path: str, tree: ast.Module) -> None:
        self.path = path
        self.tree = tree
        self.imported = _imports(path, tree)

    def original_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name) and node.id in self.imported:
            return self.imported[node.id].name
        return _terminal_name(node)

    def enum_ref(self, node: ast.expr) -> _EnumRef | None:
        if isinstance(node, ast.Name):
            ref = self.imported.get(node.id, _EnumRef(node.id, None))
        elif isinstance(node, ast.Attribute):
            ref = _EnumRef(node.attr, None)
        else:
            return None
        return ref if STATE_ENUM_NAME.search(ref.name) else None

    def member_ref(self, node: ast.expr) -> tuple[_EnumRef, str] | None:
        if not isinstance(node, ast.Attribute):
            return None
        if (
            node.attr == ENUM_VALUE_ATTRIBUTE
            and isinstance(node.value, ast.Attribute)
            and self.enum_ref(node.value.value) is not None
        ):
            node = node.value
        ref = self.enum_ref(node.value)
        return None if ref is None else (ref, node.attr)

    def scan(self) -> _FileFacts:
        facts = _FileFacts(self.path)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                enum_def = self.enum_def(node)
                if enum_def is not None:
                    facts.enums.append(enum_def)
                facts.columns.extend(self.columns(node))
            elif isinstance(node, ast.Assign):
                facts.sets.extend(self.sets(node))
            elif isinstance(node, ast.If):
                facts.transitions.extend(self.transitions(node))
        return facts

    def enum_def(self, node: ast.ClassDef) -> _EnumDef | None:
        if not STATE_ENUM_NAME.search(node.name):
            return None
        if not any(self.original_name(base) in ENUM_BASES for base in node.bases):
            return None
        members = []
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target = statement.target
            else:
                continue
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                members.append((target.id, statement.lineno))
        return _EnumDef(node.name, node.lineno, tuple(members))

    def columns(self, node: ast.ClassDef) -> Iterator[_ColumnFact]:
        table = _table_name(node)
        if table is None:
            return
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value, annotation = statement.targets[0], statement.value, None
            elif isinstance(statement, ast.AnnAssign):
                target, value, annotation = (
                    statement.target,
                    statement.value,
                    statement.annotation,
                )
            else:
                continue
            if not isinstance(target, ast.Name) or not target.id.lower().endswith(
                STATE_COLUMN_SUFFIXES
            ):
                continue
            ref = self.column_call_ref(value) or self.mapped_ref(annotation)
            if ref is not None:
                yield _ColumnFact(table, target.id, statement.lineno, ref)

    def column_call_ref(self, value: ast.expr | None) -> _EnumRef | None:
        if not isinstance(value, ast.Call) or _terminal_name(value.func) not in (
            COLUMN_CALLS
        ):
            return None
        for argument in value.args:
            if (
                isinstance(argument, ast.Call)
                and self.original_name(argument.func) == SQL_ENUM_CALL
                and argument.args
            ):
                return self.enum_ref(argument.args[0])
        return None

    def mapped_ref(self, annotation: ast.expr | None) -> _EnumRef | None:
        if (
            isinstance(annotation, ast.Subscript)
            and _terminal_name(annotation.value) == MAPPED_ANNOTATION
        ):
            return self.enum_ref(annotation.slice)
        return None

    def sets(self, node: ast.Assign) -> Iterator[_SetFact]:
        member = self.member_ref(node.value)
        if member is None:
            return
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                yield _SetFact(member[0], member[1], node.lineno)

    def transitions(self, node: ast.If) -> Iterator[_TransitionFact]:
        test = node.test
        if (
            not isinstance(test, ast.Compare)
            or len(test.ops) != 1
            or not isinstance(test.left, ast.Attribute)
        ):
            return
        comparator = test.comparators[0]
        if isinstance(test.ops[0], ast.Eq):
            candidates = [comparator]
        elif isinstance(test.ops[0], ast.In) and isinstance(
            comparator, (ast.Tuple, ast.List, ast.Set)
        ):
            candidates = comparator.elts
        else:
            return
        sources = [
            member
            for member in (self.member_ref(item) for item in candidates)
            if member is not None
        ]
        if not sources:
            return
        compared = ast.unparse(test.left)
        for assignment in _body_assignments(node.body):
            target_member = self.member_ref(assignment.value)
            if target_member is None:
                continue
            if not any(
                isinstance(target, ast.Attribute) and ast.unparse(target) == compared
                for target in assignment.targets
            ):
                continue
            for source_ref, source_member in sources:
                yield _TransitionFact(
                    source_ref,
                    source_member,
                    target_member[0],
                    target_member[1],
                    assignment.lineno,
                )


def _table_name(node: ast.ClassDef) -> str | None:
    for statement in node.body:
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__tablename__"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            return statement.value.value
    return None


def _body_assignments(statements: Sequence[ast.stmt]) -> Iterator[ast.Assign]:
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Assign):
            yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        pending.extend(
            child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt)
        )


def _scan_file(item: tuple[str, str]) -> _FileFacts:
    path, source = item
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        return _FileFacts(path, parse_error=f"line {error.lineno}: {error.msg}")
    return _FileScanner(path, tree).scan()


def _module_matches(enum_path: str, module_path: str) -> bool:
    return any(
        enum_path == candidate or enum_path.endswith(f"/{candidate}")
        for candidate in (f"{module_path}.py", f"{module_path}/__init__.py")
    )


class _Resolver:
    def __init__(self, enums: Sequence[StateEnum]) -> None:
        self.by_name: defaultdict[str, list[StateEnum]] = defaultdict(list)
        for state_enum in enums:
            self.by_name[state_enum.enum].append(state_enum)

    def resolve(self, ref: _EnumRef, path: str) -> StateEnum | None:
        candidates = self.by_name.get(ref.name, [])
        if ref.module_path is not None:
            matches = [
                candidate
                for candidate in candidates
                if _module_matches(candidate.path, ref.module_path)
            ]
        else:
            matches = [candidate for candidate in candidates if candidate.path == path]
        if len(matches) == 1:
            return matches[0]
        return candidates[0] if len(candidates) == 1 else None

    def resolve_member(self, ref: _EnumRef, member: str, path: str) -> StateEnum | None:
        state_enum = self.resolve(ref, path)
        if state_enum is None or member not in {
            item.name for item in state_enum.members
        }:
            return None
        return state_enum


def _scan_sources(sources: Sequence[tuple[str, str]]) -> list[_FileFacts]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(_scan_file, sources, chunksize=PARSE_CHUNK_SIZE))


def find_states(
    repo: Path, commit: str, prefixes: Sequence[str] = DEFAULT_PREFIXES
) -> StatesReport:
    selected = [
        (path, object_id)
        for path, object_id in _git_tree(repo, commit)
        if path.endswith(".py") and path.startswith(tuple(prefixes))
    ]
    contents = _git_blobs(repo, selected)
    file_facts = _scan_sources(sorted(contents.items()))

    enums = [
        StateEnum(
            enum_def.name,
            facts.path,
            enum_def.line,
            [Member(name, line) for name, line in enum_def.members],
        )
        for facts in file_facts
        for enum_def in facts.enums
    ]
    resolver = _Resolver(enums)
    for facts in file_facts:
        for column in facts.columns:
            state_enum = resolver.resolve(column.ref, facts.path)
            if state_enum is not None:
                state_enum.columns.append(
                    StateColumn(column.table, column.column, facts.path, column.line)
                )
        for state_set in facts.sets:
            state_enum = resolver.resolve_member(
                state_set.ref, state_set.member, facts.path
            )
            if state_enum is not None:
                state_enum.sets.append(
                    StateSet(state_set.member, facts.path, state_set.line)
                )
        for fact in facts.transitions:
            source = resolver.resolve_member(
                fact.from_ref, fact.from_member, facts.path
            )
            target = resolver.resolve_member(fact.to_ref, fact.to_member, facts.path)
            if source is not None and source is target:
                source.transitions.append(
                    Transition(fact.from_member, fact.to_member, facts.path, fact.line)
                )
    for state_enum in enums:
        state_enum.columns.sort(key=lambda item: (item.path, item.line))
        state_enum.sets.sort(key=lambda item: (item.path, item.line))
        state_enum.transitions = sorted(
            set(state_enum.transitions),
            key=lambda item: (item.path, item.line, item.from_member, item.to_member),
        )
    enums.sort(key=lambda item: (item.path, item.line))
    parse_errors = [
        ParseError(facts.path, facts.parse_error)
        for facts in file_facts
        if facts.parse_error is not None
    ]
    return StatesReport(commit, enums, parse_errors)
