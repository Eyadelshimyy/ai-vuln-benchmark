"""Solidity semantic IR.

Built from the Tree-sitter syntax graph, symbols, and CFG. Compiler output is
attached by profile identity when a compiler project is present. This model is
analysis data. It does not verify findings. Partial coverage stays partial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.model import SyntaxEvent, SyntaxGraph
from app.parsing.span import SourceSpan

SCHEMA_VERSION = "phase39.1"
FUNCTION_LIMIT = 200
_LOW = frozenset({"call", "delegatecall", "staticcall", "transfer", "send"})
_TOKEN = frozenset({"transfer", "transferFrom", "safeTransfer", "safeTransferFrom", "mint", "burn"})
_ORACLE = frozenset({"latestRoundData", "latestAnswer", "consult", "observe", "getReserves"})
_SKIP_CALLS = frozenset(
    {
        "require",
        "assert",
        "revert",
        "if",
        "while",
        "for",
        "keccak256",
        "ecrecover",
        "abi",
        "unchecked",
    }
)


@dataclass(frozen=True)
class SourceSpanRef:
    file: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class StateDeclaration:
    declaration_id: str
    contract: str
    symbol: str
    type_name: str
    location: str
    span: SourceSpanRef


@dataclass(frozen=True)
class StateAccess:
    declaration_id: str
    contract: str
    symbol: str
    path: str
    index_text: str
    member_path: str
    kind: str
    domain: str
    location: str
    span: SourceSpanRef
    operation_id: str = ""
    guard_status: str = "unknown"
    guard_predicate: str = ""
    guard_dominates: str = "unknown"


@dataclass(frozen=True)
class ContractFact:
    name: str
    kind: str
    bases: tuple[str, ...]
    span: tuple[int, ...] = ()


@dataclass(frozen=True)
class SemanticCall:
    caller: str
    callee: str
    resolution: str
    target: str
    arguments: str
    value: str
    call_type: str
    external: bool
    callback_potential: bool
    return_used: bool
    success_handled: bool
    span: SourceSpanRef
    call_id: str = ""


@dataclass(frozen=True)
class GuardFact:
    status: str
    predicates: tuple[str, ...]
    modifiers: tuple[str, ...]
    helpers: tuple[str, ...]
    dominates: bool
    detail: str


@dataclass(frozen=True)
class SemanticFunction:
    contract: str
    name: str
    line: int
    visibility: str
    mutability: str
    modifiers: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    calls: tuple[str, ...]
    authorization: str
    origin: str = "parser"
    path_status: str = "unknown"
    read_before_write: tuple[str, ...] = ()
    write_before_read: tuple[str, ...] = ()
    read_write_same_operation: tuple[str, ...] = ()
    accesses: tuple[str, ...] = ()
    declaration_ids: tuple[str, ...] = ()
    span: tuple[int, int, int, int] = (0, 0, 0, 0)
    call_sites: tuple[SemanticCall, ...] = ()
    access_sites: tuple[StateAccess, ...] = ()
    guard: GuardFact | None = None
    constraints: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    source: str = ""

    @property
    def identity(self) -> str:
        return f"{self.contract}.{self.name}:{self.line}"


@dataclass
class SemanticProgram:
    schema_version: str
    file: str
    status: str
    functions: tuple[SemanticFunction, ...]
    state_variables: tuple[str, ...]
    compiler_ir: str
    compiler_ir_status: str
    notes: tuple[str, ...] = ()
    function_limit: int = FUNCTION_LIMIT
    functions_seen: int = 0
    functions_indexed: int = 0
    incomplete_reason: str = ""
    declarations: tuple[StateDeclaration, ...] = ()
    compiler_profiles: tuple[dict[str, str], ...] = ()
    compiler_version: str = ""
    contracts: tuple[ContractFact, ...] = ()

    def functions_named(self, name: str, contract: str = "") -> tuple[SemanticFunction, ...]:
        return tuple(
            item
            for item in self.functions
            if item.name == name and (not contract or item.contract == contract)
        )

    def state_writes(self, function: SemanticFunction) -> tuple[str, ...]:
        return function.writes

    def external_calls(self, function: SemanticFunction) -> tuple[str, ...]:
        return function.calls

    def authorization(self, function: SemanticFunction) -> str:
        return function.authorization

    def declaration(self, declaration_id: str) -> StateDeclaration | None:
        for item in self.declarations:
            if item.declaration_id == declaration_id:
                return item
        return None

    def function_status(self, contract: str, name: str, line: int = 0) -> str:
        """Distinguish absent, not indexed, and present."""
        if self.incomplete_reason and self.functions_seen > self.functions_indexed:
            matched = [
                item
                for item in self.functions
                if item.contract == contract
                and item.name == name
                and (not line or item.line == line)
            ]
            if matched:
                return "indexed"
            return "not_indexed"
        named = self.functions_named(name, contract)
        if line:
            named = tuple(item for item in named if item.line == line)
        if named:
            return "indexed"
        if not self.functions and self.status == "unavailable":
            return "absent"
        return "not_found"


def build_semantic_program(
    graph: SyntaxGraph, *, compiler_ir: str = "", compiler_ir_status: str = "unavailable"
) -> SemanticProgram:
    if graph.language != "solidity":
        return SemanticProgram(
            SCHEMA_VERSION, graph.file_path, "unavailable", (), (), "", "unavailable"
        )
    declarations = _declarations(graph)
    by_contract = _decls_by_contract(declarations)
    functions, seen, reason = _functions(graph, by_contract)
    status = "partial" if functions or reason else "unavailable"
    if reason:
        status = "partial"
    profiles, ir_status, version, notes = _compiler_binding(graph, compiler_ir, compiler_ir_status)
    if ir_status != "available":
        notes = (*notes, "compiler IR is not available; function facts are parser-originated")
    return SemanticProgram(
        SCHEMA_VERSION,
        graph.file_path,
        status,
        tuple(functions),
        tuple(sorted({item.symbol for item in declarations})),
        "",
        ir_status,
        notes,
        functions_seen=seen,
        functions_indexed=len(functions),
        incomplete_reason=reason,
        declarations=tuple(declarations),
        compiler_profiles=profiles,
        compiler_version=version,
        contracts=_contract_facts(graph),
    )


def render_snapshot(program: SemanticProgram) -> str:
    lines = [f"file {program.file}", f"status {program.status}", f"schema {program.schema_version}"]
    if program.incomplete_reason:
        lines.append(f"incomplete {program.incomplete_reason}")
    by_contract: dict[str, list[SemanticFunction]] = {}
    for item in program.functions:
        by_contract.setdefault(item.contract or "?", []).append(item)
    for contract in sorted(by_contract):
        lines.append(f"Contract {contract}")
        for item in by_contract[contract]:
            lines.append(f"  Function {item.name} line {item.line}")
            lines.append(f"    authorization: {item.authorization}")
            lines.append(f"    reads: {', '.join(item.reads) or '-'}")
            lines.append(f"    writes: {', '.join(item.writes) or '-'}")
            lines.append(f"    calls: {', '.join(item.calls) or '-'}")
            lines.append(f"    origin: {item.origin}")
    lines.append(f"compiler_ir {program.compiler_ir_status}")
    return "\n".join(lines)


def _compiler_binding(
    graph: SyntaxGraph, compiler_ir: str, compiler_ir_status: str
) -> tuple[tuple[dict[str, str], ...], str, str, tuple[str, ...]]:
    from app.parsing.solidity_project import current_compiler_project

    project = current_compiler_project()
    if project is None:
        status = (
            compiler_ir_status
            if compiler_ir_status in {"unavailable", "available", "partial", "truncated", "failed"}
            else "unavailable"
        )
        if compiler_ir and status == "available":
            return (
                (),
                "partial",
                "",
                ("compiler IR string was supplied without a source identity",),
            )
        return (), "unavailable" if status == "available" and not compiler_ir else status, "", ()
    version = project.compiler_version or ""
    if len({item.get("version", "") for item in project.compiler_profiles}) > 1:
        version = "mixed"
    profiles: list[dict[str, str]] = []
    for item in project.compiler_profiles:
        sources = item.get("sources", "")
        listed = [part.strip() for part in sources.split(",") if part.strip()]
        if listed and not any(_same_source(graph.file_path, part) for part in listed):
            continue
        profiles.append(
            {
                "version": item.get("version", ""),
                "tool": item.get("tool", project.tool),
                "status": item.get("status", project.status),
                "ir": item.get("ir", "unavailable"),
                "ast": item.get("ast", "available" if project.ast_available else "unavailable"),
                "storage": "available" if project.layouts else "unavailable",
                "sources": sources,
                "configuration": project.configuration_identity,
            }
        )
    if not profiles and project.compiler_profiles:
        status = "partial"
    elif project.ir_truncated:
        status = "truncated"
    elif project.status == "AVAILABLE" and project.ir_available:
        status = "available"
    elif project.status in {"INCOMPLETE", "FAILED"}:
        status = "partial" if project.status == "INCOMPLETE" else "failed"
    else:
        status = "unavailable"
    notes: tuple[str, ...] = ()
    if status != "available":
        notes = (project.detail or "compiler profile is not a complete IR",)
    return tuple(profiles), status, version, notes


def _declarations(graph: SyntaxGraph) -> list[StateDeclaration]:
    symbols = {
        (symbol.name, symbol.span.start_line if symbol.span else 0): symbol
        for symbol in graph.symbols
    }
    found: list[StateDeclaration] = []
    for event in graph.events:
        if event.kind != "sol_state":
            continue
        fields = _extra(event)
        name = fields.get("name", "")
        if not name:
            continue
        symbol = symbols.get((name, event.line))
        contract = fields.get("contract", "")
        declaration_id = (
            symbol.symbol_id
            if symbol is not None
            else f"{graph.file_path}:{contract}:{name}:{event.line}"
        )
        type_name = fields.get("type", "")
        mutability = fields.get("mutability", "storage")
        location = _location(type_name, mutability)
        found.append(
            StateDeclaration(
                declaration_id=declaration_id,
                contract=contract,
                symbol=name,
                type_name=type_name,
                location=location,
                span=_span(graph, event),
            )
        )
    return found


def _decls_by_contract(
    declarations: list[StateDeclaration],
) -> dict[str, dict[str, StateDeclaration]]:
    grouped: dict[str, dict[str, StateDeclaration]] = {}
    for item in declarations:
        grouped.setdefault(item.contract, {})[item.symbol] = item
    return grouped


def _functions(
    graph: SyntaxGraph, by_contract: dict[str, dict[str, StateDeclaration]]
) -> tuple[list[SemanticFunction], int, str]:
    events = [event for event in graph.events if event.kind == "sol_function"]
    reason = "function limit reached" if len(events) > FUNCTION_LIMIT else ""
    names = {_function_name(event) for event in events}
    found: list[SemanticFunction] = []
    for event in events[:FUNCTION_LIMIT]:
        fields = _extra(event)
        name = _function_name(event)
        if not name:
            continue
        contract = fields.get("contract", "")
        state = by_contract.get(contract, {})
        shadowed = _param_names(event.text)
        visible = {symbol: decl for symbol, decl in state.items() if symbol not in shadowed}
        inside = _inside(graph, event)
        reads, writes, same, before, after = _effects(event.text, visible, inside)
        origin = event.span.start_byte if event.span else 0
        accesses = _stamp_operation_guards(event.text, _accesses(graph, event, visible), origin)
        call_sites = _calls(graph, event, name, names, inside, contract)
        low = tuple(
            sorted({site.callee for site in call_sites if site.callee in _LOW and site.external})
        )
        modifiers = tuple(
            part.strip() for part in fields.get("modifiers", "").split(",") if part.strip()
        )
        guard = _guard(graph, event, contract, modifiers, writes, low)
        span = event.span
        found.append(
            SemanticFunction(
                contract=contract,
                name=name,
                line=event.line,
                visibility=fields.get("visibility", ""),
                mutability=fields.get("mutability", ""),
                modifiers=modifiers,
                reads=reads,
                writes=writes,
                calls=low,
                authorization=guard.status,
                path_status=_path_status(event.text),
                read_before_write=before,
                write_before_read=after,
                read_write_same_operation=same,
                accesses=tuple(sorted({item.path for item in accesses})),
                declaration_ids=tuple(
                    sorted(
                        {
                            visible[item].declaration_id
                            for item in set(reads) | set(writes)
                            if item in visible
                        }
                    )
                ),
                span=(
                    span.start_byte if span else 0,
                    span.end_byte if span else 0,
                    span.start_line if span else event.line,
                    span.end_line if span else event.line,
                ),
                call_sites=tuple(call_sites),
                access_sites=tuple(accesses),
                guard=guard,
                constraints=_constraints(event.text),
                parameters=tuple(sorted(_param_names(event.text))),
                source=event.text,
            )
        )
    return found, len(events), reason


def _effects(
    text: str,
    visible: dict[str, StateDeclaration],
    inside: list[SyntaxEvent],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    del inside
    from app.parsing.solidity_cfg import extract_body, nested_statements
    from app.parsing.solidity_expr import occurrences

    reads: set[str] = set()
    writes: set[str] = set()
    same: set[str] = set()
    body = extract_body(text) or text
    for _offset, statement in nested_statements(body):
        items = occurrences(statement, set(visible))
        statement_reads = {item.base for item in items if item.kind != "write"}
        statement_writes = {item.base for item in items if item.kind != "read"}
        reads |= {item.base for item in items if item.kind != "write"}
        writes |= statement_writes
        same |= statement_reads & statement_writes
    return (
        tuple(sorted(reads)),
        tuple(sorted(writes)),
        tuple(sorted(same)),
        tuple(sorted(same)),
        (),
    )


def _accesses(
    graph: SyntaxGraph, function: SyntaxEvent, visible: dict[str, StateDeclaration]
) -> list[StateAccess]:
    from app.parsing.solidity_cfg import extract_body, nested_statements
    from app.parsing.solidity_expr import occurrences

    text = function.text
    found: list[StateAccess] = []
    body = extract_body(text) or text
    body_at = text.find(body) if body else 0
    origin = function.span.start_byte if function.span else 0
    for at, statement in nested_statements(body):
        for item in occurrences(statement, set(visible)):
            decl = visible.get(item.base)
            if decl is None:
                continue
            domain = "state"
            if item.index_text and "mapping" in decl.type_name:
                domain = "mapping"
            elif item.index_text:
                domain = "array"
            if item.member_path:
                domain = "struct-member"
            start = origin + body_at + at + item.start
            end = origin + body_at + at + item.end
            line = text.count("\n", 0, body_at + at + item.start) + (
                function.span.start_line if function.span else 1
            )
            found.append(
                StateAccess(
                    declaration_id=decl.declaration_id,
                    contract=decl.contract,
                    symbol=item.base,
                    path=item.path,
                    index_text=item.index_text,
                    member_path=item.member_path,
                    kind=item.kind,
                    domain=domain,
                    location=decl.location,
                    span=SourceSpanRef(graph.file_path, start, end, line, line),
                    operation_id=f"{decl.declaration_id}:{item.kind}:{start}",
                )
            )
    return found


def _stamp_operation_guards(
    text: str, accesses: list[StateAccess], origin: int
) -> list[StateAccess]:
    from app.parsing.solidity_cfg import operation_guard_at

    stamped: list[StateAccess] = []
    for item in accesses:
        if item.kind == "read":
            stamped.append(item)
            continue
        status, predicate = operation_guard_at(text, item.span.start_byte - origin)
        dominates = "yes" if status == "guarded" else "no" if status == "unguarded" else "unknown"
        stamped.append(
            StateAccess(
                declaration_id=item.declaration_id,
                contract=item.contract,
                symbol=item.symbol,
                path=item.path,
                index_text=item.index_text,
                member_path=item.member_path,
                kind=item.kind,
                domain=item.domain,
                location=item.location,
                span=item.span,
                operation_id=item.operation_id,
                guard_status=status,
                guard_predicate=predicate,
                guard_dominates=dominates,
            )
        )
    return stamped


def _declaration_span(span: SourceSpan | None) -> tuple[int, ...]:
    if span is None:
        return ()
    return (
        span.start_byte,
        span.end_byte,
        span.start_line,
        span.start_column,
        span.end_line,
        span.end_column,
    )


def _contract_facts(graph: SyntaxGraph) -> tuple[ContractFact, ...]:
    found: list[ContractFact] = []
    for event in graph.events:
        if event.kind != "sol_contract":
            continue
        fields = _extra(event)
        bases: list[str] = []
        for part in fields.get("bases", "").split(","):
            match = re.match(r"\s*([A-Za-z_]\w*)", part)
            if match:
                bases.append(match.group(1))
        found.append(
            ContractFact(
                event.text.strip(),
                fields.get("kind", "contract") or "contract",
                tuple(bases),
                _declaration_span(event.span),
            )
        )
    return tuple(found)


def _same_source(file_path: str, source: str) -> bool:
    file_norm = file_path.replace("\\", "/").rstrip("/")
    source_norm = source.replace("\\", "/").lstrip("./")
    return file_norm == source_norm or file_norm.endswith("/" + source_norm)


def _calls(
    graph: SyntaxGraph,
    function: SyntaxEvent,
    function_name: str,
    names: set[str],
    inside: list[SyntaxEvent],
    contract: str,
) -> list[SemanticCall]:
    span = function.span
    found: list[SemanticCall] = []
    seen: set[tuple[int, str]] = set()
    for site in graph.calls:
        if span is None or site.span is None:
            continue
        if not (span.start_byte <= site.span.start_byte < span.end_byte):
            continue
        if site.name in _SKIP_CALLS:
            continue
        key = (site.span.start_byte, site.name)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            _classify_call(
                graph,
                function_name,
                site.name,
                site.qualified,
                site.span,
                inside,
                function.text,
                contract,
            )
        )
    return found


def _classify_call(
    graph: SyntaxGraph,
    caller: str,
    callee: str,
    qualified: str,
    span: SourceSpan | None,
    inside: list[SyntaxEvent],
    function_text: str,
    contract: str = "",
) -> SemanticCall:
    head = re.split(r"[\(\{]", qualified, maxsplit=1)[0].strip()
    target = head[: -len(callee)].rstrip(".") if callee and head.endswith(callee) else head
    value = ""
    value_match = re.search(r"\{\s*value\s*:\s*([^}]+)\}", qualified)
    if value_match:
        value = value_match.group(1).strip()
    checked = any(
        event.kind == "sol_external_call"
        and event.span is not None
        and span is not None
        and event.span.start_byte == span.start_byte
        and _extra(event).get("checked") == "true"
        for event in inside
    )
    call_type = "unresolved-external"
    external = True
    resolution = "unresolved"
    callback = False
    for member, kind in (
        (".delegatecall", "delegatecall"),
        (".staticcall", "staticcall"),
        (".call", "low-level-call"),
        (".send", "send"),
    ):
        if member in qualified and f"{member}{{" in qualified or member + "(" in qualified:
            call_type = kind
            target = qualified.split(member, 1)[0].strip()
            callee = member[1:]
            resolution = "low-level"
            break
    if call_type == "delegatecall":
        pass
    elif callee in {"delegatecall"}:
        call_type = "delegatecall"
    elif callee == "staticcall":
        call_type = "staticcall"
    elif callee == "call":
        call_type = "low-level-call"
    elif callee == "send":
        call_type = "send"
    elif callee == "transfer" and "." in head and "," not in qualified:
        call_type = "transfer"
    elif callee in _TOKEN:
        call_type = "token"
        callback = callee.lower() in {
            "safetransfer",
            "safetransferfrom",
            "transfer",
            "transferfrom",
        }
        resolution = "interface"
    elif callee in _ORACLE:
        call_type = "oracle"
        resolution = "interface"
    elif target in {"this"}:
        call_type = "this"
        resolution = "external-this"
    elif callee in {"delegatecall", "call", "staticcall", "send"}:
        resolution = "low-level"
    elif target == "super":
        external = False
        call_type = "inherited"
        resolution = "unknown" if _overload_count(graph, callee) != 1 else "resolved"
    elif "." not in head and callee:
        external = False
        call_type = "internal"
        count = _overload_count(graph, callee, contract)
        resolution = "resolved" if count == 1 else "unknown" if count > 1 else "unresolved"
    elif "." in head:
        call_type = "contract-typed"
        resolution = "unresolved"
    source_span = span
    return SemanticCall(
        caller=caller,
        callee=callee,
        resolution=resolution,
        target=target,
        arguments=qualified,
        value=value,
        call_type=call_type,
        external=external or callee in _LOW,
        callback_potential=callback or callee in {"call", "delegatecall", "transfer", "send"},
        return_used=_return_used(
            function_text, qualified, source_span.start_byte if source_span else 0
        ),
        success_handled=checked,
        span=SourceSpanRef(
            graph.file_path,
            source_span.start_byte if source_span else 0,
            source_span.end_byte if source_span else 0,
            source_span.start_line if source_span else 0,
            source_span.end_line if source_span else 0,
        ),
        call_id=f"{caller}:{callee}:{source_span.start_byte if source_span else 0}",
    )


def _overload_count(graph: SyntaxGraph, name: str, contract: str = "") -> int:
    """Count functions named ``name``. An empty contract counts every contract."""
    return sum(
        1
        for event in graph.events
        if event.kind == "sol_function"
        and _function_name(event) == name
        and (not contract or _extra(event).get("contract", "") == contract)
    )


def _return_used(function_text: str, qualified: str, start_byte: int) -> bool:
    local = function_text.find(qualified)
    if local < 0:
        local = 0
    prefix = function_text[max(0, local - 48) : local]
    del start_byte
    return bool(re.search(r"(=|\()\s*$", prefix))


def _function_names_text(graph: SyntaxGraph, name: str) -> bool:
    return any(
        _function_name(event) == name for event in graph.events if event.kind == "sol_function"
    )


def _guard(
    graph: SyntaxGraph,
    event: SyntaxEvent,
    contract: str,
    modifiers: tuple[str, ...],
    writes: tuple[str, ...],
    calls: tuple[str, ...],
) -> GuardFact:
    from app.parsing.solidity_cfg import operation_guarded, placeholder_is_guarded
    from app.parsing.solidity_modifiers import resolve_modifier

    predicates: list[str] = []
    helpers: list[str] = []
    dominates = False
    ambiguous = False
    unresolved_guard = False
    text = event.text
    injections: list[str] = []
    for raw in modifiers:
        token = re.split(r"[\(\s]", raw.strip(), maxsplit=1)[0]
        if not token:
            continue
        resolution = resolve_modifier(graph, contract, token)
        if resolution.status == "ambiguous":
            ambiguous = True
            continue
        if resolution.status == "resolved" and placeholder_is_guarded(resolution.body):
            injections.append("require(msg.sender == owner);")
            predicates.append(token)
        elif resolution.status == "unresolved":
            unresolved_guard = True
    for name, body in _helper_guards(graph).items():
        if name == _function_name(event):
            continue
        if re.search(rf"\b{re.escape(name)}\s*\(", text):
            helpers.append(name)
            predicates.append(name)
            injections.append("require(msg.sender == owner);")
            del body
    if injections:
        brace = text.find("{")
        if brace >= 0:
            text = text[: brace + 1] + " ".join(injections) + text[brace + 1 :]
    operations = [item for item in (*writes, *calls) if item]
    verdicts: list[bool | None] = [operation_guarded(text, operation) for operation in operations]
    if not operations:
        status = "unknown"
    elif ambiguous or unresolved_guard or any(item is None for item in verdicts):
        status = "unknown"
    elif all(item is True for item in verdicts):
        dominates = True
        status = "guarded"
    elif any(item is False for item in verdicts):
        status = "unguarded"
    else:
        status = "unknown"
    for match in re.finditer(r"require\s*\(([^)]*)\)", event.text):
        predicates.append(match.group(0)[:160])
    return GuardFact(
        status, tuple(dict.fromkeys(predicates)), modifiers, tuple(helpers), dominates, ""
    )


def _helper_guards(graph: SyntaxGraph) -> dict[str, str]:
    from app.parsing.solidity_cfg import operation_guarded

    grouped: dict[str, list[str]] = {}
    for event in graph.events:
        if event.kind != "sol_function":
            continue
        name = _function_name(event)
        if name:
            grouped.setdefault(name, []).append(event.text)
    helpers: dict[str, str] = {}
    for name, bodies in grouped.items():
        if len(bodies) != 1:
            continue
        start = bodies[0].find("{")
        end = bodies[0].rfind("}")
        if start < 0 or end <= start:
            continue
        inner = bodies[0][start + 1 : end]
        probe = "function _h() {" + inner + " owner = next; }"
        if operation_guarded(probe, "owner = next") is True:
            helpers[name] = inner
    return helpers


def _path_status(text: str) -> str:
    from app.parsing.solidity_cfg import build_function_cfg

    cfg = build_function_cfg(text)
    if not cfg.known:
        return "unknown"
    if cfg.then_of or cfg.else_of:
        return "branch-dependent" if cfg.else_of else "conditional"
    return "unconditional"


def _constraints(text: str) -> tuple[str, ...]:
    return tuple(match.group(0)[:180] for match in re.finditer(r"require\s*\([^;]*\)", text))


def _param_names(text: str) -> set[str]:
    header = text.split("{", 1)[0]
    params = header[header.find("(") + 1 : header.rfind(")")] if "(" in header else ""
    names = set(re.findall(r"\b([A-Za-z_]\w*)\s*(?:,|$)", params))
    return {name for name in names if name not in {"memory", "storage", "calldata"}}


def _shadowed_names(text: str) -> set[str]:
    header = text.split("{", 1)[0]
    params = header[header.find("(") + 1 : header.rfind(")")] if "(" in header else ""
    names = set(re.findall(r"\b([A-Za-z_]\w*)\s*(?:,|$)", params))
    body = text.split("{", 1)[-1]
    names.update(
        re.findall(
            r"\b(?:uint\d*|int\d*|address|bool|string|bytes\d*|mapping\s*\([^;]+?\))\s+([A-Za-z_]\w*)",
            body,
        )
    )
    return {
        name
        for name in names
        if name not in {"memory", "storage", "calldata", "external", "public"}
    }


def _location(type_name: str, mutability: str) -> str:
    if "transient" in type_name or mutability == "transient":
        return "transient"
    if mutability in {"immutable", "constant"}:
        return mutability
    if "memory" in type_name:
        return "memory"
    return "storage"


def _inside(graph: SyntaxGraph, function: SyntaxEvent) -> list[SyntaxEvent]:
    span = function.span
    if span is None:
        return []
    return [
        event
        for event in graph.events
        if event.span is not None
        and span.start_byte <= event.span.start_byte < span.end_byte
        and event is not function
    ]


def _span(graph: SyntaxGraph, event: SyntaxEvent) -> SourceSpanRef:
    span = event.span
    return SourceSpanRef(
        graph.file_path,
        span.start_byte if span else 0,
        span.end_byte if span else 0,
        span.start_line if span else event.line,
        span.end_line if span else event.line,
    )


def _function_name(event: SyntaxEvent) -> str:
    match = re.search(r"function\s+([A-Za-z_]\w*)", event.text)
    if match:
        return match.group(1)
    stripped = event.text.lstrip()
    if stripped.startswith("constructor"):
        return "constructor"
    if stripped.startswith("receive"):
        return "receive"
    if stripped.startswith("fallback"):
        return "fallback"
    return ""


def _extra(event: SyntaxEvent) -> dict[str, str]:
    fields: dict[str, str] = {}
    raw = event.extra if isinstance(event.extra, str) else ""
    for part in raw.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields
