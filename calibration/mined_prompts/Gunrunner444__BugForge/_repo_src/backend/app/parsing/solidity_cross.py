"""Bounded cross-function and cross-contract Solidity relationships.

This is a project-level summary on top of the existing syntax graph. It is not
a compiler, not symbolic execution, and not a proof that a callback is
exploitable. An unresolved callee stays unresolved. A limit leaves the model
incomplete rather than safe.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from app.parsing.model import SyntaxEvent, SyntaxGraph
from app.parsing.solidity_defi import analyze_defi
from app.parsing.solidity_guards import reentrancy_guard_holds
from app.parsing.solidity_modifiers import resolve_modifier
from app.parsing.solidity_proxy import analyze_proxy

_MAX_CONTRACTS = 48
_MAX_EDGES = 240


def _contract_limit() -> int:
    from app.core.config import get_settings

    return int(
        getattr(get_settings(), "solidity_cross_max_contracts", _MAX_CONTRACTS) or _MAX_CONTRACTS
    )


def _edge_limit() -> int:
    from app.core.config import get_settings

    return int(getattr(get_settings(), "solidity_cross_max_edges", _MAX_EDGES) or _MAX_EDGES)


_CALLBACKS = {
    "tokensreceived",
    "ontokenreceived",
    "onerc721received",
    "onerc1155received",
    "onerc1155batchreceived",
}
_TOKEN_METHODS = {"transfer", "transferfrom", "safetransfer", "safetransferfrom"}
_AUTH_METHODS = {"hasrole", "isallowed", "authorize", "checkrole"}
_SKIP_CALLS = {
    "if",
    "require",
    "assert",
    "revert",
    "return",
    "while",
    "for",
    "unchecked",
    "keccak256",
    "ecrecover",
    "abi",
    "returns",
    "address",
    "uint256",
    "uint",
    "int256",
    "bytes",
    "string",
    "function",
}

_CTX: ContextVar[ProjectModel | None] = ContextVar("bugforge_project_model", default=None)


@dataclass(frozen=True)
class SymbolIdentity:
    contract: str
    function: str
    file: str
    qualified_id: str


class SymbolTable:
    """Function symbols. Duplicate contract/function pairs stay ambiguous."""

    def __init__(self) -> None:
        self.entries: list[tuple[SymbolIdentity, SyntaxEvent]] = []

    def add(self, ident: SymbolIdentity, event: SyntaxEvent) -> None:
        self.entries.append((ident, event))

    def matches(self, contract: str, function: str) -> list[tuple[SymbolIdentity, SyntaxEvent]]:
        return [
            item
            for item in self.entries
            if item[0].contract == contract and item[0].function == function
        ]

    def status(self, contract: str, function: str) -> str:
        count = len(self.matches(contract, function))
        if count == 1:
            return "resolved"
        if count > 1:
            return "ambiguous"
        return "missing"

    def get(self, key: tuple[str, str]) -> SyntaxEvent | None:
        found = self.matches(key[0], key[1])
        if len(found) == 1:
            return found[0][1]
        return None

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        return self.status(str(key[0]), str(key[1])) == "resolved"

    def items(self) -> list[tuple[tuple[str, str], SyntaxEvent]]:
        return [((ident.contract, ident.function), event) for ident, event in self.entries]


@dataclass(frozen=True)
class ContractRef:
    name: str
    kind: str
    file_path: str
    bases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CallEdge:
    caller_contract: str
    caller_function: str
    callee_contract: str
    callee_function: str
    kind: str
    status: str
    detail: str


@dataclass(frozen=True)
class DataFlow:
    contract: str
    function: str
    source: str
    sink: str
    kind: str
    branch_sensitive: bool
    known: bool


@dataclass(frozen=True)
class CrossFinding:
    kind: str
    contract: str
    function: str
    summary: str
    confidence: str


@dataclass
class ProjectModel:
    contracts: list[ContractRef] = field(default_factory=list)
    calls: list[CallEdge] = field(default_factory=list)
    flows: list[DataFlow] = field(default_factory=list)
    findings: list[CrossFinding] = field(default_factory=list)
    economics: list[str] = field(default_factory=list)
    incomplete: bool = False
    limit_reason: str = ""
    edge_keys: set[tuple[str, str, str, str, str, str]] = field(default_factory=set)


def set_project_context(graphs: dict[str, SyntaxGraph]) -> Token[ProjectModel | None]:
    return _CTX.set(build_project(graphs))


def reset_project_context(token: Token[ProjectModel | None]) -> None:
    _CTX.reset(token)


def project_context_active() -> bool:
    return _CTX.get() is not None


def analyze_project(graphs: dict[str, SyntaxGraph] | None = None) -> ProjectModel:
    """Return the scan-scoped project model, or build one for ``graphs``."""
    if graphs is None:
        current = _CTX.get()
        return current if current is not None else ProjectModel()
    return build_project(graphs)


def build_project(graphs: dict[str, SyntaxGraph]) -> ProjectModel:
    model = ProjectModel()
    solidity = {
        path: graph
        for path, graph in graphs.items()
        if graph.language == "solidity" and graph.parser_tier.value != "profile_fallback"
    }
    contracts = _contracts(solidity)
    if len(contracts) > _contract_limit():
        contracts = contracts[: _contract_limit()]
        model.incomplete = True
        model.limit_reason = "contract limit reached; further relationships are unknown"
    model.contracts = contracts
    by_name: dict[str, list[ContractRef]] = {}
    for item in contracts:
        by_name.setdefault(item.name, []).append(item)
    functions = _functions(solidity)
    for graph in solidity.values():
        _calls_for_graph(model, graph, by_name, functions)
        _flows_for_graph(model, graph, functions)
        _authorization(model, graph, by_name, functions)
        _economics(model, graph)
        if len(model.calls) >= _edge_limit():
            model.incomplete = True
            model.limit_reason = model.limit_reason or "call-edge limit reached"
            break
    _reentrancy(model, functions, solidity)
    return model


def _contracts(graphs: dict[str, SyntaxGraph]) -> list[ContractRef]:
    found: list[ContractRef] = []
    for graph in graphs.values():
        for event in graph.events:
            if event.kind != "sol_contract":
                continue
            fields = _fields(event.extra)
            bases = tuple(item for item in fields.get("bases", "").split(",") if item)
            found.append(
                ContractRef(
                    event.text.strip(), fields.get("kind", "contract"), graph.file_path, bases
                )
            )
    return found


def _add_edge(model: ProjectModel, edge: CallEdge) -> None:
    if len(model.calls) >= _edge_limit():
        model.incomplete = True
        model.limit_reason = model.limit_reason or "call-edge limit reached"
        return
    key = (
        edge.caller_contract,
        edge.caller_function,
        edge.callee_contract,
        edge.callee_function,
        edge.kind,
        edge.status,
    )
    if key in model.edge_keys:
        return
    model.edge_keys.add(key)
    model.calls.append(edge)


def _functions(graphs: dict[str, SyntaxGraph]) -> SymbolTable:
    found = SymbolTable()
    for graph in graphs.values():
        for event in graph.events:
            if event.kind != "sol_function":
                continue
            fields = _fields(event.extra)
            contract = fields.get("contract", "")
            name = fields.get("function", "")
            ident = SymbolIdentity(
                contract,
                name,
                graph.file_path,
                f"{graph.file_path}:{contract}.{name}",
            )
            found.add(ident, event)
    return found


def _calls_for_graph(
    model: ProjectModel,
    graph: SyntaxGraph,
    by_name: dict[str, list[ContractRef]],
    functions: SymbolTable,
) -> None:
    types = _variable_types(graph)

    def add(edge: CallEdge) -> None:
        _add_edge(model, edge)

    for event in graph.events:
        if event.kind != "sol_function":
            continue
        fields = _fields(event.extra)
        caller_contract = fields.get("contract", "")
        caller_function = fields.get("function", "")
        for target, member, _args, _start in _call_sites(event.text):
            if member in {"delegatecall"}:
                add(
                    CallEdge(
                        caller_contract,
                        caller_function,
                        "",
                        "",
                        "delegate",
                        "unresolved",
                        target or "delegatecall target is not a resolved contract",
                    )
                )
                continue
            if not target:
                _direct_edge(model, caller_contract, caller_function, member, by_name, functions)
                continue
            type_name = types.get((caller_contract, target), "")
            if not type_name and target in by_name:
                type_name = target
            _member_edge(
                model,
                caller_contract,
                caller_function,
                target,
                member,
                type_name,
                by_name,
                member,
                _args,
                types,
            )
    for (contract, name), event in functions.items():
        if contract not in {item.name for item in by_name.get(contract, [])}:
            continue
        if graph.file_path != _file_of(by_name, contract):
            continue
        if name.lower() in _CALLBACKS:
            _add_edge(
                model,
                CallEdge(
                    "",
                    "",
                    contract,
                    name,
                    "callback",
                    "unresolved",
                    "callback entry is visible; the external caller is not resolved from this project",
                ),
            )


def _direct_edge(
    model: ProjectModel,
    caller_contract: str,
    caller_function: str,
    name: str,
    by_name: dict[str, list[ContractRef]],
    functions: SymbolTable,
) -> None:
    if not name or name in _SKIP_CALLS:
        return
    local = functions.status(caller_contract, name)
    if local == "resolved":
        kind = "library" if _kind(by_name, caller_contract) == "library" else "direct"
        _add_edge(
            model,
            CallEdge(
                caller_contract, caller_function, caller_contract, name, kind, "resolved", name
            ),
        )
        return
    if local == "ambiguous":
        _add_edge(
            model,
            CallEdge(
                caller_contract,
                caller_function,
                "",
                name,
                "direct",
                "ambiguous",
                "duplicate symbol; callee was not selected",
            ),
        )
        return
    matches = [contract for contract in by_name if functions.status(contract, name) == "resolved"]
    ambiguous = [
        contract for contract in by_name if functions.status(contract, name) == "ambiguous"
    ]
    if ambiguous or len(matches) > 1:
        _add_edge(
            model,
            CallEdge(
                caller_contract,
                caller_function,
                "",
                name,
                "direct",
                "ambiguous",
                "multiple contracts define this function",
            ),
        )
    elif len(matches) == 1:
        _add_edge(
            model,
            CallEdge(
                caller_contract,
                caller_function,
                matches[0],
                name,
                "library" if _kind(by_name, matches[0]) == "library" else "direct",
                "resolved",
                name,
            ),
        )
    else:
        _add_edge(
            model,
            CallEdge(caller_contract, caller_function, "", name, "direct", "unresolved", name),
        )


def _member_edge(
    model: ProjectModel,
    caller_contract: str,
    caller_function: str,
    target: str,
    member: str,
    type_name: str,
    by_name: dict[str, list[ContractRef]],
    low_level: str,
    args: str = "",
    types: dict[tuple[str, str], str] | None = None,
) -> None:
    if member.lower() in _TOKEN_METHODS:
        owners = by_name.get(type_name, []) if type_name else []
        if len(owners) > 1:
            status, callee = "ambiguous", ""
        elif len(owners) == 1:
            status, callee = "resolved", type_name
        else:
            status, callee = "unresolved", ""
        _add_edge(
            model,
            CallEdge(
                caller_contract,
                caller_function,
                callee,
                member,
                "token",
                status,
                _token_detail(caller_contract, member, args, types or {}),
            ),
        )
        return
    if type_name and type_name in by_name:
        kind = "interface" if _kind(by_name, type_name) == "interface" else "direct"
        status = "ambiguous" if len(by_name[type_name]) > 1 else "resolved"
        if kind == "interface":
            status = "resolved"
            detail = "interface callee; an implementation was not selected"
        else:
            detail = target
        _add_edge(
            model,
            CallEdge(caller_contract, caller_function, type_name, member, kind, status, detail),
        )
        return
    if low_level in {"call", "staticcall", "delegatecall"} or member in {"call", "staticcall"}:
        _add_edge(
            model,
            CallEdge(
                caller_contract,
                caller_function,
                "",
                member or low_level,
                "direct",
                "unresolved",
                target or "unresolved external call",
            ),
        )
        return
    _add_edge(
        model,
        CallEdge(
            caller_contract,
            caller_function,
            "",
            member,
            "direct",
            "unresolved",
            target or member,
        ),
    )


def _flows_for_graph(
    model: ProjectModel,
    graph: SyntaxGraph,
    functions: SymbolTable,
) -> None:
    for event in graph.events:
        if event.kind != "sol_function":
            continue
        fields = _fields(event.extra)
        contract = fields.get("contract", "")
        name = fields.get("function", "")
        reads = _names_inside(graph, event, "sol_state_read")
        writes = _names_inside(graph, event, "sol_state_write")
        params = _params(fields.get("params", ""), event.text)
        origins = _origins(event.text, reads, params)
        for target, callee, args, start in _call_sites(event.text):
            branched = _inside_branch(event.text, start)
            for arg in _arg_names(args):
                origin = origins.get(arg, "")
                if not origin and arg in reads:
                    origin = f"state:{arg}"
                if not origin.startswith("state:"):
                    continue
                sink = f"{target}.{callee}" if target else callee
                model.flows.append(
                    DataFlow(
                        contract,
                        name,
                        origin,
                        sink,
                        "helper"
                        if not target and (contract, callee) in functions
                        else "state_to_call",
                        branched,
                        not branched,
                    )
                )
            if not target and (contract, callee) in functions:
                _helper_flow(model, contract, name, callee, args, functions, branched)
        for local, origin in origins.items():
            if origin.startswith("state:") and re.search(
                rf"\breturn\b[^;]*\b{re.escape(local)}\b", event.text
            ):
                model.flows.append(
                    DataFlow(contract, name, origin, f"return:{name}", "return", False, True)
                )
        for write in writes:
            if re.search(rf"\b{re.escape(write)}\b[^;\n]*=[^;\n]*\(", event.text):
                model.flows.append(
                    DataFlow(
                        contract,
                        name,
                        "call",
                        f"state:{write}",
                        "call_to_state",
                        False,
                        False,
                    )
                )


def _helper_flow(
    model: ProjectModel,
    contract: str,
    caller: str,
    callee: str,
    args: str,
    functions: SymbolTable,
    branched: bool,
) -> None:
    event = functions.get((contract, callee))
    if event is None:
        return
    params = _params(_fields(event.extra).get("params", ""), event.text)
    arg_names = _arg_names(args)
    for index, param in enumerate(params):
        if index >= len(arg_names):
            break
        if param and re.search(rf"\b{re.escape(param)}\b", event.text) and ".call" in event.text:
            model.flows.append(
                DataFlow(
                    contract,
                    caller,
                    f"arg:{arg_names[index]}",
                    f"{callee}.call",
                    "helper",
                    branched,
                    not branched,
                )
            )


def _authorization(
    model: ProjectModel,
    graph: SyntaxGraph,
    by_name: dict[str, list[ContractRef]],
    functions: SymbolTable,
) -> None:
    for event in graph.events:
        if event.kind != "sol_function":
            continue
        fields = _fields(event.extra)
        if fields.get("visibility") not in {"public", "external"}:
            continue
        contract = fields.get("contract", "")
        name = fields.get("function", "")
        modifiers = [
            re.split(r"[\(\s]", item.strip(), maxsplit=1)[0]
            for item in fields.get("modifiers", "").split(",")
            if item.strip()
        ]
        for modifier in modifiers:
            resolution = resolve_modifier(graph, contract, modifier)
            if resolution.status == "ambiguous":
                model.findings.append(
                    CrossFinding(
                        "authorization",
                        contract,
                        name,
                        f"`{modifier}` has more than one body. Authority stays unknown.",
                        "unknown",
                    )
                )
            elif resolution.status == "resolved":
                model.flows.append(
                    DataFlow(
                        contract,
                        name,
                        f"modifier:{modifier}",
                        "authorization",
                        "authorization",
                        False,
                        True,
                    )
                )
        for target, member, _args, _start in _call_sites(event.text):
            if member.lower() not in _AUTH_METHODS:
                continue
            type_name = _variable_types(graph).get((contract, target), "")
            callee = functions.get((type_name, member))
            if callee is not None and ("msg.sender" in callee.text or "hasRole" in callee.text):
                continue
            model.findings.append(
                CrossFinding(
                    "authorization",
                    contract,
                    name,
                    f"`{target}.{member}` is an external authorization call whose check was not resolved.",
                    "unknown",
                )
            )


def _economics(model: ProjectModel, graph: SyntaxGraph) -> None:
    defi = analyze_defi(graph)
    for item in defi.transitions:
        chain = " -> ".join(
            part for part in ("user", *item.inflows, item.contract, *item.outflows) if part
        )
        if chain:
            model.economics.append(f"{item.action}: {chain}")
    for call in defi.interactions:
        if call.direction == "unknown":
            continue
        model.economics.append(
            f"{call.function}: {call.token or 'token'} {call.direction} via {call.method}"
        )
    proxy = analyze_proxy(graph)
    for site in proxy.delegates:
        if site.provenance in {"state", "storage_slot"}:
            _add_edge(
                model,
                CallEdge(
                    site.contract,
                    site.function,
                    "",
                    "",
                    "proxy",
                    "unresolved" if site.provenance == "storage_slot" else "resolved",
                    f"{site.provenance}:{site.target}",
                ),
            )


def _reentrancy(
    model: ProjectModel, functions: SymbolTable, graphs: dict[str, SyntaxGraph]
) -> None:
    helpers = {
        (edge.caller_contract, edge.caller_function, edge.callee_function)
        for edge in model.calls
        if edge.status == "resolved" and edge.caller_contract == edge.callee_contract
    }
    by_file = {graph.file_path: graph for graph in graphs.values()}
    for (contract, name), event in functions.items():
        graph = by_file.get(_symbol_file(functions, contract, name))
        if graph is not None and _guarded(graph, contract, event):
            continue
        flows = [
            item
            for item in model.flows
            if item.contract == contract
            and item.function == name
            and item.known
            and item.kind in {"state_to_call", "helper"}
        ]
        if not flows:
            continue
        writes_after = _write_after_call(event.text)
        helper_writes = False
        for flow in flows:
            callee = flow.sink.split(".", 1)[0]
            if (contract, name, callee) in helpers:
                helper = functions.get((contract, callee))
                if helper is not None and _write_after_call(helper.text):
                    helper_writes = ".call" in helper.text or "transfer" in helper.text
        if writes_after or helper_writes:
            model.findings.append(
                CrossFinding(
                    "reentrancy",
                    contract,
                    name,
                    "A state-derived value reaches an external call and state is written on that path. "
                    "This is potential cross-function evidence, not an exploitability proof.",
                    "structural",
                )
            )
    for edge in model.calls:
        if edge.kind != "token" or edge.status != "resolved":
            continue
        receiver = _detail_field(edge.detail, "receiver")
        if receiver != edge.caller_contract:
            continue
        caller_event = functions.get((edge.caller_contract, edge.caller_function))
        reads = {
            item.source.removeprefix("state:")
            for item in model.flows
            if item.contract == edge.caller_contract
            and item.function == edge.caller_function
            and item.source.startswith("state:")
        }
        for (contract, name), event in functions.items():
            if contract != receiver or name.lower() not in _CALLBACKS:
                continue
            writes = _names_inside_text(event.text)
            shared = reads & writes
            if not shared and caller_event is not None:
                shared = {
                    write
                    for write in writes
                    if re.search(rf"\b{re.escape(write)}\b", caller_event.text)
                }
            if shared:
                model.findings.append(
                    CrossFinding(
                        "reentrancy",
                        edge.caller_contract,
                        edge.caller_function,
                        "A resolved token receiver is this contract, and its callback writes "
                        "state the caller read before the call. This is potential evidence, "
                        "not a confirmed reentrancy.",
                        "structural",
                    )
                )


def _symbol_file(functions: SymbolTable, contract: str, name: str) -> str:
    found = functions.matches(contract, name)
    if len(found) != 1:
        return ""
    return found[0][0].file


def _guarded(graph: SyntaxGraph, contract: str, event: SyntaxEvent) -> bool:
    fields = _fields(event.extra)
    modifiers = [
        re.split(r"[\(\s]", item.strip(), maxsplit=1)[0]
        for item in fields.get("modifiers", "").split(",")
        if item.strip()
    ]
    for modifier in modifiers:
        resolution = resolve_modifier(graph, contract, modifier)
        if resolution.status == "resolved" and reentrancy_guard_holds(resolution.body):
            return True
    return False


def _token_detail(caller: str, member: str, args: str, types: dict[tuple[str, str], str]) -> str:
    parts = [item.strip() for item in args.split(",") if item.strip()]
    if member.lower() in {"transferfrom", "safetransferfrom"} and len(parts) > 1:
        recipient = parts[1]
    else:
        recipient = parts[0] if parts else ""
    receiver = ""
    if recipient in {"address(this)", "this"} or recipient.startswith("address(this"):
        receiver = caller
    else:
        bare = recipient.removeprefix("address(").removesuffix(")") if recipient else ""
        typed = types.get((caller, bare), "")
        if typed == caller:
            receiver = caller
    return f"receiver={receiver};token_arg={recipient}"


def _detail_field(detail: str, name: str) -> str:
    prefix = f"{name}="
    for part in detail.split(";"):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _write_after_call(text: str) -> bool:
    call_at = _first_call_offset(text)
    if call_at < 0:
        return False
    return bool(re.search(r"\b[A-Za-z_]\w*\s*(?:\[[^\]]+\])?\s*(?:[+\-*/]?=)", text[call_at:]))


def _first_call_offset(text: str) -> int:
    match = re.search(r"\.(?:call|transfer|transferFrom|safeTransferFrom)\s*[({]", text)
    if match:
        return match.start()
    helper = re.search(r"\b_[A-Za-z_]\w*\s*\(", text)
    return helper.start() if helper else -1


def _origins(text: str, reads: set[str], params: list[str]) -> dict[str, str]:
    origins = {name: f"param:{name}" for name in params}
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);", text):
        local, rhs = match.group(1), match.group(2)
        state_hit = next((name for name in reads if re.search(rf"\b{re.escape(name)}\b", rhs)), "")
        if state_hit:
            origins[local] = f"state:{state_hit}"
            continue
        for known, origin in list(origins.items()):
            if re.search(rf"\b{re.escape(known)}\b", rhs):
                origins[local] = origin
                break
    return origins


def _call_sites(text: str) -> list[tuple[str, str, str, int]]:
    found: list[tuple[str, str, str, int]] = []
    for match in re.finditer(r"(?:([A-Za-z_]\w*)\s*\.\s*)?([A-Za-z_]\w*)\s*\(([^)]*)\)", text):
        target = match.group(1) or ""
        name = match.group(2)
        if name in _SKIP_CALLS or name == "call":
            if name == "call":
                found.append((target, "call", match.group(3), match.start()))
            continue
        found.append((target, name, match.group(3), match.start()))
    return found


def _inside_branch(text: str, index: int) -> bool:
    depth = 0
    branched = False
    cursor = 0
    while cursor < index:
        if text.startswith("if", cursor) and (cursor == 0 or not text[cursor - 1].isalnum()):
            branched = True
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                branched = False
        cursor += 1
    return branched and depth > 0


def _arg_names(args: str) -> list[str]:
    return [item.strip() for item in args.split(",") if re.fullmatch(r"[A-Za-z_]\w*", item.strip())]


def _params(params: str, text: str) -> list[str]:
    header = text.split("{", 1)[0]
    source = params or header
    names = re.findall(r"\b([A-Za-z_]\w*)\s*(?:,|\)|$)", source)
    return [
        name
        for name in names
        if name not in {"memory", "calldata", "storage", "uint", "uint256", "address"}
    ]


def _names_inside(graph: SyntaxGraph, function: SyntaxEvent, kind: str) -> set[str]:
    span = function.span
    if span is None:
        return set()
    names: set[str] = set()
    for event in graph.events:
        if event.kind != kind or event.span is None:
            continue
        if span.start_byte <= event.span.start_byte < span.end_byte:
            name = _fields(event.extra).get("name", "")
            if name:
                names.add(name)
    return names


def _names_inside_text(text: str) -> set[str]:
    return set(re.findall(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]+\])?\s*(?:[+\-*/]?=)", text))


def _variable_types(graph: SyntaxGraph) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for event in graph.events:
        if event.kind != "sol_state":
            continue
        fields = _fields(event.extra)
        type_name = fields.get("type", "").strip().split()[-1] if fields.get("type") else ""
        ident = re.fullmatch(r"[A-Za-z_]\w*", type_name)
        if ident:
            found[(fields.get("contract", ""), fields.get("name", ""))] = ident.group(0)
    return found


def _kind(by_name: dict[str, list[ContractRef]], name: str) -> str:
    items = by_name.get(name, [])
    if len(items) != 1:
        return ""
    return items[0].kind


def _file_of(by_name: dict[str, list[ContractRef]], name: str) -> str:
    items = by_name.get(name, [])
    return items[0].file_path if len(items) == 1 else ""


def _fields(extra: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in extra.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields
