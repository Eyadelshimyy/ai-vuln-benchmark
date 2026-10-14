"""Statement-level value dependencies for one Solidity function.

A read influences a write only through an expression, assignment, return, or
supported branch join. Source distance is not a dependency. Unplaced
operations are counted and omitted. Unknown provenance stays unknown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.parsing.solidity_cfg import FunctionCfg, node_at, node_ranges
from app.parsing.solidity_dataflow import DependencyEdge, FlowValue
from app.parsing.solidity_expr import _top_assign
from app.parsing.solidity_ir import ContractFact, SemanticFunction, SemanticProgram, StateAccess

_CONTROLLED = frozenset({"attacker", "calldata", "caller", "msg.value"})
_RANK = {
    "attacker": 100,
    "calldata": 90,
    "caller": 88,
    "msg.value": 86,
    "external": 80,
    "oracle": 78,
    "unknown": 70,
    "block": 60,
    "environment": 58,
    "state": 50,
    "authority": 40,
    "derived": 15,
    "trusted_constant": 10,
    "constant": 5,
}
_TYPE_NAMES = frozenset(
    {
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uint128",
        "uint256",
        "int",
        "int256",
        "address",
        "bool",
        "bytes",
        "bytes32",
        "string",
        "memory",
        "storage",
        "calldata",
    }
)
_SKIP_KINDS = frozenset({"entry", "exit", "join", "if", "loop", "unchecked", "try"})


@dataclass(frozen=True)
class CallResolution:
    status: str
    function_id: str = ""
    reason: str = ""


def join_provenance(found: set[str]) -> str:
    """Keep the strongest security-relevant provenance.

    Attacker-controlled inputs are not collapsed into a harmless ``derived``
    label. ``unknown`` mixed with a trusted value stays ``unknown``.
    """
    if not found:
        return "unknown"
    controlled = found & _CONTROLLED
    if len(controlled) >= 2:
        return "attacker"
    if len(controlled) == 1:
        return tuple(controlled)[0]
    if "unknown" in found:
        return "unknown"
    return max(found, key=lambda item: _RANK.get(item, 0))


def resolve_call(
    program: SemanticProgram, function: SemanticFunction, site: object
) -> CallResolution:
    """Resolve a call only when the target function is unique.

    Overloads, unresolved interfaces, and unknown external targets stay
    unknown. This does not guess.
    """
    callee = str(getattr(site, "callee", "") or "")
    call_type = str(getattr(site, "call_type", "") or "")
    target = str(getattr(site, "target", "") or "").strip()
    resolution = str(getattr(site, "resolution", "") or "")
    if call_type in {
        "delegatecall",
        "low-level-call",
        "staticcall",
        "send",
        "transfer",
        "token",
        "oracle",
    }:
        return CallResolution("unknown", reason=f"{call_type} target is not a resolved function")
    if resolution == "unknown":
        return CallResolution("unknown", reason="call resolution is ambiguous")
    same = [
        item
        for item in program.functions
        if item.name == callee
        and item.contract == function.contract
        and item.identity != function.identity
    ]
    if len(same) > 1:
        return CallResolution("unknown", reason="overload is ambiguous")
    internal = call_type in {"internal", "this", "inherited", ""} and target in {
        "",
        "this",
        "super",
    }
    if len(same) == 1 and internal and target != "super" and call_type != "inherited":
        return CallResolution("resolved", same[0].identity)
    if target == "super" or call_type == "inherited" or (internal and not same):
        inherited = _inherited(program, function, callee)
        if inherited.status == "resolved" or target == "super" or call_type == "inherited":
            return inherited
    if target:
        fact = _contract_fact(program, target)
        if fact is not None and fact.kind == "interface":
            return CallResolution("unknown", reason="interface call is not an implementation")
        named = [
            item for item in program.functions if item.contract == target and item.name == callee
        ]
        if len(named) == 1:
            return CallResolution("resolved", named[0].identity, "contract-typed")
        if len(named) > 1:
            return CallResolution("unknown", reason="external callee is ambiguous")
    if len(same) == 1 and target in {"", "this"}:
        return CallResolution("resolved", same[0].identity)
    return CallResolution("unknown", reason="callee was not resolved")


def intraprocedural(
    program: SemanticProgram, function: SemanticFunction, limits: dict[str, int]
) -> tuple[
    list[FlowValue],
    list[DependencyEdge],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    int,
    int,
]:
    values: list[FlowValue] = []
    edges: list[DependencyEdge] = []
    known: set[str] = set()
    source = function.source or ""
    cfg, ranges = node_ranges(source)
    if len(cfg.nodes) > limits["blocks"]:
        return values, edges, (), (), (), (), -1, 0
    base = function.span[0] if function.span else 0
    homes = {
        item.operation_id: node_at(ranges, item.span.start_byte - base)
        for item in function.access_sites
    }
    gaps = sum(1 for node_id in homes.values() if node_id is None)
    by_node: dict[int, list[StateAccess]] = {}
    for item in function.access_sites:
        home = homes.get(item.operation_id)
        if home is None:
            continue
        by_node.setdefault(home, []).append(item)
    param_env = _parameters(function, values, known)
    environments: dict[int, dict[str, str]] = {node.node_id: {} for node in cfg.nodes}
    environments[cfg.entry] = dict(param_env)
    iterations = 0
    for iterations in range(1, limits["iterations"] + 1):
        changed = False
        for node in cfg.nodes:
            incoming = [
                environments.get(src, {}) for src, dst, _label in cfg.edges if dst == node.node_id
            ]
            if node.node_id == cfg.entry:
                incoming = [param_env, *incoming]
            merged = _merge_env(function, node.node_id, incoming, values, known, edges)
            updated = _apply_statement(
                program,
                function,
                node.kind,
                node.text,
                node.node_id,
                merged,
                by_node.get(node.node_id, []),
                values,
                known,
                edges,
            )
            if updated != environments[node.node_id]:
                environments[node.node_id] = updated
                changed = True
        if not changed:
            break
    for access in function.access_sites:
        if len(values) >= limits["values"]:
            break
        _add(
            values,
            known,
            access.operation_id,
            access.kind,
            "state",
            (access.declaration_id,),
            function,
            (
                access.span.start_byte,
                access.span.end_byte,
                access.span.start_line,
                access.span.end_line,
            ),
        )
        _add(values, known, access.declaration_id, "declaration", "state", (), function)
        kind = "state-write" if access.kind != "read" else "state-read"
        edges.append(DependencyEdge(access.declaration_id, access.operation_id, kind))
    _influence(function, edges)
    if function.authorization == "guarded":
        sender = f"{function.identity}:caller"
        guard = f"{function.identity}:authority"
        _add(values, known, sender, "caller", "caller", (), function)
        _add(values, known, guard, "authority", "authority", (sender,), function)
        edges.append(DependencyEdge(sender, guard, "authority"))
    may_read, may_write, read_ops, write_ops = _ordering(function, cfg, ranges, base, homes)
    edges = _unique(edge for edge in edges if edge.source in known and edge.sink in known)
    return values, edges, may_read, may_write, read_ops, write_ops, iterations, gaps


def _parameters(
    function: SemanticFunction, values: list[FlowValue], known: set[str]
) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in function.parameters:
        identity = f"{function.identity}:param:{name}"
        _add(values, known, identity, "parameter", "calldata", (), function)
        env[name] = identity
    if "msg.sender" in (function.source or ""):
        identity = f"{function.identity}:caller"
        _add(values, known, identity, "caller", "caller", (), function)
    if "msg.value" in (function.source or ""):
        identity = f"{function.identity}:msg.value"
        _add(values, known, identity, "msg.value", "msg.value", (), function)
    return env


def _merge_env(
    function: SemanticFunction,
    node_id: int,
    envs: list[dict[str, str]],
    values: list[FlowValue],
    known: set[str],
    edges: list[DependencyEdge],
) -> dict[str, str]:
    if not envs:
        return {}
    merged = dict(envs[0])
    for env in envs[1:]:
        for name, identity in env.items():
            current = merged.get(name)
            if current is None:
                merged[name] = identity
                continue
            if current == identity:
                continue
            join_id = f"{function.identity}:join:{name}:{node_id}"
            deps = tuple(dict.fromkeys((current, identity)))
            _add(
                values,
                known,
                join_id,
                "join",
                join_provenance(_provenances(deps, values)),
                deps,
                function,
            )
            for dep in deps:
                edges.append(DependencyEdge(dep, join_id, "branch-join"))
            merged[name] = join_id
    return merged


def _apply_statement(
    program: SemanticProgram,
    function: SemanticFunction,
    kind: str,
    text: str,
    node_id: int,
    env: dict[str, str],
    accesses: list[StateAccess],
    values: list[FlowValue],
    known: set[str],
    edges: list[DependencyEdge],
) -> dict[str, str]:
    outgoing = dict(env)
    if kind in _SKIP_KINDS or kind in {"break", "continue", "revert"}:
        return outgoing
    if kind in {"require", "if"}:
        _expr_value(
            program,
            function,
            text,
            f"condition:{node_id}",
            "=",
            "",
            env,
            accesses,
            values,
            known,
            edges,
        )
        return outgoing
    shape, lhs, op, rhs = _split_statement(text)
    if shape == "return":
        _expr_value(
            program,
            function,
            rhs,
            f"return-value:{node_id}",
            "=",
            "",
            env,
            accesses,
            values,
            known,
            edges,
        )
        return outgoing
    if shape == "tuple":
        identity = _expr_value(
            program,
            function,
            rhs,
            f"tuple:{node_id}",
            "=",
            "",
            env,
            accesses,
            values,
            known,
            edges,
        )
        for name in re.findall(r"[A-Za-z_]\w*", lhs):
            if name not in _TYPE_NAMES:
                outgoing[name] = identity
        return outgoing
    if shape not in {"decl", "assign"}:
        return outgoing
    identity = _expr_value(
        program,
        function,
        rhs,
        f"expr:{node_id}:{_slug(lhs)}",
        op,
        lhs,
        env,
        accesses,
        values,
        known,
        edges,
    )
    if shape == "decl" or re.fullmatch(r"[A-Za-z_]\w*", lhs.strip()):
        if not _writes_matching(lhs, accesses):
            outgoing[lhs.strip()] = identity
    for access in _writes_matching(lhs, accesses):
        edges.append(DependencyEdge(identity, access.operation_id, "local-to-state"))
    return outgoing


def _expr_value(
    program: SemanticProgram,
    function: SemanticFunction,
    expr: str,
    suffix: str,
    op: str,
    lhs: str,
    env: dict[str, str],
    accesses: list[StateAccess],
    values: list[FlowValue],
    known: set[str],
    edges: list[DependencyEdge],
) -> str:
    identity = f"{function.identity}:{suffix}"
    deps = _expr_deps(program, function, expr, op, lhs, env, accesses, values, known, edges)
    provenance = join_provenance(_provenances(deps, values)) if deps else _literal_provenance(expr)
    _add(values, known, identity, "local", provenance, deps, function)
    for dep in deps:
        edges.append(DependencyEdge(dep, identity, "local-dependency"))
    return identity


def _expr_deps(
    program: SemanticProgram,
    function: SemanticFunction,
    expr: str,
    op: str,
    lhs: str,
    env: dict[str, str],
    accesses: list[StateAccess],
    values: list[FlowValue],
    known: set[str],
    edges: list[DependencyEdge],
) -> tuple[str, ...]:
    region = expr if op == "=" else f"{lhs} {expr}"
    deps: list[str] = []
    if op != "=":
        name = lhs.strip()
        if re.fullmatch(r"[A-Za-z_]\w*", name) and name in env:
            deps.append(env[name])
    for name in re.findall(r"\b([A-Za-z_]\w*)\b", region):
        if name in env and env[name] not in deps:
            deps.append(env[name])
    for access in accesses:
        if access.kind == "write":
            continue
        if _path_in(access.path, region):
            deps.append(access.operation_id)
    if re.search(r"\bmsg\.sender\b", region):
        deps.append(f"{function.identity}:caller")
    if re.search(r"\bmsg\.value\b", region):
        sender = f"{function.identity}:msg.value"
        _add(values, known, sender, "msg.value", "msg.value", (), function)
        deps.append(sender)
    if re.search(r"\bblock\.(timestamp|number|coinbase|chainid)\b", region):
        block = f"{function.identity}:block"
        _add(values, known, block, "block", "block", (), function)
        deps.append(block)
    for site in function.call_sites:
        if not re.search(rf"\b{re.escape(site.callee)}\s*\(", region):
            continue
        if site.callee in _TYPE_NAMES:
            continue
        if site.external:
            returned = external_return(function, site)
            _add(
                values,
                known,
                returned.identity,
                returned.kind,
                returned.provenance,
                returned.dependencies,
                function,
                returned.span,
            )
            deps.append(returned.identity)
            continue
        resolution = resolve_call(program, function, site)
        ret_id = f"{function.identity}:internal-return:{site.call_id}"
        extra: list[str] = []
        provenance = "unknown"
        if resolution.status == "resolved":
            callee = _function_by_id(program, resolution.function_id)
            extra, provenance = (
                _direct_return(callee, region, env, values) if callee else ([], "unknown")
            )
        _add(values, known, ret_id, "internal-return", provenance, tuple(extra), function)
        for dep in extra:
            edges.append(DependencyEdge(dep, ret_id, "return-value"))
        deps.append(ret_id)
    return tuple(dict.fromkeys(deps))


def _direct_return(
    callee: SemanticFunction,
    call_expr: str,
    env: dict[str, str],
    values: list[FlowValue],
) -> tuple[list[str], str]:
    returned = _returned_expression(callee.source)
    if not returned:
        return [], "unknown"
    order = _param_order(callee.source)
    if returned in order:
        arguments = _call_arguments(call_expr, callee.name)
        index = order.index(returned)
        if index >= len(arguments):
            return [], "unknown"
        names = re.findall(r"\b([A-Za-z_]\w*)\b", arguments[index])
        deps = [env[name] for name in names if name in env]
        if not deps:
            return [], "unknown"
        return deps, join_provenance(_provenances(tuple(deps), values))
    if returned.split("[", 1)[0] in set(callee.reads) | set(callee.writes):
        return [], "state"
    return [], "unknown"


def _ordering(
    function: SemanticFunction,
    cfg: FunctionCfg,
    ranges: dict[int, tuple[int, int]],
    base: int,
    homes: dict[str, int | None],
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    may_read: list[tuple[str, str]] = []
    may_write: list[tuple[str, str]] = []
    read_ops: list[tuple[str, str]] = []
    write_ops: list[tuple[str, str]] = []
    if not getattr(cfg, "known", False):
        return (), (), (), ()
    nodes = getattr(cfg, "nodes", [])
    for site in function.call_sites:
        call_node = node_at(ranges, site.span.start_byte - base)
        if call_node is None:
            continue
        before = {
            int(node.node_id) for node in nodes if call_node in cfg.reachable_from(node.node_id)
        }
        after = cfg.reachable_from(call_node)
        for access in function.access_sites:
            home = homes.get(access.operation_id)
            if home is None or home == call_node:
                continue
            if access.kind != "write" and home in before:
                may_read.append((site.call_id, access.path))
                read_ops.append((site.call_id, access.operation_id))
            if access.kind != "read" and home in after:
                may_write.append((site.call_id, access.path))
                write_ops.append((site.call_id, access.operation_id))
    return tuple(may_read), tuple(may_write), tuple(read_ops), tuple(write_ops)


def _influence(function: SemanticFunction, edges: list[DependencyEdge]) -> None:
    reads = {item.operation_id for item in function.access_sites if item.kind != "write"}
    writes = [item.operation_id for item in function.access_sites if item.kind != "read"]
    for write_id in writes:
        seen: set[str] = set()
        stack = [write_id]
        reached: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for edge in edges:
                if edge.sink != current or edge.kind == "state-read-influences-write":
                    continue
                if edge.source in reads and edge.source != write_id:
                    reached.append(edge.source)
                if edge.source not in seen:
                    stack.append(edge.source)
        for read_id in dict.fromkeys(reached):
            edges.append(DependencyEdge(read_id, write_id, "state-read-influences-write"))


def external_return(function: SemanticFunction, site: object) -> FlowValue:
    call_type = str(getattr(site, "call_type", "") or "")
    call_id = str(getattr(site, "call_id", "") or "")
    span = getattr(site, "span", None)
    box = (
        getattr(span, "start_byte", 0),
        getattr(span, "end_byte", 0),
        getattr(span, "start_line", 0),
        getattr(span, "end_line", 0),
    )
    return FlowValue(
        f"{function.identity}:return:{call_id}",
        "external-return",
        "oracle" if call_type == "oracle" else "external",
        (call_id,),
        "",
        function.identity,
        box,
    )


def returned_reads(callee: SemanticFunction) -> tuple[str, ...]:
    """Operation ids of state reads named by the callee's return expression."""
    expression = _returned_expression(callee.source)
    if not expression:
        return ()
    return tuple(
        item.operation_id
        for item in callee.access_sites
        if item.kind != "write" and _path_in(item.path, expression)
    )


def _inherited(program: SemanticProgram, function: SemanticFunction, name: str) -> CallResolution:
    found: list[SemanticFunction] = []
    seen: set[str] = set()
    pending = list(_bases(program, function.contract))
    steps = 0
    while pending and steps < 8:
        base = pending.pop(0)
        steps += 1
        if base in seen:
            continue
        seen.add(base)
        found.extend(
            item for item in program.functions if item.contract == base and item.name == name
        )
        pending.extend(_bases(program, base))
    unique = {item.identity: item for item in found}
    if len(unique) == 1:
        chosen = tuple(unique.values())[0]
        return CallResolution("resolved", chosen.identity, "inherited")
    if len(unique) > 1:
        return CallResolution("unknown", reason="inherited callee is ambiguous")
    return CallResolution("unknown", reason="inherited callee was not resolved")


def _bases(program: SemanticProgram, contract: str) -> tuple[str, ...]:
    fact = _contract_fact(program, contract)
    return fact.bases if fact is not None else ()


def _contract_fact(program: SemanticProgram, name: str) -> ContractFact | None:
    for item in program.contracts:
        if item.name == name:
            return item
    return None


def _function_by_id(program: SemanticProgram, identity: str) -> SemanticFunction | None:
    for item in program.functions:
        if item.identity == identity:
            return item
    return None


def _split_statement(text: str) -> tuple[str, str, str, str]:
    stripped = text.strip().rstrip(";")
    returned = re.match(r"return\b(.*)", stripped)
    if returned:
        return "return", "", "", returned.group(1).strip()
    declared = re.match(
        r"(?:u?int\d*|int\d*|address|bool|string|bytes\d*|bytes)\s+"
        r"(?:memory\s+|storage\s+|calldata\s+)?([A-Za-z_]\w*)\s*(?:=\s*(.+))?$",
        stripped,
    )
    if declared:
        return "decl", declared.group(1), "=", (declared.group(2) or "").strip()
    paired = re.match(r"\(([^)]+)\)\s*=\s*(.+)$", stripped)
    if paired:
        return "tuple", paired.group(1), "=", paired.group(2).strip()
    assign = _top_assign(stripped)
    if assign is not None:
        op, lhs, rhs, _lhs_at, _rhs_at = assign
        return "assign", lhs.strip(), op, rhs.strip().rstrip(";")
    return "other", "", "", stripped


def _writes_matching(lhs: str, accesses: list[StateAccess]) -> list[StateAccess]:
    return [
        item
        for item in accesses
        if item.kind != "read" and (_path_in(item.path, lhs) or item.path == lhs.strip())
    ]


def _path_in(path: str, expr: str) -> bool:
    if not path or not expr:
        return False
    start = 0
    while True:
        at = expr.find(path, start)
        if at < 0:
            return False
        end = at + len(path)
        if end >= len(expr) or expr[end] not in "[.A-Za-z0-9_":
            return True
        start = at + 1


def _literal_provenance(expr: str) -> str:
    text = expr.strip().rstrip(";")
    if re.fullmatch(r"0x[0-9a-fA-F]+", text):
        return "trusted_constant"
    if re.fullmatch(r"\d+", text):
        return "constant"
    if text in {"true", "false"}:
        return "constant"
    return "unknown"


def _provenances(deps: tuple[str, ...], values: list[FlowValue]) -> set[str]:
    known = {item.identity: item.provenance for item in values}
    return {known[item] for item in deps if item in known}


def _returned_expression(source: str) -> str:
    match = re.search(r"\breturn\b\s+([^;]+);", source)
    return match.group(1).strip() if match else ""


def _param_order(source: str) -> list[str]:
    header = source.split("{", 1)[0]
    if "(" not in header:
        return []
    params = header[header.find("(") + 1 : header.rfind(")")]
    names: list[str] = []
    for part in params.split(","):
        match = re.search(r"([A-Za-z_]\w*)\s*$", part.strip())
        if match and match.group(1) not in _TYPE_NAMES:
            names.append(match.group(1))
    return names


def _call_arguments(expr: str, name: str) -> list[str]:
    match = re.search(rf"\b{re.escape(name)}\s*\((.*)\)", expr)
    if match is None:
        return []
    inner = match.group(1)
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(inner):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(inner[start:index].strip())
            start = index + 1
    tail = inner[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned[:40] or "value"


def _add(
    values: list[FlowValue],
    known: set[str],
    identity: str,
    kind: str,
    provenance: str,
    deps: tuple[str, ...],
    function: SemanticFunction,
    span: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> None:
    if identity in known:
        return
    known.add(identity)
    values.append(FlowValue(identity, kind, provenance, deps, "", function.identity, span))


def _unique(edges: Iterable[DependencyEdge]) -> list[DependencyEdge]:
    seen: set[tuple[str, str, str]] = set()
    found: list[DependencyEdge] = []
    for edge in edges:
        key = (edge.source, edge.sink, edge.kind)
        if key in seen:
            continue
        seen.add(key)
        found.append(edge)
    return found
