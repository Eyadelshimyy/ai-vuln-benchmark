"""Bounded Yul structure for Solidity assembly blocks.

This is not an EVM interpreter and not a symbolic executor. Literal slots,
constant aliases, and direct ``let`` copies can be known. Arithmetic, unknown
names, and duplicate constants stay unknown. A node limit marks the model
incomplete rather than safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.parsing.model import SyntaxGraph
from app.parsing.solidity_cfg import _skip_string_or_comment

_MAX_BLOCKS = 32
_MAX_NODES = 200


def _yul_limits() -> tuple[int, int]:
    from app.core.config import get_settings

    settings = get_settings()
    blocks = int(getattr(settings, "solidity_yul_max_blocks", _MAX_BLOCKS) or _MAX_BLOCKS)
    nodes = int(getattr(settings, "solidity_yul_max_nodes", _MAX_NODES) or _MAX_NODES)
    return blocks, nodes


_IMPL_SLOT = "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
_ADMIN_SLOT = "b53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
_BEACON_SLOT = "a3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"
_CRITICAL = {
    _IMPL_SLOT: "implementation",
    _ADMIN_SLOT: "admin",
    _BEACON_SLOT: "beacon",
}
_OPS = "sstore|sload|call|delegatecall|staticcall|callcode"
_COMPUTED = "add|sub|mul|div|sdiv|mod|smod|exp|shl|shr|sar|byte|signextend|and|or|xor|not"
_CALLER = re.compile(r"\b(calldataload|calldatacopy|caller|origin)\b")
_TRANSFERS = frozenset({"call", "delegatecall", "staticcall", "callcode"})


@dataclass(frozen=True)
class YulExpression:
    text: str
    kind: str
    slot_class: str = "unknown"
    role: str = ""


@dataclass(frozen=True)
class YulVariable:
    name: str
    expression: YulExpression
    branch_dependent: bool


@dataclass(frozen=True)
class YulAssignment:
    name: str
    expression: YulExpression
    branch_dependent: bool


@dataclass(frozen=True)
class YulLoad:
    variable: str
    slot: YulExpression
    branch_dependent: bool


@dataclass(frozen=True)
class YulStore:
    slot: YulExpression
    value: str
    branch_dependent: bool
    value_from: str = ""


@dataclass(frozen=True)
class YulCall:
    op: str
    target: str
    target_from: str
    success: str
    branch_dependent: bool
    success_ignored: bool


@dataclass(frozen=True)
class YulBranch:
    kind: str
    condition: str


@dataclass(frozen=True)
class YulReturn:
    kind: str
    branch_dependent: bool


@dataclass(frozen=True)
class YulTrace:
    summary: str
    known: bool


@dataclass
class _Flow:
    stores: int = 0
    external: bool = False


@dataclass
class YulModel:
    variables: list[YulVariable] = field(default_factory=list)
    assignments: list[YulAssignment] = field(default_factory=list)
    loads: list[YulLoad] = field(default_factory=list)
    stores: list[YulStore] = field(default_factory=list)
    calls: list[YulCall] = field(default_factory=list)
    branches: list[YulBranch] = field(default_factory=list)
    returns: list[YulReturn] = field(default_factory=list)
    traces: list[YulTrace] = field(default_factory=list)
    hostile: list[str] = field(default_factory=list)
    incomplete: bool = False
    limit_reason: str = ""
    nodes: int = 0
    compiler_ir: str = "unavailable"
    compiler_status: str = ""


@dataclass
class _SuccessScope:
    """Call-success bindings for one Yul block. A branch gets its own copy."""

    bindings: dict[str, int] = field(default_factory=dict)
    checked: set[int] = field(default_factory=set)
    owned: set[int] = field(default_factory=set)
    rests: dict[int, str] = field(default_factory=dict)

    def child(self) -> _SuccessScope:
        return _SuccessScope(dict(self.bindings), set(self.checked), set(), self.rests)


def analyze_yul(graph: SyntaxGraph) -> YulModel:
    model = YulModel()
    if graph.language != "solidity":
        return model
    constants = _constants(graph)
    seen_hostile: set[str] = set()
    for event in graph.events:
        if model.incomplete:
            break
        if event.kind != "sol_function":
            continue
        for block in _assembly_blocks(_strip_comments(event.text), model):
            if model.incomplete:
                break
            block_limit, node_limit = _yul_limits()
            if model.nodes >= node_limit or len(model.loads) + len(model.calls) >= block_limit:
                _limit(model)
                break
            if re.search(r"\bassembly\b", block):
                _trace(model, "nested assembly", False)
            _note_unbounded_calldata(block, model, seen_hostile)
            _walk_block(
                block,
                constants,
                model,
                branch=False,
                flow=_Flow(),
                seen=seen_hostile,
                scope=_SuccessScope(),
            )
    _note_compiler(model)
    return model


def _walk_block(
    block: str,
    constants: dict[str, list[str]],
    model: YulModel,
    *,
    branch: bool,
    flow: _Flow,
    seen: set[str],
    scope: _SuccessScope,
) -> None:
    env: dict[str, YulExpression] = {}
    index = 0
    while index < len(block) and not model.incomplete:
        _, node_limit = _yul_limits()
        if model.nodes >= node_limit:
            _limit(model)
            break
        skipped = _skip_string_or_comment(block, index)
        if skipped is not None:
            index = skipped
            continue
        while index < len(block) and block[index].isspace():
            index += 1
        if index >= len(block):
            break
        skipped = _skip_string_or_comment(block, index)
        if skipped is not None:
            index = skipped
            continue
        if block.startswith("if", index) and _boundary(block, index):
            model.branches.append(YulBranch("if", "condition"))
            model.nodes += 1
            index = _consume_if(block, index, constants, model, flow, seen, scope)
            continue
        if block.startswith("switch", index) and _boundary(block, index):
            model.branches.append(YulBranch("switch", "condition"))
            model.nodes += 1
            index = _consume_switch(block, index, constants, model, flow, seen, scope)
            continue
        if block.startswith("else", index) and _boundary(block, index):
            model.branches.append(YulBranch("else", "condition"))
            model.nodes += 1
            index = _consume_braced(block, index, constants, model, flow, seen, scope)
            continue
        if _at_return(block, index):
            kind = "stop" if block.startswith("stop", index) else "return"
            model.returns.append(YulReturn(kind, branch))
            model.nodes += 1
            index += len(kind)
            continue
        let = re.match(r"let\s+([A-Za-z_]\w*)\s*:=\s*", block[index:])
        assign = re.match(r"([A-Za-z_]\w*)\s*:=\s*", block[index:])
        if let or assign:
            matched = let or assign
            assert matched is not None
            name = matched.group(1)
            expr_at = index + matched.end()
            _overwrite_success(name, scope, model, seen)
            if _starts_op(block, expr_at) and not _starts_with(block, expr_at, "sload"):
                index = _bind_op(
                    block,
                    expr_at,
                    name,
                    is_let=let is not None,
                    env=env,
                    constants=constants,
                    model=model,
                    branch=branch,
                    flow=flow,
                    seen=seen,
                    scope=scope,
                )
                continue
            expression, end = _read_expr(block, expr_at)
            classified = _classify(expression, env, constants)
            model.nodes += 1
            if let:
                model.variables.append(YulVariable(name, classified, branch))
            else:
                model.assignments.append(YulAssignment(name, classified, branch))
            env[name] = classified
            if classified.kind == "load":
                model.loads.append(YulLoad(name, classified, branch))
                _trace(
                    model,
                    f"{name} <- storage[{classified.text}]",
                    classified.slot_class == "known",
                )
            index = end
            continue
        if _starts_op(block, index):
            index = _bare_op(block, index, env, constants, model, branch, flow, seen, scope)
            continue
        index += 1
    _finalize_success(model, scope, seen)


def _bind_op(
    block: str,
    expr_at: int,
    name: str,
    *,
    is_let: bool,
    env: dict[str, YulExpression],
    constants: dict[str, list[str]],
    model: YulModel,
    branch: bool,
    flow: _Flow,
    seen: set[str],
    scope: _SuccessScope,
) -> int:
    op_match = re.match(rf"({_OPS})\s*\(", block[expr_at:])
    if op_match is None:
        return expr_at + 1
    op = op_match.group(1)
    open_at = expr_at + op_match.end() - 1
    close = _close_paren(block, open_at)
    if close < 0:
        _unbalanced(model, "call")
        return len(block)
    args = _split_args(block[open_at + 1 : close])
    success = name if is_let else ""
    _record_op(
        model,
        env,
        constants,
        op,
        args,
        success,
        branch,
        flow,
        block[close + 1 :],
        seen,
        scope,
    )
    if is_let and op in _TRANSFERS:
        env[name] = YulExpression(op, "call", "unknown", "")
    return close + 1


def _bare_op(
    block: str,
    index: int,
    env: dict[str, YulExpression],
    constants: dict[str, list[str]],
    model: YulModel,
    branch: bool,
    flow: _Flow,
    seen: set[str],
    scope: _SuccessScope,
) -> int:
    op_match = re.match(rf"({_OPS})\s*\(", block[index:])
    if op_match is None:
        return index + 1
    op = op_match.group(1)
    open_at = index + op_match.end() - 1
    close = _close_paren(block, open_at)
    if close < 0:
        _unbalanced(model, "call")
        return len(block)
    args = _split_args(block[open_at + 1 : close])
    _record_op(model, env, constants, op, args, "", branch, flow, block[close + 1 :], seen, scope)
    return close + 1


def _consume_if(
    block: str,
    index: int,
    constants: dict[str, list[str]],
    model: YulModel,
    flow: _Flow,
    seen: set[str],
    scope: _SuccessScope,
) -> int:
    brace = block.find("{", index)
    if brace >= 0:
        _note_condition(block[index:brace], scope)
    end = _consume_braced(block, index, constants, model, flow, seen, scope)
    cursor = end
    while cursor < len(block) and block[cursor].isspace():
        cursor += 1
    if cursor < len(block) and block.startswith("else", cursor) and _boundary(block, cursor):
        model.branches.append(YulBranch("else", "condition"))
        model.nodes += 1
        return _consume_braced(block, cursor, constants, model, flow, seen, scope)
    return end


def _consume_braced(
    block: str,
    index: int,
    constants: dict[str, list[str]],
    model: YulModel,
    flow: _Flow,
    seen: set[str],
    scope: _SuccessScope,
) -> int:
    brace = block.find("{", index)
    if brace < 0:
        return len(block)
    end = _close_brace(block, brace)
    if end < 0:
        _unbalanced(model, "branch")
        return len(block)
    child = _Flow(flow.stores, flow.external)
    _walk_block(
        block[brace + 1 : end],
        constants,
        model,
        branch=True,
        flow=child,
        seen=seen,
        scope=scope.child(),
    )
    return end + 1


def _consume_switch(
    block: str,
    index: int,
    constants: dict[str, list[str]],
    model: YulModel,
    flow: _Flow,
    seen: set[str],
    scope: _SuccessScope,
) -> int:
    cursor = index + len("switch")
    while cursor < len(block):
        skipped = _skip_string_or_comment(block, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        if block.startswith(("case", "default"), cursor) and _boundary(block, cursor):
            break
        cursor += 1
    _note_condition(block[index:cursor], scope)
    while cursor < len(block) and not model.incomplete:
        skipped = _skip_string_or_comment(block, cursor)
        if skipped is not None:
            cursor = skipped
            continue
        while cursor < len(block) and block[cursor].isspace():
            cursor += 1
        if cursor >= len(block):
            return cursor
        if block.startswith(("case", "default"), cursor) and _boundary(block, cursor):
            cursor = _consume_braced(block, cursor, constants, model, flow, seen, scope)
            continue
        return cursor
    return cursor


def _record_op(
    model: YulModel,
    env: dict[str, YulExpression],
    constants: dict[str, list[str]],
    op: str,
    args: list[str],
    success: str,
    branch: bool,
    flow: _Flow,
    rest: str,
    seen: set[str],
    scope: _SuccessScope,
) -> None:
    model.nodes += 1
    if op == "sload" and args:
        slot = _classify(args[0], env, constants)
        model.loads.append(YulLoad(success, slot, branch))
        return
    if op == "sstore" and args:
        slot = _classify(args[0], env, constants)
        value = args[1].strip() if len(args) > 1 else ""
        origin = env.get(value)
        value_from = origin.text if origin is not None and origin.kind == "load" else ""
        model.stores.append(YulStore(slot, value, branch, value_from))
        if origin is not None and origin.kind == "load" and slot.text == origin.text:
            _trace(model, f"sstore({slot.text}) <- sload({origin.text})", True)
        if slot.role in {"implementation", "admin", "beacon"}:
            _hostile(model, seen, f"sstore into {slot.role} slot")
        elif slot.slot_class == "computed":
            _hostile(model, seen, "computed storage write")
        if flow.external:
            _hostile(model, seen, "storage write after external control transfer")
        flow.stores += 1
        return
    if op not in _TRANSFERS:
        return
    target = args[1].strip() if len(args) > 1 else ""
    origin = env.get(target)
    target_from = origin.text if origin is not None else target
    model.calls.append(YulCall(op, target, target_from, success, branch, not success))
    index = len(model.calls) - 1
    scope.rests[index] = rest
    if success:
        scope.bindings[success] = index
        scope.owned.add(index)
    else:
        _hostile(model, seen, f"ignored {op} success")
        if re.search(r"\breturndatacopy\s*\(", rest):
            _hostile(model, seen, "return-data confusion")
    if origin is not None and origin.kind == "load":
        known = origin.slot_class == "known"
        _trace(model, f"{op}.target <- {target}", known)
        if origin.slot_class == "computed":
            _hostile(model, seen, f"dynamic {op} target")
    else:
        classified = origin if origin is not None else _classify(target, env, constants)
        if classified.slot_class == "computed":
            _hostile(model, seen, f"dynamic {op} target")
    source = f"{target} {target_from}"
    if _CALLER.search(source):
        if op == "delegatecall":
            _hostile(model, seen, "delegatecall to a caller-derived value")
        elif op == "call":
            _hostile(model, seen, "arbitrary call")
    if flow.stores:
        _trace(model, "call after storage write", False)
    flow.external = True


def _overwrite_success(name: str, scope: _SuccessScope, model: YulModel, seen: set[str]) -> None:
    if name not in scope.bindings:
        return
    index = scope.bindings.pop(name)
    scope.owned.discard(index)
    if index not in scope.checked:
        _mark_ignored(model, index, scope.rests.get(index, ""), seen)


def _finalize_success(model: YulModel, scope: _SuccessScope, seen: set[str]) -> None:
    for index in list(scope.owned):
        if index not in scope.checked:
            _mark_ignored(model, index, scope.rests.get(index, ""), seen)


def _mark_ignored(model: YulModel, index: int, rest: str, seen: set[str]) -> None:
    if index < 0 or index >= len(model.calls):
        return
    call = model.calls[index]
    if call.success_ignored or call.op not in _TRANSFERS:
        return
    model.calls[index] = YulCall(
        call.op,
        call.target,
        call.target_from,
        call.success,
        call.branch_dependent,
        True,
    )
    _hostile(model, seen, f"ignored {call.op} success")
    if re.search(r"\breturndatacopy\s*\(", rest):
        _hostile(model, seen, "return-data confusion")


def _note_condition(condition: str, scope: _SuccessScope) -> None:
    cleaned = _strip_comments(condition)
    cleaned = re.sub(r'"(?:\\.|[^"\\])*"', " ", cleaned)
    cleaned = re.sub(r"'(?:\\.|[^'\\])*'", " ", cleaned)
    for name, index in scope.bindings.items():
        if re.search(rf"\b{re.escape(name)}\b", cleaned):
            scope.checked.add(index)


def _note_compiler(model: YulModel) -> None:
    """Record compiler IR availability. This does not interpret the IR."""
    from app.parsing.solidity_project import current_compiler_project

    project = current_compiler_project()
    if project is None:
        model.compiler_ir = "unavailable"
        return
    model.compiler_status = project.status
    if project.ir_truncated:
        model.compiler_ir = "truncated"
    elif project.ir_available:
        model.compiler_ir = "available"
    else:
        model.compiler_ir = "unavailable"
    if project.status != "AVAILABLE" or not project.complete:
        return
    slots = {
        str(item.get("slot", "")): f"{item.get('contract', '')}.{item.get('label', '')}"
        for item in project.layouts
        if item.get("slot") and item.get("label")
    }
    for load in model.loads:
        if load.slot.slot_class != "known":
            continue
        label = slots.get(load.slot.text.strip())
        if label:
            _trace(model, f"sload matches compiler layout {label}", True)


def _classify(
    expression: str, env: dict[str, YulExpression], constants: dict[str, list[str]]
) -> YulExpression:
    text = expression.strip().rstrip(";")
    loaded = re.fullmatch(r"sload\s*\(\s*(.+)\s*\)", text)
    if loaded:
        slot = _classify(loaded.group(1), env, constants)
        return YulExpression(slot.text, "load", slot.slot_class, slot.role)
    if text in env:
        return env[text]
    values = constants.get(text)
    if values is None:
        return _slot_expression(text)
    if len(values) != 1:
        return YulExpression(text, "variable", "unknown", "")
    resolved = _slot_expression(values[0])
    return YulExpression(text, "variable", resolved.slot_class, resolved.role)


def _slot_expression(text: str) -> YulExpression:
    compact = text.lower().replace("0x", "").replace("_", "")
    role = _CRITICAL.get(compact, "")
    if re.fullmatch(r"0x[0-9a-fA-F_]+|\d+", text):
        return YulExpression(text, "literal", "known", role)
    if re.search(r'keccak256\s*\(\s*"[^"]+"\s*\)', text) or "erc7201" in text.lower():
        return YulExpression(text, "expression", "namespaced", "")
    if re.search(r"[+\-*/]", text) or re.match(rf"(?:{_COMPUTED}|keccak256)\s*\(", text):
        return YulExpression(text, "expression", "computed", "")
    return YulExpression(text, "variable", "unknown", "")


def _constants(graph: SyntaxGraph) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for event in graph.events:
        if event.kind != "sol_state":
            continue
        fields = _fields(event.extra)
        if fields.get("mutability") != "constant":
            continue
        matched = re.search(r"=\s*(0x[0-9a-fA-F]+|\d+)\s*;", event.text)
        if not matched:
            continue
        found.setdefault(fields.get("name", ""), []).append(matched.group(1))
    return found


def _assembly_blocks(text: str, model: YulModel) -> list[str]:
    blocks: list[str] = []
    cursor = 0
    while True:
        match = re.search(r"\bassembly\b", text[cursor:])
        if match is None:
            break
        brace = text.find("{", cursor + match.end())
        if brace < 0:
            _unbalanced(model, "assembly")
            break
        end = _close_brace(text, brace)
        if end < 0:
            _unbalanced(model, "assembly")
            break
        blocks.append(text[brace + 1 : end])
        cursor = end + 1
    return blocks


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//.*?$", " ", text, flags=re.MULTILINE)


def _read_expr(text: str, index: int) -> tuple[str, int]:
    depth = 0
    start = index
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None and depth == 0 and text[index] in "\"'":
            index = skipped
            continue
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char in ";\n" and depth == 0:
            return text[start:index].strip(), index + 1
        index += 1
    return text[start:index].strip(), index


def _close_paren(text: str, open_at: int) -> int:
    depth = 0
    index = open_at
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _close_brace(text: str, open_at: int) -> int:
    depth = 0
    index = open_at
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _split_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _starts_op(text: str, index: int) -> bool:
    return _boundary(text, index) and re.match(rf"({_OPS})\s*\(", text[index:]) is not None


def _starts_with(text: str, index: int, word: str) -> bool:
    return text.startswith(word, index) and _boundary(text, index)


def _at_return(text: str, index: int) -> bool:
    if text.startswith("return", index) and _boundary(text, index):
        return True
    return text.startswith("stop", index) and _boundary(text, index)


def _boundary(text: str, index: int) -> bool:
    return index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")


def _fields(extra: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in extra.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def _limit(model: YulModel) -> None:
    model.incomplete = True
    model.limit_reason = "Yul node limit reached; further assembly is unknown"


def _unbalanced(model: YulModel, kind: str) -> None:
    model.incomplete = True
    model.limit_reason = f"unbalanced Yul {kind}; the rest of the block is unknown"


def _trace(model: YulModel, summary: str, known: bool) -> None:
    model.traces.append(YulTrace(summary, known))


def _note_unbounded_calldata(block: str, model: YulModel, seen: set[str]) -> None:
    """Flag a dynamic ``calldataload`` that the block never bounds with ``calldatasize``.

    A numeric offset such as ``calldataload(0)`` is not this pattern. A later
    ``calldatasize`` check in the same block keeps the load unknown rather than hostile.
    """
    if re.search(r"\bcalldatasize\s*\(", block):
        return
    for match in re.finditer(r"\bcalldataload\s*\(\s*([^)]*)\)", block):
        argument = " ".join(match.group(1).split())
        if re.fullmatch(r"(?:0x[0-9a-fA-F]+|\d+)", argument):
            continue
        _hostile(model, seen, "unbounded calldata load")
        return


def _hostile(model: YulModel, seen: set[str], summary: str) -> None:
    if summary in seen:
        return
    seen.add(summary)
    model.hostile.append(summary)
