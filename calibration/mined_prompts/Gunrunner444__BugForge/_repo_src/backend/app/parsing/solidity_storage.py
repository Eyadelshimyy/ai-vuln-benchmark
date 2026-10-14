"""Parser storage layout, with an optional compiler overlay.

Slots computed here follow Solidity's packing rules only when every type and
every base is known. Ambiguous inheritance, an unknown type, or a missing
imported base leaves the slot unknown. A compiler layout is evidence from
that compiler. It does not replace the parser layout, and a disagreement is
kept rather than silently resolved. Nothing here is a compiler proof.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from app.parsing.keccak import keccak256
from app.parsing.model import SyntaxGraph
from app.parsing.solidity_cfg import _consume_parens
from app.parsing.solidity_compiler import (
    CompilerSemantics,
    SemanticDisagreement,
    reconcile_semantics,
)
from app.parsing.solidity_yul import YulModel, analyze_yul

_CTX: ContextVar[_StorageContext | None] = ContextVar("bugforge_storage_context", default=None)
_ELEM = {
    "bool": 1,
    "byte": 1,
    "address": 20,
    "uint": 32,
    "int": 32,
    "string": None,
    "bytes": None,
}


@dataclass(frozen=True)
class StorageVariable:
    contract: str
    name: str
    type_name: str
    slot: int | None
    offset: int | None
    packing_group: int | None
    visibility: str
    mutability: str
    origin: str
    key_type: str = ""
    value_type: str = ""
    members: tuple[str, ...] = ()
    uncertain: bool = False


@dataclass(frozen=True)
class YulAccess:
    contract: str
    function: str
    op: str
    expression: str
    known: bool
    literal: str = ""
    constant: str = ""
    resolved: str = ""
    value_from: str = ""


@dataclass(frozen=True)
class LayoutDisagreement:
    contract: str
    name: str
    parser_slot: str
    compiler_slot: str
    field: str = "slot"
    parser_value: str = ""
    compiler_value: str = ""


_CHANGE_KINDS = frozenset(
    {
        "added_before_existing",
        "removed_existing",
        "reordered",
        "type_changed",
        "slot_changed",
        "offset_changed",
        "packing_changed",
        "inheritance_changed",
        "mapping_changed",
        "struct_changed",
        "unknown",
    }
)


@dataclass(frozen=True)
class StorageChange:
    kind: str
    variable: str
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in _CHANGE_KINDS:
            raise ValueError(f"unknown storage change {self.kind}")


@dataclass(frozen=True)
class StorageLayoutComparison:
    left_contract: str
    right_contract: str
    relationship: str
    compatible: bool | None
    confidence: str
    changes: tuple[StorageChange, ...] = ()
    appended: tuple[str, ...] = ()


@dataclass(frozen=True)
class StorageNamespace:
    name: str
    slot_expression: str
    resolved_slot: str
    known: bool
    source: str


@dataclass
class StorageModel:
    variables: list[StorageVariable] = field(default_factory=list)
    order: dict[str, tuple[str, ...]] = field(default_factory=dict)
    uncertain: dict[str, str] = field(default_factory=dict)
    yul: list[YulAccess] = field(default_factory=list)
    disagreements: list[LayoutDisagreement] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    compiler_status: str = ""
    compiler_matches: int = 0
    overlaps: list[str] = field(default_factory=list)
    comparisons: list[StorageLayoutComparison] = field(default_factory=list)
    namespaces: list[StorageNamespace] = field(default_factory=list)
    yul_detail: YulModel | None = None
    semantic_disagreements: list[SemanticDisagreement] = field(default_factory=list)


@dataclass
class _ContractFacts:
    file_path: str
    bases: list[str]
    variables: list[tuple[str, str, str, str]]
    structs: dict[str, tuple[str, ...]]


@dataclass
class _StorageContext:
    facts: dict[str, _ContractFacts]
    ambiguous: set[str]
    cache: dict[int, StorageModel]


def set_storage_context(graphs: dict[str, SyntaxGraph]) -> Token[_StorageContext | None]:
    facts: dict[str, _ContractFacts] = {}
    ambiguous: set[str] = set()
    for graph in graphs.values():
        if graph.language != "solidity":
            continue
        for name, fact in _facts(graph).items():
            if name in facts:
                ambiguous.add(name)
            else:
                facts[name] = fact
    for name in ambiguous:
        facts.pop(name, None)
    return _CTX.set(_StorageContext(facts, ambiguous, {}))


def reset_storage_context(token: Token[_StorageContext | None]) -> None:
    _CTX.reset(token)


def storage_context_active() -> bool:
    return _CTX.get() is not None


def analyze_storage(graph: SyntaxGraph) -> StorageModel:
    """Return the parser layout for ``graph``.

    A scan-level cache is used when the engine has installed one. Direct
    calls recompute. Compiler facts are not invented here.
    """
    ctx = _CTX.get()
    if ctx is not None and id(graph) in ctx.cache:
        return ctx.cache[id(graph)]
    model = _analyze(graph, ctx)
    if ctx is not None:
        ctx.cache[id(graph)] = model
    return model


def apply_compiler_layout(
    model: StorageModel, compiler: CompilerSemantics, *, source_path: str = ""
) -> StorageModel:
    """Record compiler slots beside the parser layout.

    Unavailable, failed, and version-only compiler results do not create
    slots. A match requires the contract and the variable label. A shared
    label across contracts is not a match. Ambiguous identity is recorded
    and left unmatched. Parser slots are never replaced.
    """
    model.compiler_status = compiler.status
    if compiler.status != "AVAILABLE":
        return model
    grouped, unlabeled = _compiler_index(compiler, source_path)
    if unlabeled:
        model.ambiguities.append(
            "Compiler layout entries without a contract were not matched by label."
        )
    wanted = _normalize_source(source_path)
    for variable in model.variables:
        if variable.uncertain or variable.slot is None:
            continue
        matches = list(grouped.get((variable.contract, variable.name), []))
        if not matches and variable.origin != variable.contract:
            matches = list(grouped.get((variable.origin, variable.name), []))
        chosen = _layouts_for_source(matches, wanted)
        if chosen is None:
            model.ambiguities.append(
                f"Compiler source identity does not match `{wanted or variable.contract}` "
                f"for `{variable.contract}.{variable.name}`."
            )
            continue
        if len(chosen) > 1:
            model.ambiguities.append(
                f"Ambiguous compiler identity for `{variable.contract}.{variable.name}`."
            )
            continue
        if len(chosen) != 1:
            continue
        matches = chosen
        model.compiler_matches += 1
        _record_compiler_difference(model, variable, matches[0])
    if compiler.status == "AVAILABLE":
        model.semantic_disagreements.extend(
            reconcile_semantics(_parser_layout_facts(model), compiler.layouts)
        )
    return model


def _normalize_source(path: str) -> str:
    text = path.replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _layouts_for_source(matches: list[dict[str, str]], wanted: str) -> list[dict[str, str]] | None:
    """Exact source identity. ``None`` means the compiler named a different file."""
    sourced = [item for item in matches if _normalize_source(str(item.get("source", "")))]
    unsourced = [item for item in matches if item not in sourced]
    exact = [item for item in sourced if _normalize_source(str(item.get("source", ""))) == wanted]
    if exact:
        return exact
    if sourced:
        return None
    return unsourced


def compare_storage_layouts(
    left: StorageModel,
    right: StorageModel,
    left_contract: str,
    right_contract: str,
    relationship: str = "layout",
) -> StorageLayoutComparison:
    """Compare two storage models without treating shared names as identity.

    An appended variable can remain compatible. A removed, reordered, or
    shifted variable is not. Unknown layouts stay unknown.
    """
    left_seq, left_reason = _storage_sequence(left, left_contract)
    right_seq, right_reason = _storage_sequence(right, right_contract)
    if left_reason or right_reason:
        return StorageLayoutComparison(
            left_contract,
            right_contract,
            relationship,
            None,
            "unknown",
            (StorageChange("unknown", "", left_reason or right_reason),),
        )
    changes, appended = _diff_sequences(left_seq, right_seq, left_contract, right_contract)
    confidence = "structural"
    if (
        left.compiler_status == "AVAILABLE"
        and right.compiler_status == "AVAILABLE"
        and not left.disagreements
        and not right.disagreements
    ):
        confidence = "compiler"
    return StorageLayoutComparison(
        left_contract,
        right_contract,
        relationship,
        not changes,
        confidence,
        tuple(changes),
        tuple(appended),
    )


def _parser_layout_facts(model: StorageModel) -> list[dict[str, str]]:
    return [
        {
            "contract": variable.contract,
            "symbol": variable.name,
            "slot": "" if variable.slot is None else str(variable.slot),
            "offset": "" if variable.offset is None else str(variable.offset),
            "type": variable.type_name,
            "uncertain": "true" if variable.uncertain or variable.slot is None else "",
        }
        for variable in model.variables
    ]


def _analyze(graph: SyntaxGraph, ctx: _StorageContext | None) -> StorageModel:
    model = StorageModel()
    if graph.language != "solidity" or graph.parser_tier.value == "profile_fallback":
        return model
    local = _facts(graph)
    facts = dict(ctx.facts) if ctx is not None else {}
    ambiguous = set(ctx.ambiguous) if ctx is not None else set()
    for name, fact in local.items():
        if name in ambiguous:
            continue
        facts.setdefault(name, fact)
    for contract, fact in local.items():
        if contract in ambiguous:
            model.uncertain[contract] = "more than one contract uses this name"
            model.variables.extend(_unplaced(fact, contract))
            continue
        order, reason = _linearize(contract, facts, ambiguous)
        if order is None:
            model.uncertain[contract] = reason
            model.order[contract] = ()
            model.variables.extend(_unplaced(fact, contract))
            continue
        model.order[contract] = tuple(order)
        placed, uncertain = _place(order, facts)
        if uncertain:
            model.uncertain[contract] = uncertain
            model.variables.extend(_unplaced_order(order, facts))
        else:
            model.variables.extend(placed)
    model.yul = _yul_accesses(graph, local)
    model.namespaces = _namespaces(graph)
    model.yul_detail = analyze_yul(graph)
    _record_comparisons(graph, model)
    return model


def _facts(graph: SyntaxGraph) -> dict[str, _ContractFacts]:
    bases: dict[str, list[str]] = {}
    variables: dict[str, list[tuple[str, str, str, str]]] = {}
    structs: dict[str, tuple[str, ...]] = {}
    struct_conflict: set[str] = set()
    for event in graph.events:
        if event.kind == "sol_contract":
            fields = _fields(event.extra)
            name = event.text.strip()
            raw = [item.strip() for item in fields.get("bases", "").split(",") if item.strip()]
            bases[name] = list(reversed(raw))
            variables.setdefault(name, [])
        elif event.kind == "sol_state":
            fields = _fields(event.extra)
            contract = fields.get("contract", "")
            variables.setdefault(contract, []).append(
                (
                    fields.get("name", ""),
                    fields.get("type", ""),
                    fields.get("mutability", "storage"),
                    _visibility(event.text),
                )
            )
        elif event.kind == "sol_type_def" and _fields(event.extra).get("kind") == "struct":
            fields = _fields(event.extra)
            name = fields.get("name", "")
            members = _abi_members(fields.get("abi", ""))
            if not name or not members:
                continue
            if name in structs and structs[name] != members:
                struct_conflict.add(name)
            else:
                structs[name] = members
    for name in struct_conflict:
        structs.pop(name, None)
    return {
        name: _ContractFacts(graph.file_path, bases.get(name, []), variables.get(name, []), structs)
        for name in set(bases) | set(variables)
    }


def _linearize(
    contract: str, facts: dict[str, _ContractFacts], ambiguous: set[str]
) -> tuple[list[str] | None, str]:
    def merge(sequences: list[list[str]]) -> list[str] | None:
        pending = [list(item) for item in sequences if item]
        result: list[str] = []
        while any(pending):
            chosen = ""
            for sequence in pending:
                if not sequence:
                    continue
                head = sequence[0]
                if any(head in tail[1:] for tail in pending if tail):
                    continue
                chosen = head
                break
            if not chosen:
                return None
            result.append(chosen)
            for sequence in pending:
                if sequence and sequence[0] == chosen:
                    del sequence[0]
        return result

    memo: dict[str, list[str]] = {}
    stack: set[str] = set()

    def walk(name: str) -> list[str] | None:
        if name in memo:
            return memo[name]
        if name in stack or name in ambiguous or name not in facts:
            return None
        stack.add(name)
        fact = facts[name]
        sequences: list[list[str]] = []
        for base in fact.bases:
            if base in ambiguous or base not in facts:
                stack.remove(name)
                return None
            base_order = walk(base)
            if base_order is None:
                stack.remove(name)
                return None
            sequences.append(base_order)
        sequences.append(list(fact.bases))
        merged = merge(sequences)
        stack.remove(name)
        if merged is None:
            return None
        memo[name] = merged + [name]
        return memo[name]

    order = walk(contract)
    if order is None:
        return None, "a base contract is missing, duplicated, or the inheritance order is ambiguous"
    return order, ""


def _place(order: list[str], facts: dict[str, _ContractFacts]) -> tuple[list[StorageVariable], str]:
    structs: dict[str, tuple[str, ...]] = {}
    for name in order:
        structs.update(facts[name].structs)
    slot = 0
    offset = 0
    group = 0
    placed: list[StorageVariable] = []

    def bump() -> None:
        nonlocal slot, offset, group
        if offset:
            slot += 1
            offset = 0
            group += 1

    def pack(size: int) -> tuple[int, int, int]:
        nonlocal slot, offset
        if offset + size > 32:
            bump()
        chosen = slot, offset, group
        offset += size
        if offset == 32:
            bump()
        return chosen

    for name in order:
        for var_name, type_name, mutability, visibility in facts[name].variables:
            if mutability in {"constant", "immutable"}:
                placed.append(
                    StorageVariable(
                        name,
                        var_name,
                        type_name,
                        None,
                        None,
                        None,
                        visibility,
                        mutability,
                        name,
                    )
                )
                continue
            kind, detail = _classify(type_name, structs)
            if kind == "unknown":
                return [], f"`{var_name}` has type `{type_name}`, which does not determine a slot"
            if kind in {"mapping", "dynamic", "dynamic_array"}:
                bump()
                key_type, value_type = detail if isinstance(detail, tuple) else ("", "")
                placed.append(
                    StorageVariable(
                        name,
                        var_name,
                        type_name,
                        slot,
                        0,
                        group,
                        visibility,
                        mutability,
                        name,
                        key_type=key_type,
                        value_type=value_type,
                    )
                )
                slot += 1
                offset = 0
                group += 1
                continue
            if kind == "elementary" and isinstance(detail, int):
                var_slot, var_offset, var_group = pack(detail)
                placed.append(
                    StorageVariable(
                        name,
                        var_name,
                        type_name,
                        var_slot,
                        var_offset,
                        var_group,
                        visibility,
                        mutability,
                        name,
                    )
                )
                continue
            if kind == "struct":
                members = structs.get(str(detail), ())
                sizes = [_elementary_size(item) for item in members]
                if any(size is None for size in sizes):
                    return [], f"struct `{detail}` is not packed from elementary members"
                bump()
                start = slot
                start_group = group
                for size in sizes:
                    assert size is not None
                    pack(size)
                bump()
                placed.append(
                    StorageVariable(
                        name,
                        var_name,
                        type_name,
                        start,
                        0,
                        start_group,
                        visibility,
                        mutability,
                        name,
                        members=members,
                    )
                )
                continue
            if kind == "fixed_array":
                count, inner = detail if isinstance(detail, tuple) else (0, "")
                inner_size = _elementary_size(str(inner))
                if inner_size is None or not isinstance(count, int):
                    return [], f"`{var_name}` is a fixed array whose element size is unknown"
                bump()
                start = slot
                start_group = group
                for _ in range(count):
                    pack(inner_size)
                bump()
                placed.append(
                    StorageVariable(
                        name,
                        var_name,
                        type_name,
                        start,
                        0,
                        start_group,
                        visibility,
                        mutability,
                        name,
                    )
                )
    return placed, ""


def _classify(type_name: str, structs: dict[str, tuple[str, ...]]) -> tuple[str, object]:
    compact = re.sub(r"\s+", "", type_name)
    if compact.startswith("mapping(") and compact.endswith(")"):
        body = compact[len("mapping(") : -1]
        key, value = _split_mapping(body)
        return "mapping", (key, value)
    if compact.endswith("[]"):
        return "dynamic_array", compact[:-2]
    fixed = re.fullmatch(r"(.+)\[(\d+)\]", compact)
    if fixed:
        return "fixed_array", (int(fixed.group(2)), fixed.group(1))
    if compact in {"string", "bytes"}:
        return "dynamic", ""
    size = _elementary_size(compact)
    if size is not None:
        return "elementary", size
    ident = re.fullmatch(r"[A-Za-z_]\w*", compact)
    if ident and ident.group(0) in structs:
        return "struct", ident.group(0)
    return "unknown", ""


def _elementary_size(type_name: str) -> int | None:
    compact = re.sub(r"\s+", "", type_name)
    if compact in _ELEM and _ELEM[compact] is not None:
        return _ELEM[compact]
    match = re.fullmatch(r"uint(\d+)|int(\d+)|bytes(\d+)", compact)
    if not match:
        return None
    if match.group(3):
        width = int(match.group(3))
        return width if 1 <= width <= 32 else None
    bits = int(match.group(1) or match.group(2))
    if bits % 8 or not 8 <= bits <= 256:
        return None
    return bits // 8


def _split_mapping(body: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif body.startswith("=>", index) and depth == 0:
            return body[:index], body[index + 2 :]
    return body, ""


def _unplaced(fact: _ContractFacts, contract: str) -> list[StorageVariable]:
    return [
        StorageVariable(
            contract,
            name,
            type_name,
            None,
            None,
            None,
            visibility,
            mutability,
            contract,
            uncertain=True,
        )
        for name, type_name, mutability, visibility in fact.variables
    ]


def _unplaced_order(order: list[str], facts: dict[str, _ContractFacts]) -> list[StorageVariable]:
    found: list[StorageVariable] = []
    for name in order:
        found.extend(_unplaced(facts[name], name))
    return found


def _record_comparisons(graph: SyntaxGraph, model: StorageModel) -> None:
    contracts = [name for name in model.order if name]
    links, unresolved = _proxy_links(graph, contracts)
    notes: list[str] = []
    for proxy, implementation in links:
        if proxy not in contracts or implementation not in contracts:
            continue
        comparison = compare_storage_layouts(
            model, model, proxy, implementation, "proxy-implementation"
        )
        model.comparisons.append(comparison)
        if proxy in model.uncertain or implementation in model.uncertain:
            model.ambiguities.append(
                f"Layout for `{proxy}` and `{implementation}` is incomplete. No collision is claimed."
            )
            continue
        slot_note = _slot_type_collision(model, proxy, implementation)
        if slot_note:
            notes.append(slot_note)
        elif comparison.compatible is False:
            notes.append(
                _incompatible_note("proxy-implementation", proxy, implementation, comparison)
            )
    by_proxy: dict[str, list[str]] = {}
    for proxy, implementation in links:
        by_proxy.setdefault(proxy, []).append(implementation)
    for implementations in by_proxy.values():
        unique = list(dict.fromkeys(implementations))
        for index, left_name in enumerate(unique):
            for right_name in unique[index + 1 :]:
                if left_name not in contracts or right_name not in contracts:
                    continue
                comparison = compare_storage_layouts(
                    model, model, left_name, right_name, "implementation-upgrade"
                )
                model.comparisons.append(comparison)
                if left_name in model.uncertain or right_name in model.uncertain:
                    model.ambiguities.append(
                        f"Implementation layout for `{left_name}` and `{right_name}` is incomplete. "
                        "No collision is claimed."
                    )
                    continue
                slot_note = _slot_type_collision(model, left_name, right_name)
                if slot_note:
                    notes.append(slot_note)
                elif comparison.compatible is False and not comparison.appended:
                    notes.append(
                        _incompatible_note(
                            "implementation-upgrade", left_name, right_name, comparison
                        )
                    )
    for proxy in unresolved:
        model.ambiguities.append(
            f"Delegatecall in `{proxy}` does not resolve to an implementation contract. "
            "No collision is claimed."
        )
    model.overlaps = list(dict.fromkeys(notes))


def _proxy_links(
    graph: SyntaxGraph, contracts: list[str]
) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair a proxy with implementations named in its delegatecall or assignments."""
    named = [name for name in contracts if name]
    links: list[tuple[str, str]] = []
    unresolved: list[str] = []
    for event in graph.events:
        if event.kind != "sol_delegatecall":
            continue
        proxy = _fields(event.extra).get("contract", "")
        if not proxy:
            continue
        mentioned = [
            name
            for name in named
            if name != proxy and re.search(rf"\b{re.escape(name)}\b", graph.source)
        ]
        proxy_source = "\n".join(
            item.text for item in graph.events if _fields(item.extra).get("contract", "") == proxy
        )
        resolved = [
            name for name in mentioned if re.search(rf"\b{re.escape(name)}\b", proxy_source)
        ]
        if resolved:
            links.extend((proxy, name) for name in resolved)
            continue
        unresolved.append(proxy)
    return links, unresolved


def _slot_type_collision(model: StorageModel, left: str, right: str) -> str:
    """Flag a shared slot whose type or packing differs, even when names differ."""
    right_at = {
        (item.slot, item.offset): item
        for item in model.variables
        if item.contract == right and item.slot is not None and not item.uncertain
    }
    for item in model.variables:
        if item.contract != left or item.slot is None or item.uncertain:
            continue
        other = right_at.get((item.slot, item.offset))
        if other is None or _type_signature(item) == _type_signature(other):
            continue
        return (
            f"Incompatible implementation-upgrade between `{left}` and `{right}`: "
            f"slot {item.slot} offset {item.offset} holds `{item.name}` ({item.type_name}) "
            f"and `{other.name}` ({other.type_name}). "
            "An append-only upgrade is not reported. "
            "This is potential evidence, not a confirmed collision."
        )
    return ""


def _shares_layout_name(model: StorageModel, left: str, right: str) -> bool:
    left_names = {item.name for item in model.variables if item.contract == left and item.name}
    right_names = {item.name for item in model.variables if item.contract == right and item.name}
    return bool(left_names & right_names)


def _incompatible_note(
    relationship: str, left_name: str, right_name: str, comparison: StorageLayoutComparison
) -> str:
    detail = "; ".join(
        f"{item.kind} `{item.variable}` {item.detail}".strip() for item in comparison.changes
    )
    return (
        f"Incompatible {relationship} between `{left_name}` and `{right_name}`: "
        f"{detail}. Shared slot numbers are not a collision by themselves. "
        "This is potential evidence, not a confirmed collision."
    )


def _storage_sequence(model: StorageModel, contract: str) -> tuple[list[StorageVariable], str]:
    if contract in model.uncertain:
        return [], model.uncertain[contract]
    order = model.order.get(contract, ())
    if not order and contract not in model.order:
        sequence = [
            item
            for item in model.variables
            if item.contract == contract
            and item.mutability == "storage"
            and item.slot is not None
            and not item.uncertain
        ]
        if sequence or any(item.contract == contract for item in model.variables):
            return sequence, ""
        return [], f"`{contract}` has no parser storage layout"
    allowed = set(order) or {contract}
    return (
        [
            item
            for item in model.variables
            if item.contract in allowed
            and item.mutability == "storage"
            and item.slot is not None
            and not item.uncertain
        ],
        "",
    )


def _type_signature(item: StorageVariable) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        re.sub(r"\s+", "", item.type_name),
        re.sub(r"\s+", "", item.key_type),
        re.sub(r"\s+", "", item.value_type),
        item.members,
    )


def _diff_sequences(
    left: list[StorageVariable],
    right: list[StorageVariable],
    left_root: str,
    right_root: str,
) -> tuple[list[StorageChange], list[str]]:
    if _same_members(left, right) and [(item.name, _type_signature(item)) for item in left] != [
        (item.name, _type_signature(item)) for item in right
    ]:
        return (
            [
                StorageChange(
                    "reordered",
                    item.name,
                    f"slot {item.slot}",
                )
                for item in left
            ],
            [],
        )
    changes: list[StorageChange] = []
    appended: list[str] = []
    cursor = 0
    for item in left:
        match_at = _find_match(item, right, cursor)
        if match_at is None:
            named = _find_named(item, right, cursor)
            if named is None:
                changes.append(StorageChange("removed_existing", item.name, f"slot {item.slot}"))
                continue
            _append_inserts(changes, right, cursor, named)
            changes.extend(_pair_changes(item, right[named], left_root, right_root))
            cursor = named + 1
            continue
        _append_inserts(changes, right, cursor, match_at)
        changes.extend(_pair_changes(item, right[match_at], left_root, right_root))
        cursor = match_at + 1
    if not any(item.kind != "inheritance_changed" for item in changes):
        appended = [item.name for item in right[cursor:]]
    else:
        for item in right[cursor:]:
            if item.slot is not None and left and item.slot > (left[-1].slot or -1):
                appended.append(item.name)
            else:
                changes.append(
                    StorageChange("added_before_existing", item.name, f"slot {item.slot}")
                )
    return changes, appended


def _same_members(left: list[StorageVariable], right: list[StorageVariable]) -> bool:
    if len(left) != len(right) or not left:
        return False
    return sorted((item.name, _type_signature(item)) for item in left) == sorted(
        (item.name, _type_signature(item)) for item in right
    )


def _find_match(item: StorageVariable, right: list[StorageVariable], start: int) -> int | None:
    for index in range(start, len(right)):
        other = right[index]
        if _type_signature(item) != _type_signature(other):
            continue
        if item.name == other.name or item.slot == other.slot:
            return index
    return None


def _find_named(item: StorageVariable, right: list[StorageVariable], start: int) -> int | None:
    if not item.name:
        return None
    for index in range(start, len(right)):
        if right[index].name == item.name:
            return index
    return None


def _append_inserts(
    changes: list[StorageChange], right: list[StorageVariable], start: int, end: int
) -> None:
    for index in range(start, end):
        item = right[index]
        changes.append(StorageChange("added_before_existing", item.name, f"slot {item.slot}"))


def _pair_changes(
    left: StorageVariable, right: StorageVariable, left_root: str, right_root: str
) -> list[StorageChange]:
    found: list[StorageChange] = []
    left_sig = _type_signature(left)
    right_sig = _type_signature(right)
    if left_sig != right_sig:
        if left.key_type or right.key_type or left.value_type or right.value_type:
            kind = "mapping_changed"
        elif left.members or right.members:
            kind = "struct_changed"
        else:
            kind = "type_changed"
        found.append(StorageChange(kind, left.name, f"slot {left.slot}->{right.slot}"))
    if left.slot != right.slot:
        found.append(StorageChange("slot_changed", left.name, f"slot {left.slot}->{right.slot}"))
    if left.offset != right.offset:
        found.append(
            StorageChange(
                "offset_changed",
                left.name,
                f"slot {left.slot} offset {left.offset}->{right.offset}",
            )
        )
        if left_sig == right_sig:
            found.append(StorageChange("packing_changed", left.name, f"slot {left.slot}"))
    elif (
        left_sig == right_sig
        and left.packing_group != right.packing_group
        and left.slot == right.slot
    ):
        found.append(StorageChange("packing_changed", left.name, f"slot {left.slot}"))
    left_inherited = left.origin != left_root
    right_inherited = right.origin != right_root
    if (
        (left_inherited or right_inherited)
        and left.origin != right.origin
        and left.name == right.name
        and left_sig == right_sig
    ):
        found.append(
            StorageChange(
                "inheritance_changed",
                left.name,
                f"{left.origin}->{right.origin} slot {left.slot}",
            )
        )
    return found


def _yul_accesses(graph: SyntaxGraph, facts: dict[str, _ContractFacts]) -> list[YulAccess]:
    del facts
    declarations = _constant_literals(graph)
    found: list[YulAccess] = []
    for event in graph.events:
        if event.kind != "sol_function":
            continue
        fields = _fields(event.extra)
        for block in _assembly_blocks(event.text):
            lets = _yul_lets(block)
            for op, args in _yul_calls(block):
                expression = (
                    args[1] if op == "delegatecall" and len(args) > 1 else (args[0] if args else "")
                )
                resolved, known, slot_literal, constant = _resolve_yul(
                    expression, lets, declarations
                )
                value_from = ""
                if op == "sstore" and len(args) > 1:
                    value_resolved, _value_known, _value_literal, _value_constant = _resolve_yul(
                        args[1], lets, declarations
                    )
                    value_from = value_resolved
                found.append(
                    YulAccess(
                        fields.get("contract", ""),
                        fields.get("function", ""),
                        op,
                        expression,
                        known,
                        slot_literal,
                        constant,
                        resolved,
                        value_from,
                    )
                )
    return found


def resolve_yul_target(
    block: str, expression: str, graph: SyntaxGraph
) -> tuple[str, bool, str, str]:
    """Follow a bounded chain of Yul ``let`` bindings. Unknown stays unknown."""
    return _resolve_yul(expression, _yul_lets(block), _constant_literals(graph))


def _constant_literals(graph: SyntaxGraph) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for event in graph.events:
        if event.kind != "sol_state":
            continue
        fields = _fields(event.extra)
        if fields.get("mutability") != "constant":
            continue
        matched = re.search(r"=\s*(0x[0-9a-fA-F]+|\d+)\s*;", event.text)
        declarations[fields.get("name", "")] = matched.group(1) if matched else ""
    return declarations


def _yul_lets(block: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in re.finditer(r"\blet\s+([A-Za-z_]\w*)\s*:=\s*([^;\n]+)", block):
        found[match.group(1)] = match.group(2).strip()
    return found


def _resolve_yul(
    expression: str, lets: dict[str, str], constants: dict[str, str], depth: int = 0
) -> tuple[str, bool, str, str]:
    expr = expression.strip()
    if depth > 4 or not expr:
        return expr, False, "", ""
    if expr in lets:
        rhs = lets[expr].strip()
        loaded = re.fullmatch(r"sload\s*\(\s*(.+)\s*\)", rhs)
        if loaded:
            inner, known, literal, constant = _resolve_yul(
                loaded.group(1), lets, constants, depth + 1
            )
            shown = f"sload({inner})"
            return shown, known, literal, constant or inner
        return _resolve_yul(rhs, lets, constants, depth + 1)
    if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", expr):
        return expr, True, expr, ""
    if expr in constants and constants[expr]:
        return expr, True, constants[expr], expr
    return expr, False, "", expr if expr in constants else ""


def _yul_calls(block: str) -> list[tuple[str, list[str]]]:
    found: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(block):
        match = re.search(r"\b(sload|sstore|delegatecall)\s*\(", block[index:])
        if match is None:
            break
        open_at = index + match.end() - 1
        try:
            end = _consume_parens(block, open_at)
        except ValueError:
            break
        inner = block[open_at + 1 : end - 1]
        args = _split_args(inner)
        found.append((match.group(1), args))
        index = end
    return found


def _split_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _assembly_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    index = 0
    while True:
        found = re.search(r"\bassembly\s*\{", text[index:])
        if not found:
            break
        open_at = index + found.end() - 1
        depth = 0
        cursor = open_at
        while cursor < len(text):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[open_at + 1 : cursor])
                    index = cursor + 1
                    break
            cursor += 1
        else:
            break
    return blocks


_ERC7201 = re.compile(
    r"keccak256\s*\(\s*abi\.encode\s*\(\s*uint256\s*\(\s*keccak256\s*\(\s*"
    r"(?P<quote>['\"])(?P<name>.*?)(?P=quote)\s*\)\s*-\s*1\s*\)\s*\)\s*\)\s*&\s*"
    r"~\s*bytes32\s*\(\s*uint256\s*\(\s*0xff\s*\)\s*\)",
    re.DOTALL,
)
_KECCAK_LITERAL = re.compile(
    r"^keccak256\s*\(\s*(?P<quote>['\"])(?P<name>.*?)(?P=quote)\s*\)$", re.DOTALL
)


def _namespaces(graph: SyntaxGraph) -> list[StorageNamespace]:
    found: list[StorageNamespace] = []
    seen: set[str] = set()
    for event in graph.events:
        if event.kind != "sol_state":
            continue
        fields = _fields(event.extra)
        if fields.get("mutability") != "constant":
            continue
        if not re.search(r"\bbytes32\b", event.text):
            continue
        name = fields.get("name", "")
        expression = _constant_expression(event.text)
        if not name or not expression or name in seen:
            continue
        if "keccak" not in expression and not re.fullmatch(r"0x[0-9a-fA-F]+|\d+", expression):
            continue
        resolved, known = _namespace_slot(expression)
        seen.add(name)
        found.append(StorageNamespace(name, expression, resolved, known, event.text.strip()[:240]))
    return found


def _constant_expression(text: str) -> str:
    matched = re.search(r"=\s*(.+?)\s*;", text, re.DOTALL)
    return re.sub(r"\s+", " ", matched.group(1)).strip() if matched else ""


def _namespace_slot(expression: str) -> tuple[str, bool]:
    compact = re.sub(r"\s+", "", expression)
    literal = re.fullmatch(r"0x[0-9a-fA-F]+|\d+", compact)
    if literal:
        return expression.strip(), True
    erc = _ERC7201.search(expression)
    if erc:
        return _erc7201_slot(erc.group("name")), True
    custom = _KECCAK_LITERAL.match(compact)
    if custom:
        digest = keccak256(custom.group("name").encode("utf-8"))
        return "0x" + digest.hex(), True
    if "keccak256" in compact or "keccak" in compact:
        return "", False
    return "", False


def _alias_type(type_name: str) -> str:
    if type_name == "uint":
        return "uint256"
    if type_name == "int":
        return "int256"
    if type_name == "byte":
        return "bytes1"
    return type_name


def _erc7201_slot(namespace: str) -> str:
    from app.parsing.keccak import keccak256

    inner = int.from_bytes(keccak256(namespace.encode("utf-8")), "big") - 1
    outer = keccak256(inner.to_bytes(32, "big"))
    masked = int.from_bytes(outer, "big") & ~0xFF
    return "0x" + masked.to_bytes(32, "big").hex()


def _compiler_index(
    compiler: CompilerSemantics, source_path: str
) -> tuple[dict[tuple[str, str], list[dict[str, str]]], list[dict[str, str]]]:
    del source_path
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    unlabeled: list[dict[str, str]] = []
    entries = list(compiler.layouts)
    if not entries:
        entries = [
            {"label": item.get("label", ""), "slot": item.get("slot", "")}
            for item in compiler.storage
        ]
    for item in entries:
        label = str(item.get("label") or "")
        contract = str(item.get("contract") or "")
        if not label:
            continue
        if not contract:
            unlabeled.append(item)
            continue
        grouped.setdefault((contract, label), []).append(item)
    return grouped, unlabeled


def _record_compiler_difference(
    model: StorageModel, variable: StorageVariable, item: dict[str, str]
) -> None:
    compiler_slot = str(item.get("slot", ""))
    if str(variable.slot) != compiler_slot:
        model.disagreements.append(
            LayoutDisagreement(
                variable.contract,
                variable.name,
                str(variable.slot),
                compiler_slot,
                "slot",
                str(variable.slot),
                compiler_slot,
            )
        )
    if item.get("offset") not in {None, ""} and variable.offset is not None:
        if str(variable.offset) != str(item.get("offset")):
            model.disagreements.append(
                LayoutDisagreement(
                    variable.contract,
                    variable.name,
                    str(variable.slot),
                    compiler_slot,
                    "offset",
                    str(variable.offset),
                    str(item.get("offset")),
                )
            )
    compiler_type = str(item.get("type") or "")
    if compiler_type.startswith("t_"):
        compiler_type = ""
    parser_type = re.sub(r"\s+", "", variable.type_name)
    if compiler_type and compiler_type not in {parser_type, _alias_type(parser_type)}:
        model.disagreements.append(
            LayoutDisagreement(
                variable.contract,
                variable.name,
                str(variable.slot),
                compiler_slot,
                "type",
                variable.type_name,
                compiler_type,
            )
        )


def _abi_members(abi: str) -> tuple[str, ...]:
    body = abi.strip()
    if not (body.startswith("(") and body.endswith(")")):
        return ()
    parts = [item.strip() for item in body[1:-1].split(",") if item.strip()]
    return tuple(parts)


def _visibility(text: str) -> str:
    for word in ("public", "private", "internal"):
        if re.search(rf"\b{word}\b", text):
            return word
    return "internal"


def _fields(extra: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in extra.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields
