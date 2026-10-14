"""Delegatecall, upgrade, and proxy-shape evidence.

These labels are structural. A contract that resembles UUPS, a transparent
proxy, a beacon, or a diamond is not certified as that standard. Authorization
comes from a resolved modifier or a check that dominates the write. A modifier
that only contains ``_;`` is not authorization, and a function name is not
authorization either.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from app.parsing.model import SyntaxEvent, SyntaxGraph
from app.parsing.solidity_cfg import _consume_parens, operation_guarded, placeholder_is_guarded
from app.parsing.solidity_modifiers import resolve_modifier
from app.parsing.solidity_storage import StorageModel, analyze_storage, resolve_yul_target

_EIP1967_IMPL = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
_IMPL_NAMES = {"implementation", "impl"}
_PROXY_CACHE: ContextVar[dict[int, ProxyModel] | None] = ContextVar(
    "bugforge_proxy_cache", default=None
)


@dataclass(frozen=True)
class DelegateSite:
    contract: str
    function: str
    target: str
    provenance: str
    caller_controlled: bool
    upgrade_controlled: bool
    forwards_data: bool
    text: str


@dataclass(frozen=True)
class PatternEvidence:
    kind: str
    summary: str
    confidence: str
    related: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpgradeSite:
    contract: str
    function: str
    writes: tuple[str, ...]
    authorized: bool
    invokes_call: bool
    summary: str


@dataclass
class ProxyModel:
    delegates: list[DelegateSite] = field(default_factory=list)
    upgrades: list[UpgradeSite] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    pattern_evidence: list[PatternEvidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    storage: StorageModel | None = None


def set_proxy_context() -> Token[dict[int, ProxyModel] | None]:
    return _PROXY_CACHE.set({})


def reset_proxy_context(token: Token[dict[int, ProxyModel] | None]) -> None:
    _PROXY_CACHE.reset(token)


def proxy_context_active() -> bool:
    return _PROXY_CACHE.get() is not None


def analyze_proxy(graph: SyntaxGraph) -> ProxyModel:
    cache = _PROXY_CACHE.get()
    if cache is not None and id(graph) in cache:
        return cache[id(graph)]
    model = _analyze_proxy(graph)
    if cache is not None:
        cache[id(graph)] = model
    return model


def _analyze_proxy(graph: SyntaxGraph) -> ProxyModel:
    model = ProxyModel(storage=analyze_storage(graph))
    if graph.language != "solidity" or graph.parser_tier.value == "profile_fallback":
        return model
    state = _state(graph)
    model.delegates = _delegates(graph, state)
    model.upgrades = _upgrades(graph, state, model)
    model.patterns = _patterns(graph, model)
    model.notes = _notes(graph, model)
    return model


def _delegates(graph: SyntaxGraph, state: dict[tuple[str, str], str]) -> list[DelegateSite]:
    found: list[DelegateSite] = []
    for event in graph.events:
        if event.kind != "sol_delegatecall":
            continue
        fields = _fields(event.extra)
        target = fields.get("target", "")
        site = _classify_target(
            graph,
            fields.get("contract", ""),
            fields.get("function", ""),
            target,
            event.text,
            state,
        )
        found.append(site)
    for event in graph.events:
        if event.kind != "sol_function":
            continue
        fields = _fields(event.extra)
        contract = fields.get("contract", "")
        if re.search(r"address\s*\(\s*this\s*\)\s*\.\s*delegatecall", event.text):
            found.append(
                DelegateSite(
                    contract,
                    fields.get("function", ""),
                    "address(this)",
                    "self",
                    False,
                    False,
                    "msg.data" in event.text,
                    "address(this).delegatecall",
                )
            )
        for block in _assembly_blocks(event.text):
            for args in _delegatecall_args(block):
                target = args[1] if len(args) > 1 else ""
                if not target:
                    found.append(
                        DelegateSite(
                            contract,
                            fields.get("function", ""),
                            "",
                            "unknown",
                            False,
                            False,
                            False,
                            "delegatecall",
                        )
                    )
                    continue
                resolved, known, _literal, _constant = resolve_yul_target(block, target, graph)
                if known and resolved != target:
                    found.append(
                        DelegateSite(
                            contract,
                            fields.get("function", ""),
                            f"{target}<-{resolved}",
                            "storage_slot",
                            False,
                            False,
                            False,
                            "delegatecall(" + target + ")",
                        )
                    )
                    continue
                found.append(
                    _classify_target(
                        graph,
                        contract,
                        fields.get("function", ""),
                        target,
                        "delegatecall(" + target + ")",
                        state,
                    )
                )
    return found


def _delegatecall_args(block: str) -> list[list[str]]:
    found: list[list[str]] = []
    index = 0
    while index < len(block):
        match = re.search(r"\bdelegatecall\s*\(", block[index:])
        if match is None:
            break
        open_at = index + match.end() - 1
        try:
            end = _consume_parens(block, open_at)
        except ValueError:
            break
        found.append(_split_top(block[open_at + 1 : end - 1]))
        index = end
    return found


def _split_top(text: str) -> list[str]:
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


def _classify_target(
    graph: SyntaxGraph,
    contract: str,
    function: str,
    target: str,
    text: str,
    state: dict[tuple[str, str], str],
) -> DelegateSite:
    base = target.split(".", 1)[0].strip()
    compact = re.sub(r"\s+", "", target)
    forwards = "msg.data" in text or "calldata" in text
    if "address(this)" in compact or base == "address(this)":
        return DelegateSite(contract, function, target, "self", False, False, forwards, text)
    mutability = state.get((contract, base), "")
    if mutability in {"immutable", "constant"}:
        return DelegateSite(contract, function, target, mutability, False, False, forwards, text)
    if mutability == "storage":
        upgrade_controlled = _writes_are_authorized(graph, contract, base)
        return DelegateSite(
            contract,
            function,
            target,
            "state",
            False,
            upgrade_controlled,
            forwards,
            text,
        )
    if _is_parameter(graph, contract, function, base):
        return DelegateSite(contract, function, target, "parameter", True, False, forwards, text)
    if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", compact):
        return DelegateSite(contract, function, target, "constant", False, False, forwards, text)
    return DelegateSite(contract, function, target, "unknown", False, False, forwards, text)


def _upgrades(
    graph: SyntaxGraph, state: dict[tuple[str, str], str], model: ProxyModel
) -> list[UpgradeSite]:
    targets = {
        site.target.split(".", 1)[0].strip()
        for site in model.delegates
        if site.provenance == "state" and site.contract
    }
    impl_slots = {
        access.constant or access.literal
        for access in (model.storage.yul if model.storage else [])
        if access.op == "sload" and access.known
    }
    found: list[UpgradeSite] = []
    for event in _functions(graph):
        fields = _fields(event.extra)
        if fields.get("visibility") not in {"public", "external"}:
            continue
        if fields.get("mutability") in {"view", "pure"}:
            continue
        name = _function_name(event)
        contract = fields.get("contract", "")
        writes = _written_names(graph, event)
        semantic = [item for item in writes if item in targets or item.lower() in _IMPL_NAMES]
        slot_write = _sstore_known_impl(event.text, impl_slots, graph)
        beacon_or_facet = [
            item
            for item in writes
            if item.lower() in {"beacon", "facet"}
            or item.lower().endswith("facet")
            or _writes_selector_map(graph, contract, item)
        ]
        if model.delegates and beacon_or_facet and not semantic:
            semantic = beacon_or_facet
        if not semantic and not slot_write:
            continue
        changed = tuple(semantic or ([slot_write] if slot_write else writes[:1]))
        operation = changed[0] if changed else name
        authorized = _function_authorized(graph, event, operation)
        invokes = bool(re.search(r"\.call\s*\(|\binitialize\s*\(", event.text))
        reason = (
            "The upgrade write is dominated by an authorization check."
            if authorized
            else "The implementation can be replaced without an authorization check that dominates the write."
        )
        found.append(
            UpgradeSite(
                contract,
                name,
                changed,
                authorized,
                invokes,
                reason + " This is potential evidence.",
            )
        )
    del state
    return found


def _patterns(graph: SyntaxGraph, model: ProxyModel) -> list[str]:
    labels: list[str] = []
    evidence: list[PatternEvidence] = []
    if any(
        site.function in {"fallback", "receive", ""} or "fallback" in site.function
        for site in model.delegates
    ):
        labels.append("fallback-proxy")
    elif model.delegates:
        labels.append("delegatecall-proxy")
    uups = _uups_evidence(graph, model)
    beacon = _beacon_evidence(graph, model)
    transparent = _transparent_evidence(graph, model)
    diamond = _diamond_evidence(graph, model)
    for item in (uups, beacon, transparent, diamond):
        if item is None:
            continue
        labels.append(item.kind)
        evidence.append(item)
    model.pattern_evidence = evidence
    return labels


def _notes(graph: SyntaxGraph, model: ProxyModel) -> list[str]:
    notes: list[str] = []
    if model.storage and model.storage.overlaps:
        notes.extend(model.storage.overlaps)
    if model.storage and model.storage.disagreements:
        for item in model.storage.disagreements:
            notes.append(
                f"Parser slot {item.parser_slot} for `{item.contract}.{item.name}` disagrees "
                f"with compiler slot {item.compiler_slot}. Both are kept."
            )
    for event in _functions(graph):
        fields = _fields(event.extra)
        if _function_name(event) not in {"initialize", "reinitialize"}:
            continue
        if fields.get("visibility") not in {"public", "external"}:
            continue
        notes.append(
            f"`{fields.get('contract', '')}.{_function_name(event)}` is externally reachable. "
            "Whether it can be repeated depends on the initializer check, not on the function name."
        )
    if model.storage:
        for comparison in model.storage.comparisons:
            if not comparison.appended:
                continue
            for event in _functions(graph):
                if _function_name(event) not in {"initialize", "reinitialize"}:
                    continue
                owner = _fields(event.extra).get("contract", "")
                if owner not in {comparison.left_contract, comparison.right_contract}:
                    continue
                written = set(_written_names(graph, event))
                owned = {
                    item.name
                    for item in model.storage.variables
                    if item.contract == owner or item.origin == owner
                }
                missing = [
                    name for name in comparison.appended if name in owned and name not in written
                ]
                if not missing:
                    continue
                notes.append(
                    f"`{_function_name(event)}` does not write newly appended storage "
                    f"{', '.join(missing)}. This is potential evidence that new state is "
                    "uninitialized, not proof that the initializer is unprotected."
                )
    for site in model.upgrades:
        if site.invokes_call:
            notes.append(
                f"`{site.contract}.{site.function}` can call out while replacing an implementation. "
                "That may run initialization in the new context. This is not a proof that it does."
            )
    detail = model.storage.yul_detail if model.storage is not None else None
    known_impl = detail is not None and any(
        item.slot.role == "implementation" and item.slot.slot_class == "known"
        for item in detail.loads
    )
    if (
        model.storage
        and model.storage.compiler_status == "AVAILABLE"
        and model.storage.compiler_matches > 0
        and not model.storage.disagreements
        and not model.storage.semantic_disagreements
        and known_impl
    ):
        notes.append(
            "Compiler layout does not disagree with the parser. That can strengthen "
            "confidence in a known implementation slot. It does not verify the proxy."
        )
    for access in model.storage.yul if model.storage else []:
        if not access.known:
            notes.append(
                f"Yul {access.op} in `{access.contract}.{access.function}` uses `{access.expression}`, "
                "and that slot is not a literal or a known constant."
            )
        elif access.literal.lower() == _EIP1967_IMPL:
            notes.append(
                f"Yul {access.op} in `{access.contract}.{access.function}` uses the EIP-1967 "
                "implementation slot constant. The constant does not by itself authorize an upgrade."
            )
    return notes


def _writes_are_authorized(graph: SyntaxGraph, contract: str, variable: str) -> bool:
    writers = [
        event
        for event in _functions(graph)
        if _fields(event.extra).get("contract") == contract
        and variable in _written_names(graph, event)
    ]
    if not writers:
        return False
    return all(_function_authorized(graph, event, variable) for event in writers)


def _function_authorized(graph: SyntaxGraph, function: SyntaxEvent, operation: str) -> bool:
    text = function.text
    contract = _fields(function.extra).get("contract", "")
    injections: list[str] = []
    for name in _modifier_names(_fields(function.extra).get("modifiers", "")):
        resolution = resolve_modifier(graph, contract, name)
        if resolution.status == "resolved" and placeholder_is_guarded(resolution.body):
            injections.append("require(msg.sender == owner);")
    if injections:
        brace = text.find("{")
        if brace >= 0:
            text = text[: brace + 1] + " ".join(injections) + text[brace + 1 :]
    if not operation:
        return False
    return operation_guarded(text, operation[:120]) is True


def _sstore_known_impl(text: str, known_slots: set[str], graph: SyntaxGraph) -> str:
    for block in _assembly_blocks(text):
        for match in re.finditer(r"\bsstore\s*\(\s*([^,\)]+)", block):
            expression = match.group(1).strip()
            resolved, known, literal, constant = resolve_yul_target(block, expression, graph)
            if (
                expression in known_slots
                or expression.lower() == _EIP1967_IMPL
                or literal.lower() == _EIP1967_IMPL
                or (known and (constant in known_slots or literal in known_slots))
            ):
                return resolved or expression
    return ""


def _writes_selector_map(graph: SyntaxGraph, contract: str, variable: str) -> bool:
    for event in graph.events:
        if event.kind != "sol_state":
            continue
        fields = _fields(event.extra)
        if fields.get("contract") != contract or fields.get("name") != variable:
            continue
        return "bytes4" in fields.get("type", "") and "address" in fields.get("type", "")
    return False


def _uups_evidence(graph: SyntaxGraph, model: ProxyModel) -> PatternEvidence | None:
    uuid = next(
        (event for event in _functions(graph) if _function_name(event) == "proxiableUUID"),
        None,
    )
    if uuid is None:
        return None
    related: list[str] = ["proxiableUUID"]
    slot_linked = _EIP1967_IMPL in re.sub(r"\s+", "", uuid.text).lower() or any(
        access.known and access.literal.lower() == _EIP1967_IMPL
        for access in (model.storage.yul if model.storage else [])
    )
    if slot_linked:
        related.append("implementation-slot")
    upgrade_linked = any(
        any(item.lower() in _IMPL_NAMES or item.lower().startswith("0x") for item in site.writes)
        for site in model.upgrades
    )
    if upgrade_linked:
        related.append("upgrade")
    delegate_linked = any(site.provenance in {"state", "storage_slot"} for site in model.delegates)
    if delegate_linked:
        related.append("delegatecall")
    if len(related) < 2:
        return None
    return PatternEvidence(
        "uups-like",
        "proxiableUUID is structurally associated with an implementation slot or upgrade. "
        "This is not UUPS compliance and it is not a vulnerability by itself.",
        "structural" if slot_linked else "usage",
        tuple(related),
    )


def _beacon_evidence(graph: SyntaxGraph, model: ProxyModel) -> PatternEvidence | None:
    if not model.delegates:
        return None
    beacon_names = {
        _fields(event.extra).get("name", "")
        for event in graph.events
        if event.kind == "sol_state"
        and (
            _fields(event.extra).get("name", "").lower() == "beacon"
            or "beacon" in _fields(event.extra).get("type", "").lower()
        )
    }
    beacon_names.discard("")
    if not beacon_names:
        return None
    texts = "\n".join(event.text for event in _functions(graph))
    implementation_read = any(
        re.search(rf"\b{re.escape(name)}\b\s*\.\s*implementation\s*\(", texts)
        or re.search(rf"Beacon\s*\(\s*{re.escape(name)}\s*\)\s*\.\s*implementation\s*\(", texts)
        for name in beacon_names
    )
    if not implementation_read:
        return None
    return PatternEvidence(
        "beacon-like",
        "A beacon address is read for an implementation before delegatecall. "
        "This is a structural chain, not a beacon-proxy certificate.",
        "structural",
        ("proxy", "beacon", "implementation"),
    )


def _transparent_evidence(graph: SyntaxGraph, model: ProxyModel) -> PatternEvidence | None:
    fallback = [
        site
        for site in model.delegates
        if site.function in {"fallback", "receive", ""} or "fallback" in site.function
    ]
    if not fallback:
        return None
    admin_check = False
    for event in _functions(graph):
        if re.search(r"msg\.sender\s*(==|!=)\s*\w*admin\w*", event.text):
            admin_check = True
    if not admin_check:
        return None
    return PatternEvidence(
        "transparent-like",
        "An admin sender check coexists with a fallback delegatecall. "
        "The word admin alone does not make the proxy transparent.",
        "structural",
        ("admin", "fallback", "delegatecall"),
    )


def _diamond_evidence(graph: SyntaxGraph, model: ProxyModel) -> PatternEvidence | None:
    if not model.delegates:
        return None
    selector_maps = [
        _fields(event.extra).get("name", "")
        for event in graph.events
        if event.kind == "sol_state"
        and "bytes4" in _fields(event.extra).get("type", "")
        and "address" in _fields(event.extra).get("type", "")
    ]
    if not selector_maps:
        return None
    texts = "\n".join(event.text for event in _functions(graph))
    dispatch = any(name and name in texts and "delegatecall" in texts for name in selector_maps)
    cut = any(
        _function_name(event) in {"diamondCut", "updateFacet"}
        or any(name in _written_names(graph, event) for name in selector_maps)
        for event in _functions(graph)
    )
    if not dispatch or not cut:
        return None
    return PatternEvidence(
        "diamond-like",
        "A selector-to-address mapping is written and used to choose a delegatecall target. "
        "Unknown selectors stay unknown. This is not a diamond certificate.",
        "structural",
        tuple(selector_maps[:4]) + ("delegatecall",),
    )


def _is_parameter(graph: SyntaxGraph, contract: str, function: str, name: str) -> bool:
    for event in _functions(graph):
        fields = _fields(event.extra)
        if fields.get("contract") != contract or fields.get("function") != function:
            continue
        params = _fields(event.extra).get("params", "") + event.text.split("{", 1)[0]
        return bool(re.search(rf"\b{re.escape(name)}\b", params))
    return False


def _written_names(graph: SyntaxGraph, function: SyntaxEvent) -> list[str]:
    span = function.span
    if span is None:
        return []
    names: list[str] = []
    for event in graph.events:
        if event.kind != "sol_state_write" or event.span is None:
            continue
        if span.start_byte <= event.span.start_byte < span.end_byte:
            name = _fields(event.extra).get("name", "")
            if name and name not in names:
                names.append(name)
    return names


def _state(graph: SyntaxGraph) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for event in graph.events:
        if event.kind != "sol_state":
            continue
        fields = _fields(event.extra)
        found[(fields.get("contract", ""), fields.get("name", ""))] = fields.get("mutability", "")
    return found


def _functions(graph: SyntaxGraph) -> list[SyntaxEvent]:
    return [event for event in graph.events if event.kind == "sol_function"]


def _function_name(event: SyntaxEvent) -> str:
    match = re.search(r"function\s+([A-Za-z_]\w*)", event.text)
    if match:
        return match.group(1)
    if re.search(r"\bfallback\b", event.text):
        return "fallback"
    if re.search(r"\breceive\b", event.text):
        return "receive"
    return ""


def _modifier_names(modifiers: str) -> list[str]:
    names: list[str] = []
    for item in modifiers.split(","):
        token = re.split(r"[\(\s]", item.strip(), maxsplit=1)[0]
        if token:
            names.append(token)
    return names


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


def _fields(extra: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in extra.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields
