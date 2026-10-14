"""Highest-value bounty detectors (Phase 51, owner Phase 6).

Three static, evidence-based detectors that are not covered by the existing Phase 50
families:

* read-only reentrancy: a state-changing function makes an external call before it
  updates a state variable (a check-effects-interactions violation) and an exposed
  ``view`` reads that same variable without the writer's reentrancy guard, so the view
  can be observed mid-update;
* transient-storage misuse: a function writes EIP-1153 transient storage (``tstore`` or
  a ``transient`` variable) and never clears it, so a later call in the same transaction
  observes a stale value;
* EIP-7702 EOA assumption: a security gate assumes an address is a plain EOA
  (``msg.sender == tx.origin``, ``code.length == 0``, ``extcodesize == 0``,
  ``!isContract(...)``), which EIP-7702 makes unsound because an EOA can carry code.

Every result is a static candidate. It is evidence-based (the construct must actually be
present in a guard or a write), never a keyword hit, and never verification.

Phase 52 precision (Slice D):

* read-only reentrancy follows one level of internal helpers to find the external
  call (``_send`` doing ``call{value:}``), and a view that checks the reentrancy lock
  (``_reentrancyGuardEntered()``, ``nonReentrantView``, a lock flag) is not exposed;
* transient storage is judged per *entry point*, with the writes and clears of the
  modifiers it uses and the internal helpers it calls (a ``_lock``/``_unlock`` pair is
  not "left set"); a computed slot whose clear cannot be matched is not claimed;
* EIP-7702 assumptions count only inside a real gate (``require``/``assert`` or an
  ``if`` that reverts/returns), in function bodies *and* modifiers, and only when the
  subject is the caller (``msg.sender``/``tx.origin``/``caller()``/``_msgSender()``) or
  a parameter; a factory's ``predicted.code.length == 0`` deploy check is not a gate on
  an actor and is not reported.
"""

from __future__ import annotations

import re

from app.parsing.solidity_research import (
    MAX_CANDIDATES_PER_FAMILY,
    ResearchModel,
    RFunction,
    SemanticCandidate,
    line_of_text,
    match_close,
    member_calls,
    skeleton,
)

FAMILY = "high_value"

_STATE_WRITE = r"\b{name}\b\s*(?:\[[^\]]*\]\s*)*(?:=(?!=)|\+=|-=|\*=|/=)"
_REENTRANCY_GUARD = re.compile(r"nonReentrant|noReentrancy|reentrancyGuard", re.I)
_EXTERNAL_CALL_NAMES = frozenset({"call", "delegatecall", "send", "transfer"})

# EIP-1153 transient storage writes.
_TSTORE = re.compile(r"\btstore\s*\(\s*([^,]+),")
_TLOAD = re.compile(r"\btload\s*\(")
_TRANSIENT_DECL = re.compile(r"\btransient\b[^;=]*\b(\w+)\s*(?:=|;)")


def analyze_high_value(model: ResearchModel) -> list[SemanticCandidate]:
    out: list[SemanticCandidate] = []
    out.extend(_read_only_reentrancy(model))
    out.extend(_transient_misuse(model))
    out.extend(_eip7702_assumptions(model))
    return out[:MAX_CANDIDATES_PER_FAMILY]


_INTERNAL_CALL = re.compile(r"(?<![.\w])([A-Za-z_]\w*)\s*\(")
_NOT_CALLS = frozenset(
    {
        "if",
        "for",
        "while",
        "require",
        "assert",
        "revert",
        "return",
        "emit",
        "new",
        "keccak256",
        "abi",
        "address",
        "uint256",
        "payable",
        "type",
        "bytes",
        "string",
        "mapping",
    }
)


def _helpers(model: ResearchModel, contract: str, body: str) -> list[tuple[int, RFunction]]:
    """Internal/private functions of the contract called from ``body`` (one level)."""
    found: list[tuple[int, RFunction]] = []
    skel = skeleton(body)
    for match in _INTERNAL_CALL.finditer(skel):
        name = match.group(1)
        if name in _NOT_CALLS:
            continue
        for item in model.functions_named(contract, name):
            if item.has_body and item.visibility in {"internal", "private"}:
                found.append((match.start(), item))
                break
    return found


# ---- read-only reentrancy -------------------------------------------------------------------

_VIEW_LOCK_CHECK = re.compile(
    r"_reentrancyGuardEntered\s*\(|reentrancyGuardEntered\s*\(|ReentrancyGuardReentrantCall"
    r"|_status\s*==|_status\s*!=|require\s*\(\s*!\s*_?(?:locked|entered|lock)\b"
    r"|if\s*\(\s*_?(?:locked|entered|lock)\s*\)\s*revert"
)


def _read_only_reentrancy(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        state = set(model.state_vars(name))
        if not state:
            continue
        functions = [f for f in model.functions_of(name) if f.has_body]
        # variables written after an external call in a reentrancy-prone function
        prone: dict[str, tuple[RFunction, str]] = {}
        for function in functions:
            if function.mutability in {"view", "pure"}:
                continue
            via, written = _written_after_external_call(model, name, function, state)
            for var in written:
                prone.setdefault(var, (function, via))
        if not prone:
            continue
        for function in functions:
            if function.mutability != "view" or not function.exposed:
                continue
            if _has_guard(function) or _VIEW_LOCK_CHECK.search(skeleton(function.body)):
                continue  # the view refuses to answer while the lock is held
            read = _reads_vars(function, set(prone))
            for var in sorted(read):
                writer, via = prone[var]
                found.append(
                    SemanticCandidate(
                        detector="read_only_reentrancy.view_observes_mid_update",
                        family=FAMILY,
                        title="read-only reentrancy exposure",
                        summary=(
                            f"view {function.name} reads {var}, which {writer.name} updates after "
                            "an external call; the value can be observed mid-update"
                        ),
                        file=function.file,
                        line=function.line,
                        contract=name,
                        function=function.signature,
                        facts=(
                            ("state_variable", var),
                            ("writer", writer.signature),
                            ("writer_guarded", str(_has_guard(writer)).lower()),
                            ("external_call_via", via),
                        ),
                        observed=("external_call_before_state_write", "unguarded_view_read"),
                        missing=("confirm the view is consumed by an external integrator",),
                        confidence="low",
                        impact_tags=("read_only_reentrancy",),
                    )
                )
                if len(found) >= MAX_CANDIDATES_PER_FAMILY:
                    return found
    return found


def _external_offsets(body: str) -> list[int]:
    return [
        call.start
        for call in member_calls(body)
        if "value" in call.options or call.name in _EXTERNAL_CALL_NAMES
    ]


def _written_after_external_call(
    model: ResearchModel, contract: str, function: RFunction, state: set[str]
) -> tuple[str, set[str]]:
    offsets = [(offset, "direct") for offset in _external_offsets(function.body)]
    for offset, helper in _helpers(model, contract, function.body):
        if _external_offsets(helper.body):
            offsets.append((offset, f"helper:{helper.name}"))
    if not offsets:
        return "", set()
    first, via = min(offsets)
    tail = function.body[first:]
    written: set[str] = set()
    for var in state:
        if re.search(_STATE_WRITE.format(name=re.escape(var)), tail):
            written.add(var)
    return via, written


def _reads_vars(function: RFunction, variables: set[str]) -> set[str]:
    read: set[str] = set()
    for var in variables:
        if re.search(rf"\b{re.escape(var)}\b", function.body):
            read.add(var)
    return read


def _has_guard(function: RFunction) -> bool:
    return any(_REENTRANCY_GUARD.search(mod) for mod in function.modifiers)


# ---- transient-storage misuse ---------------------------------------------------------------

_TSTORE_CLEAR = re.compile(r"\btstore\s*\(\s*([^,]+),\s*0\s*\)")
_LITERAL_SLOT = re.compile(r"^(?:0x[0-9a-fA-F]+|\d+|[A-Z_][A-Z0-9_]*|[A-Za-z_]\w*\.slot)$")
_ZERO = r"(?:0|false|address\(0\)|bytes32\(0\)|0x0+)"


def _norm(slot: str) -> str:
    return re.sub(r"\s+", "", slot)


def _scope_bodies(
    model: ResearchModel, contract: str, function: RFunction
) -> list[tuple[str, str]]:
    """(label, body) for the function, the modifiers it uses and helpers it calls."""
    bodies = [("direct", function.body)]
    for name in function.modifiers:
        bare = name.split("(")[0].strip()
        modifier = model.modifier(contract, bare)
        if modifier is not None:
            bodies.append((f"modifier:{bare}", modifier.body))
    for _offset, helper in _helpers(model, contract, function.body):
        bodies.append((f"helper:{helper.name}", helper.body))
    return bodies


def _transient_misuse(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        transient_vars = set(_TRANSIENT_DECL.findall(contract.body))
        for function in model.functions_of(name, inherited=False):
            if not function.has_body or not function.exposed:
                continue  # judged per entry point, with its modifiers and helpers
            bodies = _scope_bodies(model, name, function)
            sets: dict[str, str] = {}
            cleared: set[str] = set()
            assigned: dict[str, str] = {}
            var_cleared: set[str] = set()
            for label, text in bodies:
                for slot in _TSTORE.findall(text):
                    sets.setdefault(_norm(slot), label)
                cleared |= {_norm(slot) for slot in _TSTORE_CLEAR.findall(text)}
                for var in transient_vars:
                    if re.search(rf"\b{re.escape(var)}\b\s*(?:=(?!=)|\+=|-=)", text):
                        if re.search(rf"\b{re.escape(var)}\s*=\s*{_ZERO}\s*;", text) or re.search(
                            rf"\bdelete\s+{re.escape(var)}\b", text
                        ):
                            var_cleared.add(var)
                        assigned.setdefault(var, label)
            uncleared = {slot for slot in sets if slot not in cleared}
            # a computed slot with a computed clear elsewhere cannot be matched: no claim
            if uncleared and any(not _LITERAL_SLOT.match(slot) for slot in uncleared):
                if any(not _LITERAL_SLOT.match(slot) for slot in cleared):
                    uncleared = {slot for slot in uncleared if _LITERAL_SLOT.match(slot)}
            # a write of zero is itself the clear
            uncleared = {slot for slot in uncleared if not re.fullmatch(_ZERO, slot)}
            left = {var: label for var, label in assigned.items() if var not in var_cleared}
            if not uncleared and not left:
                continue
            detail = []
            if uncleared:
                detail.append(f"tstore slots {sorted(uncleared)} are not cleared")
            if left:
                detail.append(f"transient vars {sorted(left)} are not cleared")
            via = sorted({sets[slot] for slot in uncleared} | set(left.values()))
            found.append(
                SemanticCandidate(
                    detector="transient_storage.not_cleared",
                    family=FAMILY,
                    title="transient storage left set",
                    summary=(
                        f"{function.name} writes transient storage without clearing it; "
                        + "; ".join(detail)
                        + "; a later call in the same transaction may read a stale value"
                    ),
                    file=function.file,
                    line=line_of_text(function, "tstore") if uncleared else function.line,
                    contract=name,
                    function=function.signature,
                    facts=(
                        ("uncleared_slots", ",".join(sorted(uncleared))),
                        ("uncleared_vars", ",".join(sorted(left))),
                        ("written_via", ",".join(via)),
                    ),
                    observed=("transient_write_without_clear",),
                    missing=("confirm no later clear on every path",),
                    confidence="low",
                    impact_tags=("transient_storage",),
                )
            )
            if len(found) >= MAX_CANDIDATES_PER_FAMILY:
                return found
    return found


# ---- EIP-7702 EOA assumptions ---------------------------------------------------------------

_CALLER_SUBJECTS = frozenset({"msg.sender", "tx.origin", "caller()", "_msgSender()", "origin()"})
_SUBJECT = r"([A-Za-z_][\w.]*(?:\(\))?)"
# (regex, label, polarity). "eoa" forms are true for an EOA and gate in require/assert;
# "contract" forms are true for a contract and gate in an `if (...) revert`.
_GATE_FORMS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"msg\.sender\s*==\s*tx\.origin|tx\.origin\s*==\s*msg\.sender"),
        "msg.sender == tx.origin",
        "eoa",
    ),
    (
        re.compile(r"msg\.sender\s*!=\s*tx\.origin|tx\.origin\s*!=\s*msg\.sender"),
        "msg.sender != tx.origin",
        "contract",
    ),
    (
        re.compile(_SUBJECT + r"\.code\.length\s*(?:==\s*0|<\s*1)"),
        "address.code.length == 0",
        "eoa",
    ),
    (
        re.compile(_SUBJECT + r"\.code\.length\s*(?:!=\s*0|>\s*0)"),
        "address.code.length > 0",
        "contract",
    ),
    (
        re.compile(r"extcodesize\s*\(\s*" + _SUBJECT + r"\s*\)\s*==\s*0"),
        "extcodesize(a) == 0",
        "eoa",
    ),
    (
        re.compile(r"extcodesize\s*\(\s*" + _SUBJECT + r"\s*\)\s*(?:!=|>)\s*0"),
        "extcodesize(a) > 0",
        "contract",
    ),
    (re.compile(r"!\s*\w*\.?isContract\s*\(\s*" + _SUBJECT + r"\s*\)"), "!isContract(a)", "eoa"),
    (
        re.compile(r"(?<![!\w.])\w*\.?isContract\s*\(\s*" + _SUBJECT + r"\s*\)"),
        "isContract(a)",
        "contract",
    ),
)


def _gate_kind(skel: str, index: int) -> str:
    """'require' | 'if_revert' | 'if_other' | '' for the construct enclosing ``index``."""
    depth = 0
    i = index - 1
    while i >= 0:
        char = skel[i]
        if char == ")":
            depth += 1
        elif char == "(":
            if depth == 0:
                head = skel[max(0, i - 12) : i].rstrip()
                if re.search(r"\b(require|assert)$", head):
                    return "require"
                if re.search(r"\bif$", head):
                    close = match_close(skel, i)
                    if close == -1:
                        return ""
                    start = close + 1
                    while start < len(skel) and skel[start].isspace():
                        start += 1
                    if start < len(skel) and skel[start] == "{":
                        end = match_close(skel, start)
                        statement = skel[start : end + 1 if end != -1 else start + 400][:400]
                    else:
                        stop = skel.find(";", start)
                        statement = skel[start : stop + 1 if stop != -1 else start + 200]
                    if re.search(r"\b(revert|return)\b|require\s*\(\s*false", statement):
                        return "if_revert"
                    return "if_other"
                # nested call/grouping: keep walking outwards
            else:
                depth -= 1
        elif char in ";{}" and depth == 0:
            return ""
        i -= 1
    return ""


def _subject_kind(subject: str, params: set[str]) -> str:
    subject = subject.strip()
    if subject in _CALLER_SUBJECTS:
        return "caller"
    root = subject.split(".")[0].split("(")[0]
    if root in params:
        return "parameter"
    return ""


def _eoa_gate(text: str, params: set[str]) -> tuple[str, str, str] | None:
    """(label, subject_kind, gate) for the first EOA assumption used as a gate."""
    skel = skeleton(text)
    for pattern, label, polarity in _GATE_FORMS:
        for match in pattern.finditer(skel):
            gate = _gate_kind(skel, match.start())
            wanted = {"eoa": {"require"}, "contract": {"if_revert"}}[polarity]
            if gate not in wanted:
                continue
            subject = match.group(1) if match.groups() else "msg.sender"
            kind = _subject_kind(subject, params)
            if not kind:
                continue  # a computed/local address (e.g. a predicted deployment): not an actor
            return label, kind, gate
    return None


def _eip7702_assumptions(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        modifier_gates: dict[str, tuple[str, str, str]] = {}
        for owner in model.lineage(name):
            item = model.contracts.get(owner)
            for modifier in item.modifiers if item is not None else ():
                hit = _eoa_gate(modifier.body, {p.name for p in modifier.params})
                if hit is not None:
                    modifier_gates.setdefault(modifier.name, hit)
        for function in model.functions_of(name, inherited=False):
            if not function.has_body:
                continue
            params = {p.name for p in function.params if p.name}
            hit = _eoa_gate(function.body, params)
            via = "body"
            if hit is None:
                for mod in function.modifiers:
                    bare = mod.split("(")[0].strip()
                    if bare in modifier_gates:
                        hit, via = modifier_gates[bare], f"modifier:{bare}"
                        break
            if hit is None:
                continue
            label, subject_kind, gate = hit
            found.append(
                SemanticCandidate(
                    detector="eip7702.eoa_assumption",
                    family=FAMILY,
                    title="EIP-7702 unsafe EOA assumption",
                    summary=(
                        f"{function.name} gates on '{label}', which EIP-7702 makes unsound "
                        "because an externally owned account can carry code"
                    ),
                    file=function.file,
                    line=(
                        line_of_text(function, label.split(" ")[0].split("(")[0])
                        if via == "body"
                        else function.line
                    ),
                    contract=name,
                    function=function.signature,
                    facts=(
                        ("assumption", label),
                        ("subject", subject_kind),
                        ("gate", gate),
                        ("via", via),
                    ),
                    observed=("eoa_assumption_in_guard",),
                    missing=("confirm the gate is a security boundary post-7702",),
                    confidence="medium" if subject_kind == "caller" else "low",
                    impact_tags=("eip7702", "access_control"),
                )
            )
            if len(found) >= MAX_CANDIDATES_PER_FAMILY:
                return found
    return found
