"""Low-level call outcomes that a function swallows while it returns normally.

A function that runs ``target.call(data)`` and, on the path taken when the call
fails, neither reverts nor hands the success flag back to its caller, finishes
normally whether the target action ran or not. When the transaction sender can
choose the gas, they can make the inner call fail on purpose while the outer frame
succeeds, and an upstream caller that records the outer call as "delivered" then
consumes the action without it having executed (forwarder or relayer griefing).

Both checks run over the conservative statement CFG in :mod:`solidity_cfg`:

* Failure propagation follows the CFG from the call with the success flag set to
  false. ``if`` conditions on the flag pick the failure branch, ``require``/``assert``
  of the flag reverts, a ``revert`` or a top-level ``revert`` inside an assembly
  block reverts, and ``return`` that carries the flag hands it to the caller. Any
  path that reaches the end of the function, or returns without the flag, swallows
  the failure. Anything the CFG cannot decide is reported as uncertain, and the
  candidate is kept.
* A fixed ``{gas: N}`` stipend is classified as ``NO_GUARD``,
  ``GUARD_PRESENT_BUT_INSUFFICIENT_OR_UNPROVEN`` or ``GUARD_PROVEN_SUFFICIENT``.
  A guard must dominate the call, stop the low-gas path, and its required amount
  must be shown (by a bounded source-text model, see :func:`gas_guard`) to cover
  the stipend, the EIP-150 63/64 forwarding loss and a fixed CALL overhead. Only
  ``GUARD_PROVEN_SUFFICIENT`` removes a site from reporting; it means sufficient only
  under that bounded source-text model, never a proof of runtime gas safety.
* Empty-calldata calls are analysed like every other call, including ETH
  transfers; the candidate's ``call_kind`` fact labels them, because a variable
  value may be zero at runtime and an intentional best-effort transfer is a
  possible false positive.

This is statement-level control flow over source text, not a compiler CFG. It does
not track values, aliases of the flag, calls through helpers, ``try``/``catch``
(reported as uncertain), or assembly-level ``call``. Every result is a static candidate, never verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.solidity_cfg import FunctionCfg, node_at, node_ranges
from app.parsing.solidity_research import (
    MemberCall,
    ResearchModel,
    RFunction,
    SemanticCandidate,
    match_close,
    member_calls,
    skeleton,
)

DETECTOR = "caller_context.swallowed_call_failure"
FAMILY = "caller_context"
_STATEMENT_START = re.compile(r"[;{}]")
_ZERO = re.compile(r"^(?:0+|0x0+|0\s*(?:wei|gwei|ether))$")
_EMPTY_DATA = frozenset({'""', "''", 'hex""', "hex''", "newbytes(0)", 'bytes("")', "bytes('')"})
_KEYWORDS = frozenset({"gasleft", "uint256", "uint", "type", "max", "min", "true", "false"})

NO_GUARD = "NO_GUARD"
GUARD_UNPROVEN = "GUARD_PRESENT_BUT_INSUFFICIENT_OR_UNPROVEN"
GUARD_PROVEN = "GUARD_PROVEN_SUFFICIENT"
EIP150_DIVISOR = 63
# Bounded model of gas spent between the gasleft() check and the callee starting:
# 2600 cold account access (EIP-2929) plus a margin for the opcodes, stack work and
# a small memory expansion. A value-bearing call adds 9000 (value transfer) and
# 25000 (possible new account). Large calldata/returndata memory expansion is NOT
# modelled; a "proven" result is proven only under these assumptions.
CALL_OVERHEAD = 5000
VALUE_CALL_OVERHEAD = CALL_OVERHEAD + 9000 + 25000

PROPAGATED = "propagated"
SWALLOWED = "swallowed"
UNCERTAIN = "uncertain"

# How to read a candidate. Emission is agreement with this detector, never a
# measured vulnerability and never a precision score.
EMISSION = "detector_emission"
POSSIBLE_FP = "possible_false_positive"
AMBIGUOUS = "ambiguous"
NO_VULNERABILITY_CLAIM = "none"


@dataclass(frozen=True)
class FailureOutcome:
    """What happens on the paths taken when the low-level call fails."""

    status: str  # propagated | swallowed | uncertain
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GasGuard:
    """Gas-guard status for one call site and a human-readable reason."""

    status: str  # NO_GUARD | GUARD_PRESENT_BUT_INSUFFICIENT_OR_UNPROVEN | GUARD_PROVEN_SUFFICIENT
    note: str

    @property
    def bounded(self) -> bool:
        return self.status == GUARD_PROVEN


def success_variable(body: str, call_start: int) -> str:
    """Name of the bool the call's success flag is stored in, or an empty string."""
    head = body[:call_start]
    starts = [m.end() for m in _STATEMENT_START.finditer(head)]
    statement = head[starts[-1] if starts else 0 :]
    match = re.search(r"\(\s*(?:bool\s+)?([A-Za-z_]\w*)\s*[,)]", statement)
    if match is None:
        match = re.search(r"\bbool\s+([A-Za-z_]\w*)\s*=", statement)
    if match is None or "=" not in statement:
        return ""
    return match.group(1)


def call_kind(call: MemberCall) -> tuple[str, str]:
    """(kind, description) of a ``.call``; no kind is excluded from analysis.

    Kinds: ``empty_no_value``, ``eth_transfer_literal``, ``eth_transfer_variable``,
    ``value_with_calldata``, ``calldata``. Static analysis does not know whether a
    variable value is zero at runtime.
    """
    data = re.sub(r"\s+", "", call.arguments[0]) if call.arguments else '""'
    empty = data in _EMPTY_DATA
    value = _option(call.options, "value")
    zero = not value or _ZERO.match(value.strip()) is not None
    if empty and zero:
        return (
            "empty_no_value",
            "empty calldata with no ether: runs the target's receive() or fallback()",
        )
    if empty and re.fullmatch(r"[\d_]+(?:e\d+)?(?:\s*(?:wei|gwei|ether))?", value.strip()):
        return (
            "eth_transfer_literal",
            f"plain ETH transfer of `{value}` (runs the recipient's receive())",
        )
    if empty:
        return (
            "eth_transfer_variable",
            f"ETH transfer of `{value}`; static analysis cannot tell whether it is zero at "
            "runtime (a zero value makes it a plain receive()/fallback() call)",
        )
    if not zero:
        return "value_with_calldata", f"non-empty calldata with ether value `{value}`"
    return "calldata", "non-empty calldata"


def contract_constants(model: ResearchModel) -> dict[str, int]:
    """Integer ``constant`` declarations in the model; names with conflicting values are dropped."""
    found: dict[str, int] = {}
    conflict: set[str] = set()
    pattern = r"\bconstant\s+([A-Za-z_]\w*)\s*=\s*([\d_]+(?:e\d+)?)\s*;"
    for contract in model.contracts.values():
        for name, raw in re.findall(pattern, contract.body):
            value = _int(raw)
            if value is None:
                continue
            if name in found and found[name] != value:
                conflict.add(name)
            found.setdefault(name, value)
    return {k: v for k, v in found.items() if k not in conflict}


def call_cfg(body: str, call_start: int) -> tuple[FunctionCfg, int | None]:
    """The function CFG and the node holding the call at ``call_start`` in ``body``."""
    cfg, ranges, node = _cfg_ranges(body, call_start)
    return cfg, node


def _cfg_ranges(
    body: str, call_start: int
) -> tuple[FunctionCfg, dict[int, tuple[int, int]], int | None]:
    """CFG, node ranges in ``body`` offsets, and the call's node."""
    text = "{" + body + "}"
    cfg, ranges = node_ranges(text)
    shifted = {k: (a - 1, b - 1) for k, (a, b) in ranges.items()}
    if not cfg.known:
        return cfg, shifted, None
    return cfg, shifted, node_at(shifted, call_start)


def failure_outcome(body: str, call_start: int, flag: str) -> FailureOutcome:
    """Follow the CFG from the call with ``flag`` false and classify every path."""
    cfg, start = call_cfg(body, call_start)
    if start is None:
        return FailureOutcome(UNCERTAIN, ("the control flow of the function could not be built",))
    outcomes: set[str] = set()
    reasons: list[str] = []
    seen: set[tuple[int, str]] = set()
    stack = [(dst, label) for dst, label in _out(cfg, start) if label != "fail"]
    while stack:
        node_id, label = stack.pop()
        if (node_id, label) in seen:
            continue
        seen.add((node_id, label))
        status, reason, follow = _step(cfg, node_id, flag)
        if status:
            outcomes.add(status)
            if reason:
                reasons.append(reason)
        stack.extend(follow)
    if not outcomes:
        outcomes.add(UNCERTAIN)
        reasons.append("no path from the call could be followed")
    if outcomes == {PROPAGATED}:
        return FailureOutcome(PROPAGATED, tuple(dict.fromkeys(reasons)))
    if SWALLOWED in outcomes:
        return FailureOutcome(SWALLOWED, tuple(dict.fromkeys(reasons)))
    return FailureOutcome(UNCERTAIN, tuple(dict.fromkeys(reasons)))


def gas_guard(
    body: str,
    call_start: int,
    options: str,
    constants: dict[str, int] | None = None,
    sends_value: bool = False,
) -> GasGuard:
    """Classify the gas guard of the call at ``call_start``.

    A guard must (1) dominate the call, (2) stop the low-gas path, (3) reference the
    stipend, and (4) have a required amount that, under a bounded model, covers
    ``stipend + stipend / 63 + overhead`` where overhead is at least
    :data:`CALL_OVERHEAD` (or :data:`VALUE_CALL_OVERHEAD` for a value call). The
    amount may be written directly or through one local assigned exactly once before
    the check; named integer constants are resolved from ``constants``. Anything the
    model cannot evaluate is ``GUARD_PRESENT_BUT_INSUFFICIENT_OR_UNPROVEN``. This is
    an approximation over source text, not a gas calculation.
    """
    stipend = _option(options, "gas")
    if not stipend:
        return GasGuard(NO_GUARD, "no gas option: the call forwards all remaining gas")
    cfg, ranges, call_node = _cfg_ranges(body, call_start)
    if call_node is None:
        return GasGuard(NO_GUARD, "control flow could not be built, so no guard is assumed")
    checks = [n.node_id for n in cfg.nodes if n.kind in {"require", "if"} and "gasleft" in n.text]
    if not checks:
        return GasGuard(NO_GUARD, "no gasleft() check in the function")
    tokens = _tokens(stipend)
    unrelated = False
    unproven: str = ""
    for check in checks:
        amount = _gas_check_protects(cfg, check, call_node)
        if amount is None:
            continue
        numeric = (
            _value(_norm(stipend), constants or {}) is not None
            and _fold(amount, constants or {}) is not None
        )
        if not numeric and not tokens & _expanded_tokens(body, amount):
            unrelated = True
            continue
        guard_at = ranges.get(check, (0, 0))[0]
        proven, reason = _sufficient(
            body, amount, stipend, guard_at, call_start, constants or {}, sends_value
        )
        if proven:
            return GasGuard(GUARD_PROVEN, f"{reason} ({_short(cfg.nodes[check].text)})")
        unproven = unproven or f"{reason} ({_short(cfg.nodes[check].text)})"
    if unproven:
        return GasGuard(GUARD_UNPROVEN, unproven)
    if unrelated:
        return GasGuard(
            NO_GUARD, "a dominating gasleft() check does not reference this call's stipend"
        )
    return GasGuard(
        NO_GUARD,
        "a gasleft() check exists but does not protect this call (after it or on another path)",
    )


def parameter_binding(function: RFunction, expression: str, type_hint: str) -> str:
    """Parameter name when ``expression`` is that parameter, else ``unknown``.

    ``payable(name)`` and ``address(name)`` unwrap. Anything else, including
    ``msg.sender`` and ``address(this)``, stays unknown: the harness must not
    invent a callee for a receiver the caller does not supply.
    """
    text = expression.strip()
    for _ in range(3):
        match = re.fullmatch(r"(?:payable|address)\s*\(\s*(.+)\s*\)", text)
        if match is None:
            break
        text = match.group(1).strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", text):
        return "unknown"
    for param in function.params:
        if param.name != text:
            continue
        declared = param.type_name.strip()
        if type_hint == "address" and "address" in declared:
            return text
        if type_hint == "bytes" and re.match(r"bytes(?!\d)", declared):
            return text
    return "unknown"


def evidence_classification(
    function: RFunction, call: MemberCall, kind: str, outcome: str
) -> tuple[str, str]:
    """(evidence_class, pattern). Neither value is a vulnerability verdict."""
    transfer = kind in {"eth_transfer_literal", "eth_transfer_variable"}
    if not transfer:
        if outcome == UNCERTAIN:
            return AMBIGUOUS, "ambiguous_failure_path"
        return EMISSION, "swallowed_low_level_call"
    receiver = call.receiver
    options = call.options or ""
    if (
        "msg.sender" in receiver
        or function.name.startswith("validate")
        or "fund" in options.lower()
    ):
        return POSSIBLE_FP, "prefund_style_transfer"
    if "hook" in function.contract.lower() or "hook" in function.name.lower():
        return POSSIBLE_FP, "best_effort_hook"
    return POSSIBLE_FP, "best_effort_transfer"


def swallowed_call_failures(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    constants = contract_constants(model)
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        if contract.is_interface:
            continue
        for function in model.functions_of(name, inherited=False):
            if not function.has_body or function.mutability in {"view", "pure"}:
                continue
            candidate = _function_candidate(function, constants)
            if candidate is not None:
                found.append(candidate)
    return found


# ---- internals ------------------------------------------------------------------------------


def _out(cfg: FunctionCfg, node_id: int) -> list[tuple[int, str]]:
    return [(dst, label) for src, dst, label in cfg.edges if src == node_id]


def _step(cfg: FunctionCfg, node_id: int, flag: str) -> tuple[str, str, list[tuple[int, str]]]:
    """Outcome reached at a node (if any), a reason, and the edges to keep following."""
    node = cfg.nodes[node_id]
    text = node.text.strip()
    onward = [(dst, label) for dst, label in _out(cfg, node_id) if label != "fail"]
    if node.kind == "exit":
        return SWALLOWED, "the failure path reaches the end of the function", []
    if node.kind == "revert":
        return PROPAGATED, "", []
    if node.kind == "return":
        if _mentions(text, flag):
            return PROPAGATED, "", []
        return SWALLOWED, f"the failure path returns without the flag: {_short(text)}", []
    if node.kind == "require":
        polarity = _flag_test(_first_argument(text), flag)
        if polarity == "flag":
            return PROPAGATED, "", []
        if polarity == "other" and _mentions(text, flag):
            return UNCERTAIN, f"unrecognized check of the flag: {_short(text)}", onward
        return "", "", onward
    if node.kind == "if":
        polarity = _flag_test(_strip_parens(_if_condition(text)), flag)
        branches = {label: (dst, label) for dst, label in _out(cfg, node_id)}
        if polarity == "flag":  # failure makes the condition false
            return "", "", [branches["else"]] if "else" in branches else []
        if polarity == "not_flag":  # failure makes the condition true
            return "", "", [branches["then"]] if "then" in branches else []
        if _mentions(text, flag):
            return UNCERTAIN, f"compound condition on the flag: {_short(text)}", onward
        return "", "", onward
    if node.kind == "try":
        # The shared CFG does not model try/catch bodies or what follows them.
        return UNCERTAIN, "try/catch on the failure path is not modelled", []
    if node.kind == "stmt":
        if re.match(r"assembly\b", text):
            return _assembly_step(text, onward)
        if _reassigns(text, flag):
            return UNCERTAIN, f"the flag is overwritten before any check: {_short(text)}", []
    return "", "", onward


def _assembly_step(
    text: str, onward: list[tuple[int, str]]
) -> tuple[str, str, list[tuple[int, str]]]:
    brace = text.find("{")
    skel = skeleton(text)
    close = match_close(skel, brace) if brace >= 0 else -1
    if close < 0:
        return UNCERTAIN, "assembly block could not be read", onward
    inner = skel[brace + 1 : close]
    depth = 0
    top = []
    for char in inner:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        top.append(char if depth == 0 else " ")
    top_text = "".join(top)
    if re.search(r"\brevert\s*\(", top_text):
        return PROPAGATED, "", []
    if re.search(r"\b(?:revert|return|stop|invalid)\s*\(", inner):
        return UNCERTAIN, "assembly on the failure path ends the call only conditionally", onward
    return "", "", onward


def _gas_check_protects(cfg: FunctionCfg, check: int, call: int) -> str | None:
    """The compared amount when ``check`` stops the low-gas path to ``call``, else None."""
    if not cfg.dominates(check, call) or check == call:
        return None
    node = cfg.nodes[check]
    if node.kind == "require":
        side, amount = _gas_comparison(_first_argument(node.text))
        return amount if side == "enough" else None
    side, amount = _gas_comparison(_strip_parens(_if_condition(node.text)))
    if side is None:
        return None
    then_ids = cfg.then_of.get(check, set())
    else_ids = cfg.else_of.get(check, set())
    join = cfg.join_of.get(check)
    if side == "enough":
        if call in then_ids:
            return amount
        if call in else_ids:
            return None
        return amount if _branch_terminates(cfg, check, else_ids, "else", join) else None
    if call in then_ids:
        return None
    return amount if _branch_terminates(cfg, check, then_ids, "then", join) else None


def _branch_terminates(
    cfg: FunctionCfg, node: int, region: set[int], label: str, join: int | None
) -> bool:
    if join is None:
        return False
    if not region:
        return not any(s == node and d == join and lab == label for s, d, lab in cfg.edges)
    return not any(d == join and s in region for s, d, _lab in cfg.edges)


def _gas_comparison(condition: str) -> tuple[str | None, str]:
    """('enough'|'low', compared amount) for one ``gasleft()`` comparison, else (None, '')."""
    text = condition.strip()
    if re.search(r"&&|\|\|", text):
        return None, ""
    forward = re.fullmatch(r"gasleft\s*\(\s*\)\s*(>=|>|<=|<)\s*(.+)", text, re.S)
    if forward:
        op, amount = forward.group(1), forward.group(2)
        return ("enough" if op in {">=", ">"} else "low"), amount
    backward = re.fullmatch(r"(.+?)\s*(>=|>|<=|<)\s*gasleft\s*\(\s*\)", text, re.S)
    if backward:
        amount, op = backward.group(1), backward.group(2)
        return ("enough" if op in {"<=", "<"} else "low"), amount
    return None, ""


def _flag_test(condition: str, flag: str) -> str:
    """'flag' when the condition is the flag, 'not_flag' when it is its negation."""
    text = re.sub(r"\s+", "", _strip_parens(condition))
    name = re.escape(flag)
    if re.fullmatch(rf"{name}|{name}==true|true=={name}|{name}!=false", text):
        return "flag"
    if re.fullmatch(rf"!\(?{name}\)?|{name}==false|false=={name}|{name}!=true", text):
        return "not_flag"
    return "other"


def _first_argument(statement: str) -> str:
    open_at = statement.find("(")
    if open_at < 0:
        return ""
    skel = skeleton(statement)
    close = match_close(skel, open_at)
    if close < 0:
        return ""
    inner = statement[open_at + 1 : close]
    depth = 0
    for index, char in enumerate(skeleton(inner)):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            return inner[:index]
    return inner


def _if_condition(statement: str) -> str:
    open_at = statement.find("(")
    if open_at < 0:
        return ""
    close = match_close(skeleton(statement), open_at)
    return statement[open_at : close + 1] if close >= 0 else ""


def _strip_parens(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and match_close(skeleton(text), 0) == len(text) - 1:
        text = text[1:-1].strip()
    return text


def _reassigns(statement: str, flag: str) -> bool:
    name = re.escape(flag)
    if re.search(rf"(?<![\w.]){name}\s*=(?!=)", statement):
        return True
    return re.search(rf"\(\s*(?:bool\s+)?{name}\s*,[^;]*\)\s*=(?!=)", statement) is not None


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}\b", text) is not None


def _option(options: str, key: str) -> str:
    match = re.search(rf"\b{key}\s*:\s*([^,]+)", options)
    return match.group(1).strip() if match else ""


def _tokens(expression: str) -> set[str]:
    found = set(re.findall(r"[A-Za-z_]\w*|\d[\d_]*", expression))
    return {item for item in found if item not in _KEYWORDS}


def _sufficient(
    body: str,
    amount: str,
    stipend: str,
    guard_at: int,
    call_at: int,
    constants: dict[str, int],
    sends_value: bool,
) -> tuple[bool, str]:
    """Bounded check that ``amount`` covers stipend + stipend/63 + overhead."""
    need_overhead = VALUE_CALL_OVERHEAD if sends_value else CALL_OVERHEAD
    for name in _identifiers(stipend):
        if _assignments(body, name, guard_at, call_at):
            return False, f"unproven: stipend `{name}` is reassigned between the check and the call"
    expression = _strip_parens(amount)
    if re.fullmatch(r"[A-Za-z_]\w*", expression) and expression not in constants:
        sites = _assignment_rhs(body, expression)
        if len(sites) > 1:
            return False, (
                f"unproven: required amount `{expression}` is assigned {len(sites)} times; "
                "the detector cannot tell which value the check uses"
            )
        if sites:
            at, rhs = sites[0]
            if at > guard_at:
                return False, f"unproven: `{expression}` is assigned after the check"
            for name in _identifiers(rhs):
                if _assignments(body, name, at, guard_at):
                    return False, (
                        f"unproven: `{name}` is reassigned before the check uses `{expression}`"
                    )
            expression = _strip_parens(rhs)
    terms = _top_level_terms(expression)
    if terms is None:
        return False, f"unproven: `{_short(expression)}` is not a sum the detector can evaluate"
    norm_stipend = _norm(stipend)
    stipend_value = _value(norm_stipend, constants)
    has_stipend = has_eip150 = False
    overhead = 0
    for term in terms:
        norm = _norm(term)
        if norm == norm_stipend:
            has_stipend = True
        elif norm in {f"{norm_stipend}/63", f"({norm_stipend})/63"}:
            has_eip150 = True
        elif norm in {f"{norm_stipend}*64/63", f"({norm_stipend})*64/63"}:
            has_stipend = has_eip150 = True
        else:
            folded = _value(norm, constants)
            if folded is None:
                return False, f"unproven: term `{term.strip()}` cannot be evaluated statically"
            overhead += folded
    if stipend_value is not None and not (has_stipend or has_eip150):
        needed = stipend_value + stipend_value // EIP150_DIVISOR + need_overhead
        if overhead >= needed:
            return True, (
                f"sufficient only under the bounded source-text gas model, not a runtime safety proof: {overhead} >= {needed} "
                f"(stipend + stipend/63 + {need_overhead} overhead)"
            )
        return False, f"insufficient: requires {overhead}, the bounded model needs {needed}"
    missing = []
    if not has_stipend:
        missing.append("the stipend itself")
    if not has_eip150:
        missing.append(
            "the EIP-150 `stipend / 63` term (the callee may receive less than the stipend)"
        )
    if overhead < need_overhead:
        missing.append(f"a fixed overhead of at least {need_overhead} gas (found {overhead})")
    if missing:
        return False, "insufficient or unproven: the required amount lacks " + "; ".join(missing)
    return True, (
        "sufficient only under the bounded source-text gas model, not a runtime safety proof: stipend + stipend/63 + "
        f"{overhead} >= {need_overhead} overhead"
    )


def _fold(expression: str, constants: dict[str, int]) -> int | None:
    """Integer value of a sum of literals/known constants, else None."""
    terms = _top_level_terms(_strip_parens(expression))
    if terms is None:
        return None
    total = 0
    for term in terms:
        value = _value(_norm(term), constants)
        if value is None:
            return None
        total += value
    return total


def _top_level_terms(expression: str) -> list[str] | None:
    """Split a sum on top-level ``+``; None if it has other top-level operators."""
    skel = skeleton(expression)
    depth = 0
    cuts = [-1]
    for index, char in enumerate(skel):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "+":
            cuts.append(index)
        elif depth == 0 and char in "-?<>=&|%":
            return None
    cuts.append(len(expression))
    terms = [expression[a + 1 : b] for a, b in zip(cuts, cuts[1:], strict=False)]
    return [t for t in terms if t.strip()] or None


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _int(raw: str) -> int | None:
    text = raw.replace("_", "")
    match = re.fullmatch(r"(\d+)(?:e(\d+))?", text)
    if match is None:
        return None
    return int(match.group(1)) * int(10 ** int(match.group(2) or 0))


def _value(norm: str, constants: dict[str, int]) -> int | None:
    """Integer value of a literal, a known constant, or ``a/63`` / ``a*b`` of those."""
    norm = _strip_parens(norm)
    if norm in constants:
        return constants[norm]
    direct = _int(norm)
    if direct is not None:
        return direct
    match = re.fullmatch(r"([\w]+)([*/])([\w]+)", norm)
    if match:
        left, right = _value(match.group(1), constants), _value(match.group(3), constants)
        if left is None or right is None or (match.group(2) == "/" and right == 0):
            return None
        return left * right if match.group(2) == "*" else left // right
    return None


def _identifiers(expression: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z_]\w*", expression) if t not in _KEYWORDS}


def _assignment_rhs(body: str, name: str) -> list[tuple[int, str]]:
    pattern = rf"(?<![\w.]){re.escape(name)}\s*(?:[-+*/%]?=)(?!=)\s*([^;]+);"
    return [(m.start(), m.group(1)) for m in re.finditer(pattern, body)]


def _assignments(body: str, name: str, start: int, end: int) -> bool:
    """Whether ``name`` is assigned (or ++/--) strictly between two offsets."""
    segment = body[start:end]
    name_re = re.escape(name)
    if re.search(rf"(?<![\w.]){name_re}\s*(?:[-+*/%]?=)(?!=)", segment):
        return True
    return (
        re.search(rf"(?:\+\+|--)\s*{name_re}\b|(?<![\w.]){name_re}\s*(?:\+\+|--)", segment)
        is not None
    )


def _expanded_tokens(body: str, expression: str, depth: int = 3) -> set[str]:
    """Tokens of ``expression`` plus those of local assignments it names (bounded depth).

    ``uint256 required = limit + limit / 63 + OVERHEAD; if (gasleft() < required)``
    ties the check to ``limit``. Assignments are matched by name in the function
    body, not by position, so a reassigned local is a known imprecision.
    """
    found = _tokens(expression)
    frontier = set(found)
    for _ in range(depth):
        nxt: set[str] = set()
        for name in frontier:
            pattern = rf"(?<![\w.]){re.escape(name)}\s*=(?!=)\s*([^;]+);"
            for match in re.finditer(pattern, body):
                nxt |= _tokens(match.group(1)) - found
        if not nxt:
            break
        found |= nxt
        frontier = nxt
    return found


def _short(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:80]


def _gas_behavior(call: MemberCall, guard: GasGuard) -> str:
    stipend = _option(call.options, "gas")
    if not stipend:
        return (
            "forwards all remaining gas (at most 63/64 of gasleft() per EIP-150), so the "
            "transaction sender's gas choice decides what the callee receives"
        )
    return (
        f"requests a fixed stipend of `{stipend}` ({guard.status}: {guard.note}); with less "
        "gas left the callee receives at most 63/64 of the remainder, so the sender's gas "
        "choice may still make the call fail"
    )


_BEST_EFFORT = (
    "an ETH transfer whose failure is ignored can be deliberate best-effort behaviour "
    "(for example an ERC-4337 prefund or a refund to a contract that rejects ether); this "
    "candidate is a possible false positive until the caller's expectations are checked"
)


def _function_candidate(
    function: RFunction, constants: dict[str, int] | None = None
) -> SemanticCandidate | None:
    reported: tuple[MemberCall, str, FailureOutcome, GasGuard, tuple[str, str]] | None = None
    sites: list[str] = []
    for call in member_calls(function.body):
        if call.name != "call":
            continue
        flag = success_variable(function.body, call.start)
        if not flag:
            continue
        kind = call_kind(call)
        sends_value = kind[0] in {
            "eth_transfer_literal",
            "eth_transfer_variable",
            "value_with_calldata",
        }
        line = function.line + function.body[: call.start].count("\n")
        outcome = failure_outcome(function.body, call.start, flag)
        guard = gas_guard(function.body, call.start, call.options, constants, sends_value)
        stipend = _option(call.options, "gas")
        gas_word = f"stipend {stipend}" if stipend else "all remaining gas"
        sites.append(f"L{line}: {kind[0]}, {gas_word}, {guard.status}, failure {outcome.status}")
        if outcome.status == PROPAGATED or guard.bounded:
            continue
        if reported is None:
            reported = (call, flag, outcome, guard, kind)
    if reported is None:
        return None
    call, flag, outcome, guard, kind = reported
    line = function.line + function.body[: call.start].count("\n")
    gas_behavior = _gas_behavior(call, guard)
    transfer = kind[0] in {"eth_transfer_literal", "eth_transfer_variable"}
    evidence_class, pattern = evidence_classification(function, call, kind[0], outcome.status)
    target_parameter = parameter_binding(function, call.receiver, "address")
    data_argument = re.sub(r"\s+", "", call.arguments[0]) if call.arguments else ""
    if not data_argument or data_argument in _EMPTY_DATA:
        calldata_parameter = "empty"
    else:
        calldata_parameter = parameter_binding(function, call.arguments[0], "bytes")
    if outcome.status == SWALLOWED:
        failure_text = (
            "on at least one path taken when the call fails, the function neither reverts "
            "nor returns the flag, so it completes normally whether or not the call succeeded"
        )
    else:
        failure_text = (
            "propagation of a failure could not be established on every failure path "
            f"({'; '.join(outcome.reasons) or 'unknown'}), so the candidate is kept"
        )
    impact = (
        "A failed ether transfer is then silent. "
        + _BEST_EFFORT[0].upper()
        + _BEST_EFFORT[1:]
        + "."
        if transfer
        else "If an upstream caller records the outer call as delivered, the action can be "
        "consumed without executing."
    )
    facts: list[tuple[str, str]] = [
        ("success_variable", flag),
        ("failure_propagation", outcome.status),
        ("failure_paths", "; ".join(outcome.reasons)[:240] or "none recorded"),
        ("gas_forwarded", gas_behavior),
        ("gas_guard", guard.status),
        ("gas_guard_reason", guard.note[:240]),
        ("call_kind", kind[0]),
        ("calldata", kind[1]),
        ("call_sites", " | ".join(sites)[:400]),
        ("bounded_call_paths", str(sum(1 for site in sites if GUARD_PROVEN in site))),
        ("call_receiver", call.receiver[:120] or "unknown"),
        ("target_parameter", target_parameter),
        ("calldata_parameter", calldata_parameter),
        ("evidence_class", evidence_class),
        ("pattern", pattern),
        ("vulnerability_claim", NO_VULNERABILITY_CLAIM),
        (
            "analysis",
            "statement-level CFG over source text; bounded gas model "
            f"(stipend + stipend/63 + {CALL_OVERHEAD} or {VALUE_CALL_OVERHEAD} for value calls); "
            "not value- or alias-aware",
        ),
    ]
    if transfer:
        facts.append(("intent", _BEST_EFFORT))
    return SemanticCandidate(
        detector=DETECTOR,
        family=FAMILY,
        title=(
            "Ether transfer failure may be ignored while the function returns normally"
            if transfer
            else "Low-level call failure may be swallowed while the function returns normally"
        ),
        summary=(
            f"{function.contract}.{function.name} stores the result of `{call.text[:60]}` in "
            f"`{flag}`; {failure_text}. The call {gas_behavior}. {impact} Static candidate only."
        ),
        file=function.file,
        line=line,
        contract=function.contract,
        function=function.signature,
        path=(f"{function.contract}.{function.signature} @{function.file}:{function.line}",),
        facts=tuple(facts),
        observed=(call.text[:120],),
        missing=(
            "proof that an upstream caller or user relies on the call having succeeded",
            "proof that a third party can submit the outer transaction with chosen gas",
            "proof that the skipped action has value (impact)",
        ),
        confidence="low",
        impact_tags=("griefing", "liveness"),
    )
