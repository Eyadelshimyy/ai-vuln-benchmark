"""Semantic Solidity checks inspired by public audit methodology.

These are structural comparisons of functions already in the syntax graph.
They do not mark a finding verified, and they do not treat a keyword as proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.model import SyntaxEvent, SyntaxGraph
from app.parsing.solidity_ir import SemanticProgram, build_semantic_program

_ACCOUNTING = re.compile(
    r"\b(totalSupply|totalShares|totalAssets|totalDebt|cumulativeEarmarked|"
    r"rewardPerShare|_redemptionWeight|totalStaked|totalDeposits)\b"
)
_GROUPS = (
    frozenset(
        {
            "deposit",
            "mint",
            "withdraw",
            "redeem",
            "claim",
            "claimredemption",
            "unstake",
            "startunstake",
        }
    ),
    frozenset({"vote", "poke", "reset", "harvest"}),
)
_AUTH = re.compile(r"\bonly(Owner|Role|Admin|Governor)\b|\brequiresAuth\b", re.I)
_GUARD = re.compile(r"\b(nonReentrant|whenNotPaused|initializer)\b")
_CMP = re.compile(r"\b([A-Za-z_]\w*)\s*(>=|<=|==|>|<)\s*([A-Za-z_]\w*)")
_SPOT = re.compile(r"\b(balanceOf|getReserves)\s*\(")
_ORACLE = re.compile(r"\b(twap|observe|consult|latestRoundData|oracle)\b", re.I)
_VALUE_MOVE = re.compile(r"\b(transfer|transferFrom|safeTransfer|call)\s*\(")
_VIRTUAL = re.compile(r"\+\s*1\b|VIRTUAL|MINIMUM_LIQUIDITY|DECIMAL_OFFSET", re.I)


@dataclass(frozen=True)
class MethodologyIssue:
    rule_id: str
    function: str
    contract: str
    summary: str
    line: int = 0


def analyze_methodology(graph: SyntaxGraph) -> list[MethodologyIssue]:
    if graph.language != "solidity":
        return []
    program = build_semantic_program(graph)
    grouped = _functions_by_contract(graph)
    found: list[MethodologyIssue] = []
    for contract, functions in grouped.items():
        found.extend(_accounting(contract, functions, program))
        found.extend(_sibling_auth(contract, functions))
        found.extend(_boundaries(contract, functions))
        found.extend(_erc4626(contract, functions))
        found.extend(_flash_spot(contract, functions))
    return found


def _extra(event: SyntaxEvent) -> dict[str, str]:
    fields: dict[str, str] = {}
    raw = event.extra if isinstance(event.extra, str) else ""
    for part in raw.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def _functions_by_contract(graph: SyntaxGraph) -> dict[str, list[SyntaxEvent]]:
    grouped: dict[str, list[SyntaxEvent]] = {}
    for event in graph.events:
        if event.kind != "sol_function":
            continue
        grouped.setdefault(_extra(event).get("contract", ""), []).append(event)
    return grouped


def _name(event: SyntaxEvent) -> str:
    match = re.search(r"function\s+([A-Za-z_]\w*)", event.text)
    if match:
        return match.group(1)
    return _extra(event).get("name", "")


def _visibility(event: SyntaxEvent) -> str:
    return _extra(event).get("visibility", "")


def _modifiers(event: SyntaxEvent) -> str:
    return _extra(event).get("modifiers", "")


def _family_key(name: str) -> str:
    compact = re.sub(r"[^A-Za-z]", "", name).lower()
    for group in _GROUPS:
        if any(token in compact for token in group):
            return "|".join(sorted(group))
    return ""


def _accounting_names(text: str) -> set[str]:
    return set(_ACCOUNTING.findall(text))


def _ir_writes(program: SemanticProgram, event: SyntaxEvent, contract: str) -> set[str]:
    name = _name(event)
    for item in program.functions_named(name, contract):
        if item.line == event.line:
            return set(item.writes)
    return set()


def _accounting(
    contract: str, functions: list[SyntaxEvent], program: SemanticProgram
) -> list[MethodologyIssue]:
    family = [item for item in functions if _family_key(_name(item))]
    found: list[MethodologyIssue] = []
    for index, left in enumerate(family):
        left_names = _accounting_names(left.text) | _ir_writes(program, left, contract)
        for right in family[index + 1 :]:
            right_names = _accounting_names(right.text) | _ir_writes(program, right, contract)
            if len(left_names | right_names) < 2 or left_names == right_names:
                continue
            short, short_names, long_names = (
                (right, right_names, left_names)
                if len(right_names) < len(left_names)
                else (left, left_names, right_names)
            )
            if not short_names or not (short_names < long_names):
                continue
            if "return" not in short.text:
                continue
            missing = long_names - short_names
            found.append(
                MethodologyIssue(
                    "sol.accounting_desync",
                    _name(short),
                    contract,
                    f"`{contract}.{_name(short)}` updates {sorted(short_names)} and returns "
                    f"without the coupled writes {sorted(missing)} performed by its sibling. "
                    "This is potential evidence, not a confirmed accounting bug.",
                    line=short.line,
                )
            )
    return found


def _sibling_auth(contract: str, functions: list[SyntaxEvent]) -> list[MethodologyIssue]:
    buckets: dict[str, list[SyntaxEvent]] = {}
    for event in functions:
        key = _family_key(_name(event))
        if key and _visibility(event) in {"public", "external", ""}:
            buckets.setdefault(key, []).append(event)
    found: list[MethodologyIssue] = []
    for events in buckets.values():
        protected = [item for item in events if _AUTH.search(_modifiers(item) + item.text)]
        if not protected:
            continue
        for event in events:
            if event in protected:
                continue
            if _visibility(event) in {"internal", "private"}:
                continue
            if re.search(r"\b(view|pure)\b", event.text.split("{", 1)[0]):
                continue
            body = event.text.split("{", 1)[-1]
            if not _VALUE_MOVE.search(event.text) and not re.search(
                r"\b[A-Za-z_]\w*\s*(?:\+=|-=|=)", body
            ):
                continue
            found.append(
                MethodologyIssue(
                    "sol.sibling_auth",
                    _name(event),
                    contract,
                    f"`{contract}.{_name(event)}` is an external sibling of "
                    f"`{_name(protected[0])}` but does not repeat its authorization modifier. "
                    "This is potential evidence, not a confirmed access-control bug.",
                    line=event.line,
                )
            )
    return found


def _boundaries(contract: str, functions: list[SyntaxEvent]) -> list[MethodologyIssue]:
    comparisons: dict[tuple[str, str], list[tuple[SyntaxEvent, str]]] = {}
    for event in functions:
        if not _family_key(_name(event)):
            continue
        for match in _CMP.finditer(event.text):
            left, op, right = match.group(1), match.group(2), match.group(3)
            if left in {"require", "assert", "if"}:
                continue
            ordered = sorted((left, right))
            key = (ordered[0], ordered[1])
            comparisons.setdefault(key, []).append((event, op))
    found: list[MethodologyIssue] = []
    for key, uses in comparisons.items():
        ops = {op for _, op in uses}
        strict = {op for op in ops if op in {">", "<"}}
        inclusive = {op for op in ops if op in {">=", "<="}}
        if not strict or not inclusive:
            continue
        if not any(
            _VALUE_MOVE.search(event.text) or re.search(r"\b[A-Za-z_]\w*\s*(?:\+=|-=)", event.text)
            for event, _op in uses
        ):
            continue
        event = uses[0][0]
        found.append(
            MethodologyIssue(
                "sol.boundary",
                _name(event),
                contract,
                f"`{contract}` compares `{key[0]}` and `{key[1]}` with both {sorted(ops)} "
                "across sibling functions. An off-by-one at the boundary is possible. "
                "This is potential evidence, not a confirmed boundary bug.",
                line=event.line,
            )
        )
    return found


def _erc4626(contract: str, functions: list[SyntaxEvent]) -> list[MethodologyIssue]:
    vault_names = {_name(event).lower() for event in functions} & {
        "deposit",
        "mint",
        "withdraw",
        "redeem",
        "previewdeposit",
        "previewmint",
        "previewwithdraw",
        "previewredeem",
    }
    if len(vault_names) < 2:
        return []
    found: list[MethodologyIssue] = []
    for event in functions:
        name = _name(event).lower()
        if name not in {"deposit", "mint"}:
            continue
        text = event.text
        if "totalSupply" not in text or "totalAssets" not in text or "/" not in text:
            continue
        if _VIRTUAL.search(text):
            continue
        if not re.search(r"totalSupply\s*/\s*totalAssets|totalAssets\s*/\s*totalSupply", text):
            if not re.search(r"/\s*total(?:Assets|Supply)\b", text):
                continue
        found.append(
            MethodologyIssue(
                "sol.erc4626_inflation",
                _name(event),
                contract,
                f"`{contract}.{_name(event)}` converts assets and shares from totalSupply and "
                "totalAssets without a virtual offset or minimum liquidity. A first deposit "
                "can inflate the exchange rate. This is potential evidence, not a confirmed bug.",
                line=event.line,
            )
        )
    return found


def _flash_spot(contract: str, functions: list[SyntaxEvent]) -> list[MethodologyIssue]:
    found: list[MethodologyIssue] = []
    for event in functions:
        text = event.text
        if not _SPOT.search(text) or not _VALUE_MOVE.search(text):
            continue
        if re.search(r"\b(view|pure)\b", text.split("{", 1)[0]):
            continue
        if _ORACLE.search(text):
            continue
        if "/" not in text and "*" not in text:
            continue
        if re.search(r"\bflashLoan\b", _name(event)) and not _SPOT.search(text):
            continue
        found.append(
            MethodologyIssue(
                "sol.flash_spot",
                _name(event),
                contract,
                f"`{contract}.{_name(event)}` prices a value transfer from a same-transaction "
                "balance or reserve and does not consult a freshness-checked oracle. "
                "A flash loan is not itself the bug. This is potential evidence, not a confirmed one.",
                line=event.line,
            )
        )
    return found
