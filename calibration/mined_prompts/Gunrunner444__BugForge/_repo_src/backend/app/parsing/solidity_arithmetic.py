"""Arithmetic and batch-operation analysis.

Findings need a value flow, not a keyword: an accumulation over caller-supplied
data that can wrap, a narrowing cast of an arithmetic result that is never bounded,
a division that precedes a multiplication on a value-bearing path, or opposing
operations that round the same way. Checked arithmetic on a full-width integer is
not reported, because it reverts instead of corrupting state.
"""

from __future__ import annotations

import re

from app.parsing.solidity_research import (
    ResearchModel,
    RFunction,
    SemanticCandidate,
    cap,
    member_calls,
    mentions,
)
from app.parsing.solidity_version import solidity_language_facts

FAMILY = "arithmetic"
_NARROW = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint96",
    "uint112",
    "uint128",
    "uint160",
    "int128",
)
_CEIL = re.compile(r"mulDivUp|divUp|ceilDiv|Rounding\.(Up|Ceil)|\+\s*\w+\s*-\s*1\s*\)\s*/", re.I)
_VALUE_EFFECT = re.compile(
    r"\.(transfer|safeTransfer|transferFrom|safeTransferFrom|send)\s*\(|\bcall\s*\{\s*value|\b_?(mint|burn)\s*\("
)


def _arithmetic_mode(model: ResearchModel, function: RFunction) -> str:
    pragma = model.contracts[function.contract].pragma
    return solidity_language_facts(f"pragma solidity {pragma};").arithmetic if pragma else "unknown"


def analyze_arithmetic(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for function in model.all_functions():
        if not function.has_body or function.contract not in model.contracts:
            continue
        if model.contracts[function.contract].is_interface:
            continue
        mode = _arithmetic_mode(model, function)
        found.extend(_batch_accumulation(function, mode))
        found.extend(_downcast(function))
        found.extend(_division_first(function))
    found.extend(_rounding_asymmetry(model))
    return cap(found)


def _candidate(
    function: RFunction,
    detector: str,
    title: str,
    summary: str,
    offset: int,
    facts: tuple[tuple[str, str], ...],
    observed: tuple[str, ...],
    missing: tuple[str, ...],
    confidence: str,
    in_loop: bool = False,
    tags: tuple[str, ...] = ("value_transfer",),
) -> SemanticCandidate:
    return SemanticCandidate(
        detector=f"arithmetic.{detector}",
        family=FAMILY,
        title=title,
        summary=summary,
        file=function.file,
        line=function.line + function.body[:offset].count("\n"),
        contract=function.contract,
        function=function.signature,
        path=(f"{function.contract}.{function.signature} @{function.file}:{function.line}",),
        facts=(*facts, ("in_loop", str(in_loop).lower())),
        observed=observed,
        missing=missing,
        confidence=confidence,
        impact_tags=(*tags, "arithmetic_in_loop") if in_loop else tags,
    )


def _loop_spans(body: str) -> list[tuple[int, int]]:
    from app.parsing.solidity_research import match_close, skeleton

    skel = skeleton(body)
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\bfor\s*\(", skel):
        close_paren = match_close(skel, match.end() - 1)
        if close_paren == -1:
            continue
        brace = skel.find("{", close_paren)
        if brace == -1:
            continue
        end = match_close(skel, brace)
        if end != -1:
            spans.append((brace, end))
    return spans


def _unchecked_ranges(body: str) -> list[tuple[int, int]]:
    from app.parsing.solidity_research import match_close, skeleton

    skel = skeleton(body)
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"\bunchecked\s*\{", skel):
        end = match_close(skel, match.end() - 1)
        if end != -1:
            ranges.append((match.end(), end))
    return ranges


def _batch_accumulation(function: RFunction, mode: str) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    body = function.body
    arrays = {p.name for p in function.params if p.type_name.endswith("[]") and p.name}
    if not arrays:
        return found
    unchecked = _unchecked_ranges(body)
    for start, end in _loop_spans(body):
        inside = body[start:end]
        for match in re.finditer(r"\b(\w+)\s*\+=\s*([^;]*);", inside):
            accumulator, operand = match.group(1), match.group(2)
            if not any(re.search(rf"\b{re.escape(a)}\s*\[", operand) for a in arrays):
                continue
            absolute = start + match.start()
            in_unchecked = any(low <= absolute <= high for low, high in unchecked)
            wraps = mode == "wrapping" or in_unchecked
            declaration = re.search(rf"\b(u?int\d*)\s+{re.escape(accumulator)}\b", body)
            kind = declaration.group(1) if declaration else "uint256"
            downcast = re.search(rf"\b(uint\d+)\s*\(\s*{re.escape(accumulator)}\s*\)", body)
            narrow = kind in _NARROW
            if not (wraps or (narrow and downcast)):
                continue
            used = bool(_VALUE_EFFECT.search(body[end:])) or bool(
                re.search(rf"(require|if)\s*\([^;]*\b{re.escape(accumulator)}\b", body[end:])
            )
            if not used:
                continue
            reason = (
                "the sum is unchecked, so it can wrap"
                if wraps
                else f"the sum is narrowed with {downcast.group(1) if downcast else kind} afterwards"
            )
            found.append(
                _candidate(
                    function,
                    "batch_accumulation_overflow",
                    "Caller-supplied batch can overflow its own total",
                    f"{function.contract}.{function.name} sums `{operand.strip()[:40]}` over a "
                    f"caller-supplied array into `{accumulator}` ({kind}); {reason}, and the total "
                    f"then gates a transfer or limit check.",
                    absolute,
                    (
                        ("accumulator", accumulator),
                        ("accumulator_type", kind),
                        ("arithmetic", mode),
                    ),
                    ("array elements are summed in a loop", "the total is used afterwards"),
                    ("checked full-width accumulation or an explicit bound",),
                    "medium",
                    in_loop=True,
                    tags=("value_transfer", "custody"),
                )
            )
    return found


def _downcast(function: RFunction) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    body = function.body
    for match in re.finditer(r"\b(uint(?:8|16|32|64|96|112|128|160))\s*\(", body):
        from app.parsing.solidity_research import match_close, skeleton

        skel = skeleton(body)
        close = match_close(skel, match.end() - 1)
        if close == -1:
            continue
        inner = body[match.end() : close]
        if not re.search(r"[*+\-/]|\*\*", inner):
            continue
        if re.search(r"\b(block\.timestamp|type\s*\()", inner) and "*" not in inner:
            continue
        kind = match.group(1)
        bounded = re.search(rf"type\s*\(\s*{kind}\s*\)\s*\.\s*max", body) or re.search(
            r"SafeCast|toUint\d+", body
        )
        if bounded:
            continue
        statement = body[body.rfind(";", 0, match.start()) + 1 : body.find(";", close)]
        if not (
            _VALUE_EFFECT.search(body)
            or re.search(r"[^=!<>]=[^=]", statement.replace(match.group(0), "", 1))
        ):
            continue
        found.append(
            _candidate(
                function,
                "unsafe_downcast_after_arithmetic",
                "Arithmetic result is narrowed without a bound",
                f"{function.contract}.{function.name} narrows `{inner.strip()[:50]}` to {kind} "
                "with no maximum check or SafeCast. A truncating cast does not revert, so the "
                "stored or transferred value silently loses its high bits.",
                match.start(),
                (("target_type", kind), ("expression", inner.strip()[:60])),
                ("an arithmetic expression is cast to a narrower integer",),
                (f"type({kind}).max bound or SafeCast",),
                "medium",
            )
        )
    return found


def _division_first(function: RFunction) -> list[SemanticCandidate]:
    body = function.body
    found: list[SemanticCandidate] = []
    pattern = re.compile(r"\(\s*[^()]*\w\s*/\s*[^()]*\w\s*\)\s*\*\s*\w|\b\w+\s*/\s*\w+\s*\*\s*\w+")
    for match in pattern.finditer(body):
        text = match.group(0)
        if "//" in text or "1e" in text.split("/", 1)[1].split("*")[0]:
            pass
        effect = bool(_VALUE_EFFECT.search(body)) or bool(
            re.search(r"\b\w+\s*(\[[^\]]*\])*\s*[-+]?=[^=]", body[match.end() :])
        )
        if not effect:
            continue
        found.append(
            _candidate(
                function,
                "division_before_multiplication",
                "Division loses precision before a multiplication",
                f"{function.contract}.{function.name} divides before multiplying in "
                f"`{text.strip()[:60]}`, so the multiplier amplifies the truncated remainder in a "
                "value-bearing result.",
                match.start(),
                (("expression", text.strip()[:80]),),
                ("integer division precedes multiplication",),
                ("multiplication before division, or mulDiv",),
                "medium",
                in_loop=any(low <= match.start() <= high for low, high in _loop_spans(body)),
            )
        )
    return found


def _rounding_asymmetry(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        if contract.is_interface:
            continue
        for function in contract.functions:
            if not function.has_body or not function.exposed:
                continue
            for conversion in re.finditer(
                r"\b(\w+)\s*=\s*([^;]*?)\s*\*\s*([^;/]+?)\s*/\s*([^;]+);", function.body
            ):
                shares, numerator, scale, denominator = (g.strip() for g in conversion.groups())
                if _CEIL.search(conversion.group(0)):
                    continue
                paid = next(
                    (
                        p.name
                        for p in function.params
                        if p.name
                        and mentions(numerator, p.name)
                        and any(
                            c.name in {"transfer", "safeTransfer"}
                            and len(c.arguments) >= 2
                            and c.arguments[1].strip() == p.name
                            for c in member_calls(function.body)
                        )
                    ),
                    "",
                )
                if not paid:
                    continue
                tail = function.body[conversion.end() :]
                burns = bool(
                    re.search(
                        rf"-=\s*{re.escape(shares)}\b|\b_?burn\s*\([^;]*\b{re.escape(shares)}\b",
                        tail,
                    )
                )
                if not burns:
                    continue
                found.append(
                    _candidate(
                        function,
                        "rounding_burn_floor",
                        "Shares burned for a fixed asset payout are rounded down",
                        f"{name}.{function.name} pays out exactly `{paid}` assets and burns "
                        f"`{shares}` computed with floor division. The caller can withdraw assets "
                        "for fewer shares than they are worth, and repeating the call extracts the "
                        "rounding difference from other holders.",
                        conversion.start(),
                        (("shares_variable", shares), ("assets_parameter", paid)),
                        ("fixed asset payout", "share burn uses floor division"),
                        ("round-up conversion on the burn side",),
                        "medium",
                        tags=("vault_accounting", "value_transfer", "reserve_accounting"),
                    )
                )
    return found
