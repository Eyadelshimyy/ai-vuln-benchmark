"""Oracle quality and quorum analysis.

The analysis describes a price path: which sources a price function reads, which
validations surround each read, how several sources are aggregated, and which
consumer uses the result. A candidate names the valuation path and the control
that is missing or too weak. An oracle is not unsafe because it is an oracle, and a
single-source design is not unsafe because it has one source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.parsing.solidity_research import (
    MemberCall,
    ResearchModel,
    RFunction,
    SemanticCandidate,
    cap,
    member_calls,
    mentions,
    plain_calls,
)

FAMILY = "oracle_quality"
MAX_ORACLE_SOURCES = 16
MAX_PRODUCERS = 64

_CHAINLINK = ("latestRoundData", "latestAnswer")
_SPOT = ("getReserves", "slot0")
_TWAP = ("observe", "consult", "currentCumulativePrice", "price0CumulativeLast")
_USE_HINTS = (
    ("collateral", "collateral_valuation"),
    ("liquidat", "liquidation"),
    ("borrow", "debt"),
    ("debt", "debt"),
    ("mint", "minting"),
    ("redeem", "redemption"),
    ("reward", "rewards"),
    ("share", "share_conversion"),
    ("withdraw", "withdrawals"),
    ("leverage", "leverage"),
    ("swap", "swaps"),
)


@dataclass
class PriceSource:
    kind: str  # chainlink | spot_reserves | spot_slot0 | twap | adapter
    receiver: str
    line: int
    offset: int


@dataclass
class PriceFacts:
    function: RFunction
    sources: list[PriceSource] = field(default_factory=list)
    freshness: bool = False
    round_check: bool = False
    positivity: bool = False
    bounds: bool = False
    decimals: bool = False
    deviation: bool = False
    aggregated: bool = False
    tolerant: bool = False
    quorum_checked: bool = False
    quorum_min: int | None = None
    aggregation: str = ""
    fallback_weaker: str = ""
    price_variables: tuple[str, ...] = ()

    @property
    def kinds(self) -> set[str]:
        return {item.kind for item in self.sources}


def analyze_oracle_quality(model: ResearchModel) -> list[SemanticCandidate]:
    producers = _producers(model)
    found: list[SemanticCandidate] = []
    for facts in producers:
        consumers = _consumers(model, facts.function)
        found.extend(_candidates(model, facts, consumers))
    return cap(found)


def _producers(model: ResearchModel) -> list[PriceFacts]:
    found: list[PriceFacts] = []
    for function in model.all_functions():
        if not function.has_body or function.contract not in model.contracts:
            continue
        facts = _facts(model, function)
        if facts.sources:
            found.append(facts)
        if len(found) >= MAX_PRODUCERS:
            break
    return found


def _facts(model: ResearchModel, function: RFunction) -> PriceFacts:
    body = function.body
    facts = PriceFacts(function)
    for call in member_calls(body):
        line = function.line + body[: call.start].count("\n")
        if call.name in _CHAINLINK:
            facts.sources.append(PriceSource("chainlink", call.receiver, line, call.start))
        elif call.name in _SPOT:
            kind = "spot_slot0" if call.name == "slot0" else "spot_reserves"
            facts.sources.append(PriceSource(kind, call.receiver, line, call.start))
        elif call.name in _TWAP:
            facts.sources.append(PriceSource("twap", call.receiver, line, call.start))
        elif _adapter_read(model, function, call.receiver, call.name) or _indexed_source(
            body, call
        ):
            facts.sources.append(PriceSource("adapter", call.receiver, line, call.start))
    facts.sources = facts.sources[:MAX_ORACLE_SOURCES]
    if not facts.sources:
        return facts
    names = _round_names(body)
    facts.price_variables = tuple(v for v in (names.get("answer"), "price") if v)
    facts.freshness = _freshness(body, names)
    facts.round_check = bool(
        names.get("answeredInRound")
        and names.get("roundId")
        and re.search(
            rf"{names['answeredInRound']}\s*(>=|<|>|<=)\s*{names['roundId']}"
            rf"|{names['roundId']}\s*(>=|<|>|<=)\s*{names['answeredInRound']}",
            body,
        )
    )
    facts.positivity = _positivity(body, names, facts.price_variables)
    facts.bounds = bool(
        re.search(r"\b(min|max)\w*(Price|Answer|Bound)\w*|\b(MIN|MAX)_\w*PRICE\w*", body)
    )
    facts.decimals = "decimals" in body
    facts.deviation = bool(re.search(r"deviation|maxDelta|\babs\s*\(|diff\s*[<>]=?", body, re.I))
    _aggregation(model, function, facts)
    _fallback(function, facts)
    return facts


def _indexed_source(body: str, call: MemberCall) -> bool:
    """A call on one element of an array inside a loop reads one of several sources."""
    if not re.search(r"\w+\s*\[\s*\w+\s*\]\s*\)?$", call.receiver.strip()):
        return False
    before = body[: call.start]
    return bool(re.search(r"\bfor\s*\(", before)) and call.name not in {"length", "push", "pop"}


def _adapter_read(model: ResearchModel, function: RFunction, receiver: str, name: str) -> bool:
    """A call to an in-model function that itself reads a price source."""
    state = model.state_vars(function.contract)
    base = re.sub(r"\[.*\]", "", receiver.strip())
    contract = model.contract_for_type(state.get(base, "")) or model.contract_for_type(
        re.sub(r"\(.*\)", "", receiver.strip())
    )
    if not contract:
        return False
    for target in model.functions_named(contract, name):
        if any(call.name in _CHAINLINK + _SPOT + _TWAP for call in member_calls(target.body)):
            return True
    return False


def _round_names(body: str) -> dict[str, str]:
    match = re.search(r"\(([^)]*)\)\s*=\s*[^;]*\blatestRoundData\s*\(", body) or re.search(
        r"\blatestRoundData\s*\(\s*\)\s*returns\s*\(([^)]*)\)", body
    )
    if not match:
        return {}
    order = ("roundId", "answer", "startedAt", "updatedAt", "answeredInRound")
    names: dict[str, str] = {}
    for key, part in zip(order, match.group(1).split(","), strict=False):
        tokens = part.split()
        if tokens and re.fullmatch(r"[A-Za-z_]\w*", tokens[-1]) and len(tokens) >= 2:
            names[key] = tokens[-1]
    return names


def _freshness(body: str, names: dict[str, str]) -> bool:
    updated = names.get("updatedAt", "")
    if updated and re.search(
        rf"block\.timestamp[^;]*\b{updated}\b|\b{updated}\b[^;]*block\.timestamp", body
    ):
        return True
    return bool(
        re.search(
            r"block\.timestamp\s*-\s*\w+[^;]*(<|<=|>|>=)\s*\w*(stale|heartbeat|maxAge|delay|MAX)",
            body,
            re.I,
        )
    )


def _positivity(body: str, names: dict[str, str], variables: tuple[str, ...]) -> bool:
    for name in {*variables, names.get("answer", "")}:
        if name and re.search(
            rf"\b{name}\b\s*(>|<=|<|>=|==|!=)\s*0\b|0\s*(<|>=|>|<=)\s*\b{name}\b", body
        ):
            return True
    return False


def _aggregation(model: ResearchModel, function: RFunction, facts: PriceFacts) -> None:
    body = function.body
    loop = re.search(r"\bfor\s*\([^)]*\)\s*\{", body)
    indexed = re.search(
        r"\b\w+\s*\[\s*\w+\s*\]\s*\.\s*\w+\s*\(|\(\s*\w+\s*\[\s*\w+\s*\]\s*\)\s*\.\s*\w+", body
    )
    facts.aggregated = bool(loop and indexed and facts.sources)
    if not facts.aggregated:
        return
    facts.tolerant = bool(
        re.search(r"\bcatch\b[^{]*\{[^}]*\bcontinue\b", body)
        or re.search(
            r"\b(if|require)\s*\([^)]*(price|answer|value|ok|success)[^)]*\)\s*\{?\s*(continue|valid\w*\s*\+\+|\+\+\s*valid)",
            body,
        )
        or re.search(r"if\s*\([^)]*(==\s*0|<=\s*0|!ok|!success)[^)]*\)\s*(\{\s*)?continue", body)
        or re.search(r"\btry\b", body)
    )
    counters = re.findall(r"\b(valid\w*|count\w*|ok\w*|live\w*|n)\s*(?:\+\+|\+=\s*1)", body)
    counters += re.findall(r"(?:\+\+)\s*(valid\w*|count\w*|ok\w*|live\w*)", body)
    for counter in dict.fromkeys(counters):
        compare = re.search(
            rf"(require|if)\s*\(\s*{counter}\s*(>=|>|<|<=|==)\s*([A-Za-z_]\w*|\d+)", body
        )
        if compare:
            facts.quorum_checked = True
            facts.quorum_min = _quorum_value(model, function, compare.group(3), compare.group(2))
            break
    facts.aggregation = (
        "median"
        if re.search(r"median|sort", body, re.I)
        else "average"
        if re.search(r"/\s*(valid\w*|count\w*|n)\b", body)
        else "other"
    )


def _quorum_value(
    model: ResearchModel, function: RFunction, token: str, operator: str
) -> int | None:
    if token.isdigit():
        number = int(token)
    else:
        text = "\n".join(
            model.contracts[name].body
            for name in model.lineage(function.contract)
            if name in model.contracts
        )
        found = re.search(rf"\b{re.escape(token)}\b\s*(?:=|:=)\s*(\d+)\s*;", text)
        if not found:
            return None
        number = int(found.group(1))
    return number + 1 if operator == ">" else number


def _fallback(function: RFunction, facts: PriceFacts) -> None:
    if len(facts.sources) < 2 or facts.aggregated:
        return
    first, second = facts.sources[0], facts.sources[1]
    if first.receiver == second.receiver:
        return
    body = function.body
    primary_segment = body[: second.offset]
    fallback_segment = body[second.offset :]
    if not re.search(r"\bcatch\b|\belse\b|\bif\b", primary_segment + fallback_segment):
        return
    names = _round_names(body)
    primary = int(_freshness(primary_segment, names)) + int(
        _positivity(primary_segment, names, facts.price_variables)
    )
    secondary = int(_freshness(fallback_segment, names)) + int(
        _positivity(fallback_segment, names, facts.price_variables)
    )
    if primary > secondary:
        facts.fallback_weaker = (
            f"primary read is validated by {primary} control(s), fallback by {secondary}"
        )


def _consumers(model: ResearchModel, producer: RFunction) -> list[tuple[RFunction, list[str]]]:
    found: list[tuple[RFunction, list[str]]] = []
    for function in model.all_functions():
        if function is producer or not function.has_body:
            continue
        calls = any(
            name == producer.name
            and function.contract in {producer.contract, *model.lineage(function.contract)}
            for name, _args, _ in plain_calls(function.body)
        ) or any(
            call.name == producer.name
            and model.contract_for_type(
                model.state_vars(function.contract).get(
                    re.sub(r"\(.*\)|\[.*\]", "", call.receiver), ""
                )
                or re.sub(r"\(.*\)", "", call.receiver)
            )
            == producer.contract
            for call in member_calls(function.body)
        )
        if not calls:
            continue
        hints = _use_hints(function)
        found.append((function, hints))
    return sorted(found, key=lambda item: (item[0].file, item[0].line))[:8]


def _use_hints(function: RFunction) -> list[str]:
    text = f"{function.name} {' '.join(sorted(set(re.findall(r'[A-Za-z_]\w*', function.body))))}"
    hints: list[str] = []
    for needle, label in _USE_HINTS:
        if re.search(needle, text, re.I) and label not in hints:
            hints.append(label)
    return hints


def _candidates(
    model: ResearchModel, facts: PriceFacts, consumers: list[tuple[RFunction, list[str]]]
) -> list[SemanticCandidate]:
    function = facts.function
    consumer, hints = next(iter(consumers), (None, []))
    sensitive = bool(hints)
    path = [f"{function.contract}.{function.signature} @{function.file}:{function.line}"]
    if consumer is not None:
        path.insert(0, f"{consumer.contract}.{consumer.signature} @{consumer.file}:{consumer.line}")
    base_facts = (
        ("sources", str(len(facts.sources))),
        ("source_kinds", ",".join(sorted(facts.kinds))),
        ("aggregated", str(facts.aggregated).lower()),
        ("quorum_minimum", "unknown" if facts.quorum_min is None else str(facts.quorum_min)),
        ("use", ",".join(hints) if hints else "unknown"),
        ("use_basis", "identifier hints in the consuming function"),
    )
    confidence = "medium" if sensitive else "low"
    tags = (
        "oracle_dependent_value",
        "collateral_valuation" if "collateral_valuation" in hints else "reserve_accounting",
    )
    found: list[SemanticCandidate] = []

    def make(
        detector: str,
        title: str,
        summary: str,
        missing: tuple[str, ...],
        observed: tuple[str, ...],
        line: int | None = None,
    ) -> SemanticCandidate:
        return SemanticCandidate(
            detector=f"oracle.{detector}",
            family=FAMILY,
            title=title,
            summary=summary,
            file=function.file,
            line=line or function.line,
            contract=function.contract,
            function=function.signature,
            path=tuple(path),
            facts=base_facts,
            observed=observed,
            missing=missing,
            confidence=confidence,
            impact_tags=tags,
        )

    use_text = f" It feeds {', '.join(hints)}." if hints else ""
    if (
        facts.aggregated
        and facts.tolerant
        and (not facts.quorum_checked or (facts.quorum_min is not None and facts.quorum_min <= 1))
    ):
        reason = (
            "no minimum count of valid sources is enforced"
            if not facts.quorum_checked
            else f"the enforced minimum is {facts.quorum_min} valid source"
        )
        spot = (
            " Spot-reserve sources are included."
            if "spot_reserves" in facts.kinds or "spot_slot0" in facts.kinds
            else ""
        )
        found.append(
            make(
                "insufficient_quorum",
                "Oracle aggregation accepts too few independent sources",
                f"{function.contract}.{function.name} aggregates {len(facts.sources)} source "
                f"reads, skips sources that fail or return zero, and {reason}, so one remaining "
                f"source defines the price.{spot}{use_text}",
                ("minimum count of independent valid sources greater than one",),
                (
                    "price aggregated across a source list",
                    "failed or zero sources are skipped",
                ),
            )
        )
    if "chainlink" in facts.kinds and not facts.freshness:
        found.append(
            make(
                "stale_source_accepted",
                "Price source accepted without a freshness check",
                f"{function.contract}.{function.name} reads a round-based feed and never compares "
                f"its update time with block.timestamp.{use_text}",
                ("updatedAt freshness bound",),
                ("round data is read",),
            )
        )
    if "chainlink" in facts.kinds and not facts.positivity and not facts.fallback_weaker:
        found.append(
            make(
                "price_not_validated",
                "Returned price is used without zero or negative handling",
                f"{function.contract}.{function.name} reads a signed feed answer without "
                f"checking that it is positive.{use_text}",
                ("answer > 0 check",),
                ("signed answer is read",),
            )
        )
    if facts.fallback_weaker:
        found.append(
            make(
                "fallback_weaker",
                "Fallback price path is validated less than the primary path",
                f"{function.contract}.{function.name} has a primary and a fallback read: "
                f"{facts.fallback_weaker}.{use_text}",
                ("the same validation on the fallback source",),
                ("two distinct sources are read",),
            )
        )
    spot_kinds = facts.kinds & {"spot_reserves", "spot_slot0"}
    if (
        spot_kinds
        and not (facts.kinds & {"twap", "chainlink"})
        and not facts.deviation
        and sensitive
    ):
        found.append(
            make(
                "spot_price_sensitive_use",
                "Spot pool price used for a security-sensitive valuation",
                f"{function.contract}.{function.name} derives a price from a pool's current "
                f"state with no time-weighted source or deviation bound.{use_text}",
                ("time-weighted or independently validated price",),
                ("spot reserves or slot0 are read",),
            )
        )
    if facts.aggregated and facts.sources and facts.kinds <= {"spot_reserves", "spot_slot0"}:
        found.append(
            make(
                "source_independence_not_established",
                "Aggregated sources are all manipulable spot prices",
                f"Every source read by {function.contract}.{function.name} is a pool spot price, "
                f"so the sources can be moved together in one block.{use_text}",
                ("a source that cannot be moved by the same actor in the same block",),
                ("all sources are spot prices",),
            )
        )
    if (
        facts.aggregated
        and len(facts.sources) >= 2
        and not facts.decimals
        and "chainlink" in facts.kinds
    ):
        found.append(
            make(
                "decimals_not_normalized",
                "Source answers are aggregated without decimals normalization",
                f"{function.contract}.{function.name} combines several feed answers and never "
                "reads their decimals.",
                ("decimals() normalization before aggregation",),
                ("several feeds are aggregated",),
            )
        )
    del model
    return found


__all__ = ["PriceFacts", "analyze_oracle_quality", "mentions"]
