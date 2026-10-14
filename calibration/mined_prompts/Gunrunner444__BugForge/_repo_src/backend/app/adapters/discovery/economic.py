"""In-process economic observations.

The engine does not start a process, open a network connection, or call a
model. Missing observations stay unsupported. A result is evidence, not
authority, and it cannot mark a finding verified.
"""

from __future__ import annotations

from collections.abc import Callable

from app.discovery.capabilities import EngineAvailability, EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.results import DynamicResult
from app.parsing.solidity_economics import (
    AssetDelta,
    Conversion,
    ConversionKind,
    EconomicEvidence,
    EconomicResult,
    OracleStatus,
    Quantity,
    amm_invariant,
    bind_economic_evidence,
    donation_inflation,
    erc20_delta,
    fee_on_transfer,
    flash_position,
    lending_transition,
    native_eth_delta,
    net_change,
    oracle_dependency,
    parse_integer,
    preview_discrepancy,
    reserve_delta,
    share_delta,
)

_VERSION = "phase46"


class EconomicEngine(DiscoveryEngine):
    @property
    def engine_id(self) -> str:
        return "bugforge-economic"

    @property
    def display_name(self) -> str:
        return "BugForge economic analysis"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset({"solidity"})

    def capabilities(self) -> frozenset[EngineCapability]:
        return frozenset({EngineCapability.ECONOMIC_SIMULATION, EngineCapability.RESULTS_INGESTION})

    def availability(self) -> EngineAvailability:
        return EngineAvailability.AVAILABLE

    def version(self) -> str:
        return _VERSION

    def _start_campaign(self, request: AnalysisRequest) -> DynamicResult:
        return _observe(self, request)

    def _analyze_target(self, request: AnalysisRequest) -> DynamicResult:
        return _observe(self, request)


def _observe(engine: EconomicEngine, request: AnalysisRequest) -> DynamicResult:
    case = request.extra.get("case", "")
    if not case:
        return DynamicResult(
            engine=engine.engine_id,
            engine_version=engine.version(),
            language=request.language,
            target=request.target,
            status=ResultStatus.UNSUPPORTED,
            executed=False,
            provenance="economic-calculation",
            oracle_explanation="no economic observations were supplied",
            oracle_kind="economic",
            metadata=_meta("unsupported", bound=False),
        )
    result, malformed = _dispatch(request.extra)
    if malformed:
        result = EconomicResult(
            OracleStatus.UNKNOWN.value,
            "a numeric observation was malformed and was not treated as a value",
            ("malformed input is unknown",),
        )
    evidence = _bind(engine, request, result)
    if not evidence.bound:
        result = EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "the observation is not bound to project, source, compiler, and sequence identity",
            (*result.assumptions, "caller-supplied values are not runtime evidence"),
            deltas=result.deltas,
            relation=result.relation,
            conversion=result.conversion,
        )
        evidence = bind_economic_evidence(
            EconomicEvidence(
                evidence.project,
                evidence.target,
                evidence.sequence_id,
                evidence.transaction_indexes,
                evidence.actors,
                evidence.tokens,
                evidence.initial_state,
                evidence.final_state,
                result.deltas,
                evidence.conversion_source,
                evidence.oracle_source,
                evidence.invariant,
                result.status,
                result.assumptions,
                evidence.source_locations,
                evidence.engine,
                evidence.tool_version,
                evidence.environment,
                evidence.source_snapshot,
                evidence.compiler_configuration,
                evidence.contract,
                evidence.function_identity,
                "externally-supplied",
            )
        )
    return DynamicResult(
        engine=engine.engine_id,
        engine_version=engine.version(),
        language=request.language,
        target=request.target,
        campaign_id=request.campaign_id,
        source_file=request.source_file,
        contract=request.contract,
        function=request.function,
        status=ResultStatus.INGESTED,
        executed=False,
        provenance="economic-calculation",
        oracle_explanation=result.explanation,
        oracle_kind="economic",
        metadata=_meta(result.status, bound=evidence.bound, calculation=evidence.observation_class),
        economic_evidence=evidence,
    )


def _meta(status: str, *, bound: bool, calculation: str = "economic-calculation") -> dict[str, str]:
    return {
        "verified": "false",
        "economic_status": status,
        "evidence_class": "economic",
        "environment": "in-process",
        "bound": str(bound).lower(),
        "observation_class": calculation,
        "input_provenance": "externally-supplied",
    }


def _dispatch(extra: dict[str, str]) -> tuple[EconomicResult, bool]:
    case = extra.get("case", "")
    if case == "eth":
        return _pair(extra, native_eth_delta, "eth", "wei")
    if case == "erc20":
        return _pair(extra, erc20_delta, "erc20", extra.get("unit", "token"))
    if case == "share":
        return _pair(extra, share_delta, "share", "shares")
    if case == "reserve":
        return _pair(extra, reserve_delta, "reserve", extra.get("unit", "token"))
    if case == "net":
        values, malformed = _numbers(
            extra, ("final", "initial", "costs", "fees", "repayments"), signed=False
        )
        if malformed:
            return _unsupported(), True
        return net_change(
            final=values["final"],
            initial=values["initial"],
            costs=values["costs"],
            fees=values["fees"],
            repayments=values["repayments"],
            token=extra.get("token", ""),
        ), False
    if case == "flash":
        values, malformed = _numbers(
            extra, ("owned_initial", "owned_final", "borrowed", "repayment", "fee"), signed=False
        )
        if malformed:
            return _unsupported(), True
        return flash_position(
            owned_initial=values["owned_initial"],
            owned_final=values["owned_final"],
            borrowed=values["borrowed"],
            repayment=values["repayment"],
            fee=values["fee"],
            token=extra.get("token", ""),
        ), False
    if case == "donation":
        required, malformed = _required(
            extra, ("assets_before", "shares_before", "donated", "deposit_assets", "shares_minted")
        )
        if malformed:
            return _unsupported(), True
        return donation_inflation(
            assets_before=required["assets_before"],
            shares_before=required["shares_before"],
            donated=required["donated"],
            deposit_assets=required["deposit_assets"],
            shares_minted=required["shares_minted"],
            established=frozenset(extra.get("established", "").split(",")) - {""},
            observed=extra.get("observed") == "true",
        ), False
    if case == "preview":
        preview_text = extra.get("preview", "")
        executed_text = extra.get("executed", "")
        required, malformed = _required_raw(
            {"preview": preview_text, "executed_amount": executed_text},
            ("preview", "executed_amount"),
        )
        if malformed:
            return _unsupported(), True
        return preview_discrepancy(
            preview=required["preview"],
            executed=required["executed_amount"],
            established=frozenset(extra.get("established", "").split(",")) - {""},
            operation=extra.get("operation", ""),
        ), False
    if case == "fee":
        values, malformed = _numbers(extra, ("requested",), signed=False)
        deltas, delta_bad = _numbers(extra, ("sender_delta", "receiver_delta"), signed=True)
        accounted, accounted_bad = _numbers(extra, ("accounted",), signed=False)
        if malformed or delta_bad or accounted_bad:
            return _unsupported(), True
        if (
            values["requested"] is None
            or deltas["sender_delta"] is None
            or deltas["receiver_delta"] is None
        ):
            return EconomicResult(
                OracleStatus.INCOMPLETE.value, "transfer quantities are incomplete"
            ), False
        return fee_on_transfer(
            requested=values["requested"],
            sender_delta=deltas["sender_delta"],
            receiver_delta=deltas["receiver_delta"],
            accounted=accounted["accounted"],
            semantics=extra.get("semantics", "unknown"),
        ), False
    if case == "oracle":
        return oracle_dependency(
            state_changed=extra.get("state_changed") == "true",
            freshness=extra.get("freshness", "unknown"),
            design=extra.get("design", "unknown"),
            input_relation=extra.get("input_relation") == "true",
            valued=extra.get("valued") == "true",
            sensitive=extra.get("sensitive") == "true",
        ), False
    if case == "amm":
        values, malformed = _numbers(extra, ("reserve0", "reserve1"), signed=False)
        if malformed:
            return _unsupported(), True
        return amm_invariant(
            reserve0=values["reserve0"],
            reserve1=values["reserve1"],
            model=extra.get("model", ""),
        ), False
    if case == "lending":
        values, malformed = _numbers(
            extra,
            (
                "debt_before",
                "debt_after",
                "repayment",
                "collateral_before",
                "collateral_after",
                "principal",
                "interest",
                "protocol_fee",
                "bad_debt",
            ),
            signed=False,
        )
        if malformed:
            return _unsupported(), True
        return lending_transition(
            debt_before=values["debt_before"],
            debt_after=values["debt_after"],
            repayment=values["repayment"],
            collateral_before=values["collateral_before"],
            collateral_after=values["collateral_after"],
            health=extra.get("health", "unknown"),
            liquidated=extra.get("liquidated") == "true",
            debt_semantics=extra.get("debt_semantics", "unknown"),
            principal=values["principal"],
            interest=values["interest"],
            protocol_fee=values["protocol_fee"],
            bad_debt=values["bad_debt"],
        ), False
    if case == "convert":
        values, malformed = _numbers(
            extra,
            (
                "final",
                "initial",
                "costs",
                "fees",
                "repayments",
                "numerator",
                "denominator",
                "other_amount",
            ),
            signed=False,
        )
        if malformed:
            return _unsupported(), True
        source = Conversion(
            kind=extra.get("conversion_kind", ConversionKind.UNSUPPORTED.value),
            from_token=extra.get("from_token", ""),
            to_token=extra.get("to_token", ""),
            numerator=values["numerator"],
            denominator=values["denominator"],
            provenance=extra.get("conversion_provenance", ""),
        )
        return net_change(
            final=values["final"],
            initial=values["initial"],
            costs=values["costs"],
            fees=values["fees"],
            repayments=values["repayments"],
            token=extra.get("token", ""),
            other_token=extra.get("other_token", ""),
            other_amount=values["other_amount"],
            conversion=source,
        ), False
    return _unsupported(), False


def _unsupported() -> EconomicResult:
    return EconomicResult(OracleStatus.UNSUPPORTED.value, "the economic case is unsupported")


def _bind(
    engine: EconomicEngine, request: AnalysisRequest, result: EconomicResult
) -> EconomicEvidence:
    raw_index = request.extra.get("transaction_index", "")
    index = parse_integer(raw_index, signed=False) if raw_index else None
    actors = tuple(part for part in request.extra.get("actor", "").split(",") if part)
    tokens = tuple(part for part in request.extra.get("token", "").split(",") if part)
    locations = (request.source_file,) if request.source_file else ()
    evidence = EconomicEvidence(
        str(request.repo_root),
        request.target,
        request.extra.get("sequence_id", ""),
        (index,) if index is not None else (),
        actors,
        tokens,
        request.extra.get("initial_state", ""),
        request.extra.get("final_state", ""),
        result.deltas,
        result.conversion or request.extra.get("conversion_provenance", ""),
        request.extra.get("oracle_source", ""),
        result.relation,
        result.status,
        result.assumptions,
        locations,
        engine.engine_id,
        engine.version(),
        "in-process",
        request.extra.get("source_snapshot", ""),
        request.extra.get("compiler_configuration", ""),
        request.contract,
        request.function,
        "economic-calculation",
    )
    return bind_economic_evidence(evidence, runtime_established=False)


def _pair(
    extra: dict[str, str],
    function: Callable[[Quantity, Quantity], AssetDelta],
    kind: str,
    unit: str,
) -> tuple[EconomicResult, bool]:
    before, before_bad = _quantity(extra, "before", kind, unit)
    after, after_bad = _quantity(extra, "after", kind, unit)
    if before_bad or after_bad:
        return EconomicResult(
            OracleStatus.UNKNOWN.value, "the balance observation is malformed"
        ), True
    if before is None or after is None:
        return EconomicResult(
            OracleStatus.UNKNOWN.value, "the balance observation is incomplete"
        ), False
    delta = function(before, after)
    if delta.known == "unknown":
        return EconomicResult(
            OracleStatus.UNKNOWN.value, "the delta is unknown", deltas=(delta,)
        ), False
    if delta.known == "positive":
        status = OracleStatus.POTENTIAL_POSITIVE_DELTA.value
    elif delta.known == "negative":
        status = OracleStatus.POTENTIAL_LOSS.value
    else:
        status = OracleStatus.BALANCED.value
    return EconomicResult(status, f"{kind} delta {delta.delta}", deltas=(delta,)), False


def _quantity(
    extra: dict[str, str], prefix: str, kind: str, unit: str
) -> tuple[Quantity | None, bool]:
    raw = extra.get(prefix, "")
    if raw == "":
        return None, False
    amount = parse_integer(raw, signed=False)
    if amount is None:
        return None, True
    index = parse_integer(extra.get("transaction_index", "0"), signed=False)
    if index is None:
        return None, True
    return Quantity(
        token=extra.get("token", ""),
        unit=unit,
        actor=extra.get("actor", ""),
        amount=amount,
        source="externally-supplied",
        transaction_index=index,
        snapshot=prefix,
        provenance="externally-supplied",
        confidence="observed",
        kind=kind,
    ), False


def _numbers(
    extra: dict[str, str], names: tuple[str, ...], *, signed: bool
) -> tuple[dict[str, int | None], bool]:
    found: dict[str, int | None] = {}
    for name in names:
        raw = extra.get(name, "")
        if raw == "" or raw == "unknown":
            found[name] = None
            continue
        value = parse_integer(raw, signed=signed)
        if value is None:
            return found, True
        found[name] = value
    return found, False


def _required(extra: dict[str, str], names: tuple[str, ...]) -> tuple[dict[str, int], bool]:
    return _required_raw(extra, names)


def _required_raw(extra: dict[str, str], names: tuple[str, ...]) -> tuple[dict[str, int], bool]:
    found: dict[str, int] = {}
    for name in names:
        value = parse_integer(extra.get(name, ""), signed=False)
        if value is None:
            return found, True
        found[name] = value
    return found, False
