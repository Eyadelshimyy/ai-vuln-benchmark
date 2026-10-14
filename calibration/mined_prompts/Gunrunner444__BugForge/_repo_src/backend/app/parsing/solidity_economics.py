"""Bounded economic observations for local sequences.

This is not a protocol simulator, an EVM, or an exploit generator. A positive
delta is an observation. It is not confirmation, and it is not a vulnerability
status. Cross-asset value is computed only from an explicit conversion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum

_UINT256 = 2**256
_INTEGER = re.compile(r"[+-]?\d+")

_FORBIDDEN = frozenset(
    {"safe", "verified", "proved", "exploited", "confirmed", "profitable", "unprofitable"}
)


class OracleStatus(StrEnum):
    POTENTIAL_POSITIVE_DELTA = "potential_positive_delta"
    POTENTIAL_LOSS = "potential_loss"
    INVARIANT_VIOLATION = "invariant_violation"
    BALANCED = "balanced"
    NON_PROFITABLE = "non_profitable"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"


_ALLOWED = frozenset(item.value for item in OracleStatus)


class ConversionKind(StrEnum):
    EXACT_LOCAL = "exact_local"
    PROTOCOL = "protocol"
    FIXED_BENCHMARK = "fixed_benchmark"
    LOCAL_ORACLE = "local_oracle"
    UNSUPPORTED = "unsupported"


def canonical_status(status: str) -> str:
    if status in _FORBIDDEN or status not in _ALLOWED:
        return OracleStatus.UNKNOWN.value
    return status


def parse_integer(raw: object, *, signed: bool = False) -> int | None:
    """Parse one integer. Malformed, negative, or overflow-like input is unknown."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not _INTEGER.fullmatch(text):
            return None
        try:
            value = int(text)
        except ValueError:
            return None
    else:
        return None
    if value < 0 and not signed:
        return None
    if abs(value) >= _UINT256:
        return None
    return value


@dataclass(frozen=True)
class Quantity:
    token: str
    unit: str
    actor: str
    amount: int | None
    source: str
    transaction_index: int
    snapshot: str
    provenance: str
    confidence: str
    assumptions: tuple[str, ...] = ()
    kind: str = "balance"


@dataclass(frozen=True)
class EconomicSnapshot:
    snapshot_id: str
    quantities: tuple[Quantity, ...]
    caller: str = ""
    transaction_index: int = 0
    transaction_value: int | None = None


@dataclass(frozen=True)
class AssetDelta:
    kind: str
    token: str
    unit: str
    actor: str
    before: int | None
    after: int | None
    delta: int | None
    known: str
    transaction_index: int | None = None
    snapshot: str = ""
    contract: str = ""
    source: str = ""


@dataclass(frozen=True)
class Conversion:
    kind: str
    from_token: str
    to_token: str
    numerator: int | None
    denominator: int | None
    provenance: str


@dataclass(frozen=True)
class EconomicResult:
    status: str
    explanation: str
    assumptions: tuple[str, ...] = ()
    deltas: tuple[AssetDelta, ...] = ()
    conversion: str = ""
    relation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", canonical_status(self.status))


@dataclass(frozen=True)
class EconomicEvidence:
    project: str
    target: str
    sequence_id: str
    transaction_indexes: tuple[int, ...]
    actors: tuple[str, ...]
    tokens: tuple[str, ...]
    initial_state: str
    final_state: str
    deltas: tuple[AssetDelta, ...]
    conversion_source: str
    oracle_source: str
    invariant: str
    status: str
    assumptions: tuple[str, ...]
    source_locations: tuple[str, ...]
    engine: str
    tool_version: str
    environment: str
    source_snapshot: str
    compiler_configuration: str
    contract: str = ""
    function_identity: str = ""
    observation_class: str = "economic-calculation"
    bound: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", canonical_status(self.status))


_OBSERVATION_CLASSES = frozenset(
    {
        "runtime-observation",
        "externally-supplied",
        "static-semantic",
        "economic-calculation",
    }
)
_BINDING_FIELDS = (
    "project",
    "target",
    "sequence_id",
    "source_snapshot",
    "compiler_configuration",
    "contract",
    "engine",
    "tool_version",
    "environment",
    "initial_state",
    "final_state",
)


def bind_economic_evidence(
    evidence: EconomicEvidence, *, runtime_established: bool = False
) -> EconomicEvidence:
    """Downgrade observations that are not bound. Caller text is not runtime evidence."""
    kind = evidence.observation_class
    if kind not in _OBSERVATION_CLASSES:
        kind = "externally-supplied"
    if kind == "runtime-observation" and not runtime_established:
        kind = "externally-supplied"
    missing = [name for name in _BINDING_FIELDS if not str(getattr(evidence, name) or "").strip()]
    if evidence.function_identity == "" and evidence.contract == "":
        missing.append("contract")
    if not evidence.actors or not evidence.tokens:
        missing.append("actor-or-token")
    if not evidence.source_locations:
        missing.append("source_locations")
    if evidence.deltas and not evidence.transaction_indexes:
        missing.append("transaction_indexes")
    assumptions = evidence.assumptions
    status = evidence.status
    bound = False
    if missing:
        status = OracleStatus.INCOMPLETE.value
        note = "caller-supplied values are not runtime evidence"
        if note not in assumptions:
            assumptions = (*assumptions, note)
    else:
        bound = True
        note = "an economic calculation is not verification"
        if note not in assumptions:
            assumptions = (*assumptions, note)
    return replace(
        evidence,
        observation_class=kind,
        bound=bound,
        status=canonical_status(status),
        assumptions=assumptions,
    )


def snapshot_of(*quantities: Quantity, snapshot_id: str, caller: str = "") -> EconomicSnapshot:
    return EconomicSnapshot(
        snapshot_id, quantities, caller, quantities[0].transaction_index if quantities else 0
    )


def quantity_delta(before: Quantity, after: Quantity, *, kind: str) -> AssetDelta:
    if before.token != after.token or before.unit != after.unit or before.actor != after.actor:
        return AssetDelta(
            kind,
            before.token,
            before.unit,
            before.actor,
            before.amount,
            after.amount,
            None,
            "unknown",
        )
    if before.amount is None or after.amount is None:
        return AssetDelta(
            kind,
            before.token,
            before.unit,
            before.actor,
            before.amount,
            after.amount,
            None,
            "unknown",
        )
    delta = after.amount - before.amount
    known = "zero" if delta == 0 else "positive" if delta > 0 else "negative"
    return AssetDelta(
        kind, before.token, before.unit, before.actor, before.amount, after.amount, delta, known
    )


def native_eth_delta(before: Quantity, after: Quantity) -> AssetDelta:
    return quantity_delta(before, after, kind="eth")


def erc20_delta(before: Quantity, after: Quantity) -> AssetDelta:
    return quantity_delta(before, after, kind="erc20")


def share_delta(before: Quantity, after: Quantity) -> AssetDelta:
    return quantity_delta(before, after, kind="share")


def debt_delta(before: Quantity, after: Quantity) -> AssetDelta:
    return quantity_delta(before, after, kind="debt")


def reserve_delta(before: Quantity, after: Quantity) -> AssetDelta:
    return quantity_delta(before, after, kind="reserve")


def fee_delta(before: Quantity, after: Quantity) -> AssetDelta:
    return quantity_delta(before, after, kind="fee")


def actor_delta(before: Quantity, after: Quantity) -> AssetDelta:
    return quantity_delta(before, after, kind="actor")


def apply_conversion(amount: int | None, source: Conversion | None, token: str) -> int | None:
    """Convert only through an explicit source. Missing prices stay unknown."""
    if amount is None:
        return None
    if source is None or source.kind == ConversionKind.UNSUPPORTED.value:
        return None
    if (
        source.from_token != token
        or source.numerator is None
        or source.numerator < 0
        or source.denominator is None
        or source.denominator <= 0
    ):
        return None
    return amount * source.numerator // source.denominator


def net_change(
    *,
    final: int | None,
    initial: int | None,
    costs: int | None,
    fees: int | None,
    repayments: int | None,
    token: str,
    other_token: str = "",
    other_amount: int | None = None,
    conversion: Conversion | None = None,
) -> EconomicResult:
    """final - initial - costs - fees - repayments, in one token unit."""
    converted = 0
    conversion_note = ""
    if other_token and other_token != token:
        converted_amount = apply_conversion(other_amount, conversion, other_token)
        if converted_amount is None:
            return EconomicResult(
                OracleStatus.UNKNOWN.value,
                "cross-asset value is unknown without an explicit conversion",
                ("no market price was invented",),
                conversion=conversion.kind if conversion else "absent",
            )
        converted = converted_amount
        conversion_note = conversion.kind if conversion else ""
    parts = (final, initial, costs, fees, repayments)
    if any(part is None for part in parts):
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "a cost, fee, or balance was not established",
            ("incomplete inputs are not profit or loss",),
        )
    assert final is not None and initial is not None
    assert costs is not None and fees is not None and repayments is not None
    net = final + converted - initial - costs - fees - repayments
    return _signed(net, token, conversion_note)


def flash_position(
    *,
    owned_initial: int | None,
    owned_final: int | None,
    borrowed: int | None,
    repayment: int | None,
    fee: int | None,
    token: str,
) -> EconomicResult:
    """Borrowed principal is an obligation, not attacker-owned profit."""
    assumptions = (
        "borrowed principal is not profit",
        "not a vulnerability confirmation",
    )
    if borrowed is None:
        return EconomicResult(OracleStatus.UNKNOWN.value, "borrowed amount is unknown", assumptions)
    if repayment is None:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "repayment is not established; the borrowed amount remains an unsettled obligation",
            assumptions,
            relation="debt",
        )
    if owned_initial is None or owned_final is None or fee is None:
        return EconomicResult(
            OracleStatus.UNKNOWN.value, "owned balance or fee is unknown", assumptions
        )
    net = owned_final - owned_initial - repayment - fee
    result = _signed(net, token, "")
    return EconomicResult(result.status, result.explanation, assumptions, relation="flash")


def erc4626_rounding(
    *,
    assets: int,
    total_assets: int,
    supply: int,
    direction: str,
    established: frozenset[str],
) -> EconomicResult:
    if "erc4626-deposit" not in established:
        return EconomicResult(
            OracleStatus.UNSUPPORTED.value,
            "ERC-4626 deposit semantics were not established",
        )
    if assets < 0 or total_assets < 0 or supply < 0:
        return EconomicResult(OracleStatus.UNKNOWN.value, "a negative vault quantity is unknown")
    if total_assets == 0 or supply == 0 or assets == 0:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "zero-share or zero-asset transition",
            ("a zero transition is not an exploit",),
        )
    if direction == "down":
        expected = assets * supply // total_assets
    elif direction == "up":
        expected = (assets * supply + total_assets - 1) // total_assets
    else:
        return EconomicResult(OracleStatus.UNKNOWN.value, "rounding direction is unknown")
    return EconomicResult(
        OracleStatus.BALANCED.value,
        f"established {direction} rounding expects {expected} shares",
        ("the vault is not assumed compliant beyond this relation",),
        relation="assets/shares",
    )


# preview MUST be no more than the exact share or asset amount.
_PREVIEW_AT_MOST = frozenset(
    {"previewDeposit", "previewRedeem", "convertToShares", "convertToAssets"}
)
# preview MUST be no fewer than the exact asset amount.
_PREVIEW_AT_LEAST = frozenset({"previewMint", "previewWithdraw"})


def preview_discrepancy(
    *,
    preview: int,
    executed: int,
    established: frozenset[str],
    operation: str = "",
) -> EconomicResult:
    """Compare one ERC-4626 preview with execution. Direction depends on the operation."""
    if "erc7540" in established or "async-vault" in established:
        return EconomicResult(
            OracleStatus.UNSUPPORTED.value,
            "asynchronous vault semantics are outside ordinary ERC-4626 assumptions",
        )
    if "erc4626-preview" not in established and operation not in established:
        return EconomicResult(
            OracleStatus.UNSUPPORTED.value,
            "preview semantics were not established",
        )
    if preview < 0 or executed < 0:
        return EconomicResult(OracleStatus.UNKNOWN.value, "a negative preview quantity is unknown")
    if operation not in _PREVIEW_AT_MOST and operation not in _PREVIEW_AT_LEAST:
        return EconomicResult(
            OracleStatus.UNKNOWN.value,
            "the preview operation is not established; a numeric mismatch is not an invariant",
        )
    assumptions = ("a directional mismatch is not automatically exploitable",)
    relation = operation
    if preview == executed:
        return EconomicResult(
            OracleStatus.BALANCED.value,
            f"{operation} matches execution",
            assumptions,
            relation=relation,
        )
    if operation in _PREVIEW_AT_MOST and preview > executed:
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            f"{operation} exceeds the executed amount",
            assumptions,
            relation=relation,
        )
    if operation in _PREVIEW_AT_LEAST and preview < executed:
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            f"{operation} is below the executed amount",
            assumptions,
            relation=relation,
        )
    return EconomicResult(
        OracleStatus.INCOMPLETE.value,
        f"{operation} is conservative relative to execution",
        assumptions,
        relation=relation,
    )


def donation_inflation(
    *,
    assets_before: int,
    shares_before: int,
    donated: int,
    deposit_assets: int,
    shares_minted: int,
    established: frozenset[str],
    observed: bool,
) -> EconomicResult:
    if "donation" not in established and "erc4626-deposit" not in established:
        return EconomicResult(
            OracleStatus.UNSUPPORTED.value, "donation semantics were not established"
        )
    if not observed:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "donation effect has no execution observation",
            ("a static pattern is not exploitability",),
        )
    if shares_before <= 0 or assets_before <= 0:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value, "initial share price is not established"
        )
    before = deposit_assets * shares_before // assets_before
    after_assets = assets_before + donated
    after = deposit_assets * shares_before // after_assets
    if shares_minted < before and shares_minted <= after:
        return EconomicResult(
            OracleStatus.POTENTIAL_LOSS.value,
            "an external asset increase reduced shares minted for the same deposit",
            ("candidate donation effect; not an exploit confirmation",),
            relation="donation/share-conversion",
        )
    return EconomicResult(
        OracleStatus.BALANCED.value, "share conversion did not move with the donation"
    )


def fee_on_transfer(
    *,
    requested: int,
    sender_delta: int,
    receiver_delta: int,
    accounted: int | None,
    semantics: str,
) -> EconomicResult:
    """Require a sender decrease and a receiver increase. A function name is not semantics."""
    if semantics in {"", "unknown"}:
        return EconomicResult(
            OracleStatus.UNKNOWN.value,
            "token transfer semantics are unknown; amount sent is not assumed received",
        )
    if semantics in {"custom", "unsupported"}:
        return EconomicResult(
            OracleStatus.UNSUPPORTED.value,
            "custom transfer behavior is unsupported",
        )
    if semantics not in {"fee-on-transfer", "exact"}:
        return EconomicResult(OracleStatus.UNSUPPORTED.value, "transfer semantics are unsupported")
    if requested < 0:
        return EconomicResult(OracleStatus.UNKNOWN.value, "a negative requested amount is unknown")
    if sender_delta >= 0 or receiver_delta <= 0:
        return EconomicResult(
            OracleStatus.UNKNOWN.value,
            "a sender decrease must be negative and a receiver increase must be positive",
            ("a reversed delta is not a transfer observation",),
            relation="transfer-direction",
        )
    sent = -sender_delta
    received = receiver_delta
    assumptions = ("candidate fee observation; not an exploit confirmation",)
    if semantics == "exact":
        accounted_ok = accounted is None or accounted == requested
        if sent == requested and received == requested and accounted_ok:
            return EconomicResult(
                OracleStatus.BALANCED.value,
                "exact transfer matches requested, sender decrease, receiver increase, and accounted",
                relation="exact-transfer",
            )
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            "exact-transfer quantities are inconsistent",
            assumptions,
            relation="exact-transfer",
        )
    if received > sent:
        return EconomicResult(
            OracleStatus.UNKNOWN.value,
            "the receiver increase exceeds the sender decrease",
            assumptions,
            relation="fee-on-transfer",
        )
    fee = sent - received
    if accounted is not None and accounted != received:
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            f"accounted {accounted} differs from received {received}; requested {requested}; fee {fee}",
            assumptions,
            relation="fee-on-transfer",
        )
    if sent != requested:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "the requested amount does not match the sender decrease",
            assumptions,
            relation="fee-on-transfer",
        )
    if fee == 0:
        return EconomicResult(
            OracleStatus.BALANCED.value,
            "sender decrease and receiver increase match the requested amount",
            relation="fee-on-transfer",
        )
    return EconomicResult(
        OracleStatus.POTENTIAL_LOSS.value,
        f"observed fee {fee} on requested {requested}",
        ("the fee is an observation, not a vulnerability",),
        relation="fee-on-transfer",
    )


def oracle_dependency(
    *,
    state_changed: bool,
    freshness: str,
    design: str,
    input_relation: bool,
    valued: bool,
    sensitive: bool,
) -> EconomicResult:
    if design not in {"spot", "twap", "caller", "unknown"}:
        return EconomicResult(OracleStatus.UNSUPPORTED.value, "oracle design is unsupported")
    if design == "unknown":
        return EconomicResult(OracleStatus.UNKNOWN.value, "oracle design is unknown")
    if freshness == "stale":
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "a stale observation is not a current price",
            ("staleness is not itself a vulnerability",),
            relation=design,
        )
    if freshness not in {"fresh", "unknown"}:
        return EconomicResult(OracleStatus.UNSUPPORTED.value, "oracle freshness is unsupported")
    if not input_relation:
        return EconomicResult(
            OracleStatus.UNKNOWN.value,
            "a price feed is not treated as manipulable without a state relationship",
        )
    if state_changed and valued and sensitive:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "state change, oracle observation, valuation, and a sensitive action were recorded",
            ("a dependency is not confirmation",),
            relation=f"{design}:state->oracle->valuation",
        )
    return EconomicResult(
        OracleStatus.INCOMPLETE.value, "the oracle sequence is incomplete", relation=design
    )


def amm_invariant(
    *,
    reserve0: int | None,
    reserve1: int | None,
    later0: int | None = None,
    later1: int | None = None,
    model: str,
    fee_established: bool = False,
) -> EconomicResult:
    if model != "constant-product":
        return EconomicResult(
            OracleStatus.UNSUPPORTED.value,
            "a pair-like contract is not treated as Uniswap",
        )
    if reserve0 is None or reserve1 is None:
        return EconomicResult(OracleStatus.UNKNOWN.value, "a reserve is unknown")
    product = reserve0 * reserve1
    if later0 is None or later1 is None:
        return EconomicResult(
            OracleStatus.BALANCED.value,
            f"constant-product observation {product}",
            ("only the established product is recorded",),
            relation="reserve0*reserve1",
        )
    later = later0 * later1
    if later == product:
        return EconomicResult(
            OracleStatus.BALANCED.value,
            f"product unchanged at {product}",
            relation="reserve0*reserve1",
        )
    if not fee_established:
        return EconomicResult(
            OracleStatus.UNKNOWN.value,
            "the product changed and no fee relation is established",
            relation="reserve0*reserve1",
        )
    if later < product:
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            "the established product decreased",
            ("candidate invariant break; not an exploit confirmation",),
            relation="reserve0*reserve1",
        )
    return EconomicResult(
        OracleStatus.BALANCED.value,
        "product is non-decreasing under the fee relation",
        relation="reserve0*reserve1",
    )


def lending_transition(
    *,
    debt_before: int | None,
    debt_after: int | None,
    repayment: int | None,
    collateral_before: int | None,
    collateral_after: int | None,
    health: str,
    liquidated: bool,
    debt_semantics: str = "unknown",
    principal: int | None = None,
    interest: int | None = None,
    protocol_fee: int | None = None,
    bad_debt: int | None = None,
) -> EconomicResult:
    """Reconcile debt only when the protocol semantics name every debt-changing term."""
    if debt_before is None or debt_after is None:
        return EconomicResult(OracleStatus.UNKNOWN.value, "debt is unknown")
    assumptions = ("candidate accounting observation; not confirmation",)
    if liquidated and health == "unknown":
        return EconomicResult(
            OracleStatus.UNKNOWN.value, "liquidation health is unknown", assumptions
        )
    if liquidated and health == "healthy":
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            "liquidation under a healthy condition",
            assumptions,
            relation="health/liquidation",
        )
    if liquidated and collateral_after == 0 and debt_after > 0:
        return EconomicResult(
            OracleStatus.POTENTIAL_LOSS.value,
            "bad debt remains after collateral is exhausted",
            assumptions,
            relation="bad-debt",
        )
    if (
        collateral_before is not None
        and collateral_after is not None
        and collateral_before != collateral_after
        and not liquidated
        and debt_semantics not in {"repayment-only", "with-interest"}
    ):
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "collateral movement is not a debt invariant by itself",
            assumptions,
            relation="collateral/debt",
        )
    if debt_semantics == "repayment-only":
        if repayment is None or principal is not None or interest is not None:
            return EconomicResult(
                OracleStatus.INCOMPLETE.value,
                "repayment-only reconciliation is missing a required term",
                assumptions,
                relation="debt/repayment",
            )
        if debt_before - repayment == debt_after:
            return EconomicResult(
                OracleStatus.BALANCED.value,
                "debt changed only by the established repayment",
                assumptions,
                relation="debt/repayment",
            )
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            "debt change differs from the established repayment-only relation",
            assumptions,
            relation="debt/repayment",
        )
    if debt_semantics == "with-interest":
        parts = (principal, repayment, interest, protocol_fee, bad_debt)
        if any(part is None for part in parts):
            return EconomicResult(
                OracleStatus.INCOMPLETE.value,
                "interest, fees, principal, repayment, or bad debt was not established",
                assumptions,
                relation="debt/components",
            )
        assert principal is not None and repayment is not None
        assert interest is not None and protocol_fee is not None and bad_debt is not None
        expected = principal + interest + protocol_fee - repayment - bad_debt
        if expected == debt_after:
            return EconomicResult(
                OracleStatus.BALANCED.value,
                "debt matches principal, interest, fees, repayment, and bad debt",
                assumptions,
                relation="debt/components",
            )
        return EconomicResult(
            OracleStatus.INVARIANT_VIOLATION.value,
            "debt does not match the established component relation",
            assumptions,
            relation="debt/components",
        )
    if debt_before != debt_after or repayment is not None or liquidated:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "debt reconciliation is incomplete until every debt-changing operation is established",
            assumptions,
            relation="debt/unknown",
        )
    return EconomicResult(
        OracleStatus.INCOMPLETE.value,
        "no established lending relation covers this transition",
        assumptions,
    )


def correlate_event(event: str, *, established: frozenset[str]) -> EconomicResult:
    if event == "Transfer" and "balance" in established:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "Transfer correlated with a balance update and a share calculation",
            (
                "static semantics and the runtime event stay distinct; correlation is not verification",
            ),
            relation="event->balance->shares",
        )
    if event == "Swap" and "reserve" in established:
        return EconomicResult(
            OracleStatus.INCOMPLETE.value,
            "Swap correlated with a reserve update and a price-sensitive valuation",
            ("correlation is not verification",),
            relation="event->reserve->valuation",
        )
    return EconomicResult(
        OracleStatus.UNSUPPORTED.value, "the event has no established economic correlation"
    )


def _signed(net: int, token: str, conversion: str) -> EconomicResult:
    if net > 0:
        return EconomicResult(
            OracleStatus.POTENTIAL_POSITIVE_DELTA.value,
            f"net {net} {token}",
            ("a positive delta is not a confirmed exploit",),
            conversion=conversion,
        )
    if net < 0:
        return EconomicResult(
            OracleStatus.POTENTIAL_LOSS.value,
            f"net {net} {token}",
            ("a loss observation is not confirmation",),
            conversion=conversion,
        )
    return EconomicResult(
        OracleStatus.NON_PROFITABLE.value,
        f"net 0 {token}",
        ("zero net is not a proof of safety",),
        conversion=conversion,
    )
