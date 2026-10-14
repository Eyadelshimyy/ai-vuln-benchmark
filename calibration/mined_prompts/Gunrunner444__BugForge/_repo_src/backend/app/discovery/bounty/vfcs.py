"""Vulnerable Function Call Sequences, feedback-directed mutation, and minimization.

A VFCS is a short plan that is derived from a static candidate and built only from
functions that exist in the analyzed sources. Where an environment primitive is
unavoidable (a donation, an approval, a clock change) it is marked as a primitive
together with what established that it applies. Nothing here runs a transaction,
so a VFCS is never a finding, a counterexample, or a reproduction, and mutation and
minimization never produce verification.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

from app.discovery.orchestration.codec import digest
from app.discovery.sequences import MAX_MUTATIONS as _EXISTING_MUTATIONS
from app.discovery.sequences import MAX_SEQUENCE_LENGTH as _EXISTING_LENGTH
from app.parsing.solidity_research import (
    ResearchModel,
    RFunction,
    SemanticCandidate,
    model_source_problems,
)

MAX_VFCS_LENGTH = _EXISTING_LENGTH
MAX_VFCS_CANDIDATES = 16
MAX_MUTATIONS = _EXISTING_MUTATIONS
MAX_MINIMIZATION_ATTEMPTS = 24
MAX_FEEDBACK_SIGNALS = 16

ATTACKER = "attacker"
VICTIM = "victim"
ATTACKER_CONTROLLED = "attacker_controlled"
UNCONSTRAINED = "unconstrained"
_BOUNDARY_VALUES = ("zero", "one", "max_uint", "contract_balance", "total_supply")


@dataclass(frozen=True)
class SequenceIdentity:
    """What the sequence is bound to. A sequence cannot be replayed against anything else."""

    campaign_id: str = ""
    source_snapshot: str = ""
    compiler_configuration: str = ""
    fork_reference: str = ""
    program_context: str = ""
    # chain:address of the deployment the sequence targets ("" = source only).
    deployment: str = ""
    # path -> sha256 of the source set the calls were built from. Empty means the
    # sequence has no content identity; execution must refuse it. Not part of the
    # sequence id, so existing ids stay stable.
    source_hashes: tuple[tuple[str, str], ...] = ()
    implementation: str = ""


@dataclass(frozen=True)
class VfcsCall:
    contract: str
    function: str
    role: str
    actor: str
    arguments: tuple[tuple[str, str], ...] = ()
    primitive: bool = False
    established_by: str = ""

    @property
    def identity(self) -> str:
        return f"{self.contract}.{self.function}"


@dataclass(frozen=True)
class Vfcs:
    sequence_id: str
    template: str
    origin: str
    calls: tuple[VfcsCall, ...]
    property_under_test: str
    derived_from: str
    identity: SequenceIdentity
    status: str = "candidate"
    verified: bool = False


@dataclass(frozen=True)
class VfcsResult:
    sequences: tuple[Vfcs, ...]
    skipped: tuple[tuple[str, str], ...]
    truncated: bool


@dataclass(frozen=True)
class FeedbackSignal:
    """What a run learned. Only these kinds steer mutation."""

    kind: str  # coverage_gain | symbolic_counterexample | near_miss | contradiction |
    # caller_context_mismatch | oracle_mismatch
    sequence_id: str
    call_index: int = -1
    values: tuple[tuple[str, str], ...] = ()
    engine: str = ""  # the fuzzer that produced the signal (foundry/echidna/medusa/ityfuzz)
    call_instance: str = ""  # stable call-instance id, resolved to call_index when given


def source_hashes_of(sources: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """sha256 of each source file. This is content identity, not a snapshot label."""
    return tuple(
        sorted(
            (path, hashlib.sha256(text.encode("utf-8")).hexdigest())
            for path, text in sources.items()
        )
    )


def stamp_source_identity(sequence: Vfcs, sources: Mapping[str, str]) -> Vfcs:
    """Bind a sequence to source text. The caller supplies the text it claims, not the executor."""
    return replace(
        sequence, identity=replace(sequence.identity, source_hashes=source_hashes_of(sources))
    )


def sequence_id_of(calls: Iterable[VfcsCall], template: str, identity: SequenceIdentity) -> str:
    # Content hashes and implementation are not part of the id. A deployment still is,
    # and sequences without one keep the id shape from before that field existed.
    bound: dict[str, str] = {
        "campaign_id": identity.campaign_id,
        "source_snapshot": identity.source_snapshot,
        "compiler_configuration": identity.compiler_configuration,
        "fork_reference": identity.fork_reference,
        "program_context": identity.program_context,
    }
    if identity.deployment:
        bound["deployment"] = identity.deployment
    return "vf_" + digest(
        (
            [(c.contract, c.function, c.role, c.actor, c.arguments, c.primitive) for c in calls],
            template,
            bound,
        )
    )


# ---- generation -----------------------------------------------------------------------------


def generate(
    model: ResearchModel,
    candidates: Iterable[SemanticCandidate],
    identity: SequenceIdentity = SequenceIdentity(),
    *,
    limit: int = MAX_VFCS_CANDIDATES,
    sources: Mapping[str, str] | None = None,
) -> VfcsResult:
    """Build sequences. ``sources`` records the content identity those sequences may execute against.

    When ``sources`` is supplied it must be the text the model was parsed from.
    A mismatch does not restamp the model or the plan against the new text.
    """
    limit = max(0, min(limit, MAX_VFCS_CANDIDATES))
    found: dict[str, Vfcs] = {}
    skipped: list[tuple[str, str]] = []
    truncated = False
    if sources is not None:
        problems = model_source_problems(model, sources)
        if problems:
            reason = "model source attestation does not match the supplied sources: " + problems[0]
            for candidate in candidates:
                key = f"{candidate.detector}@{candidate.contract}.{candidate.function}"
                skipped.append((key, reason))
            return VfcsResult((), tuple(skipped), False)
        identity = replace(identity, source_hashes=source_hashes_of(sources))
    for candidate in candidates:
        key = f"{candidate.detector}@{candidate.contract}.{candidate.function}"
        if len(found) >= limit:
            truncated = True
            skipped.append((key, "the sequence limit was reached"))
            continue
        builder = _BUILDERS.get(candidate.detector)
        if builder is None:
            skipped.append((key, "no sequence template applies to this detector"))
            continue
        built = builder(model, candidate)
        if isinstance(built, str):
            skipped.append((key, built))
            continue
        template, calls, prop = built
        if not calls or len(calls) > MAX_VFCS_LENGTH:
            skipped.append((key, "the sequence would exceed the length limit"))
            continue
        sequence = Vfcs(
            sequence_id=sequence_id_of(calls, template, identity),
            template=template,
            origin=f"template:{template}",
            calls=tuple(calls),
            property_under_test=prop,
            derived_from=key,
            identity=identity,
        )
        if sequence.sequence_id in found:
            skipped.append((key, "deduplicated: an equivalent sequence was already generated"))
            continue
        found[sequence.sequence_id] = sequence
    return VfcsResult(tuple(found.values()), tuple(skipped), truncated)


Built = tuple[str, list[VfcsCall], str] | str
Builder = Callable[[ResearchModel, SemanticCandidate], Built]


def _resolve(model: ResearchModel, contract: str, signature: str) -> RFunction | None:
    for function in model.functions_of(contract):
        if function.signature == signature and function.has_body:
            return function
    return None


def _split_callee(text: str) -> tuple[str, str]:
    contract, _, signature = text.partition(".")
    return contract, signature


def _call(
    function: RFunction,
    role: str,
    actor: str,
    overrides: Mapping[str, str] | None = None,
    default: str = UNCONSTRAINED,
) -> VfcsCall:
    overrides = overrides or {}
    arguments = tuple(
        (item.name or f"arg{i}", overrides.get(item.name, default))
        for i, item in enumerate(function.params)
    )
    return VfcsCall(function.contract, function.signature, role, actor, arguments)


def _primitive(contract: str, function: str, role: str, actor: str, why: str) -> VfcsCall:
    return VfcsCall(contract, function, role, actor, (), True, why)


def _own(model: ResearchModel, candidate: SemanticCandidate) -> RFunction | str:
    function = _resolve(model, candidate.contract, candidate.function)
    return function if function is not None else "the candidate function is not in the model"


def _withdraw_like(model: ResearchModel, contract: str, exclude: str) -> RFunction | None:
    for function in model.functions_of(contract):
        if (
            function.exposed
            and function.kind == "function"
            and function.signature != exclude
            and function.has_body
            and function.params
            and any(
                token in function.body for token in (".transfer(", ".safeTransfer(", ".call{value")
            )
        ):
            return function
    return None


def _dispatch(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    callee_text = candidate.fact("eventual_callee")
    callee_contract, callee_signature = _split_callee(callee_text)
    if "(" not in callee_signature or _resolve(model, callee_contract, callee_signature) is None:
        return "the nested callee is not a function in the model"
    nested = f"nested_call:{callee_contract}.{callee_signature}"
    overrides = {item.name: nested for item in function.params if "bytes" in item.type_name}
    calls = [_call(function, "dispatch", ATTACKER, overrides, ATTACKER_CONTROLLED)]
    return "nested-dispatch", calls, "the nested callee must authorize the original actor"


def _approval_dispatch(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    approve = _primitive(
        "token",
        "approve(address,uint256)",
        "approve",
        VICTIM,
        f"{function.contract} pulls tokens from users, so users may approve it",
    )
    call = _call(function, "execute", ATTACKER, default=ATTACKER_CONTROLLED)
    return (
        "approve→transferFrom",
        [approve, call],
        "an allowance granted to the contract must not be spendable by another actor",
    )


def _actor_parameter(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    callee_contract, callee_signature = _split_callee(candidate.fact("eventual_callee"))
    if _resolve(model, callee_contract, callee_signature) is None:
        return "the callee is not a function in the model"
    acted_for = candidate.fact("acted_for_parameter").strip("` ")
    overrides = {acted_for: VICTIM} if acted_for else {}
    calls = [_call(function, "relay", ATTACKER, overrides, ATTACKER_CONTROLLED)]
    return "forwarded-actor", calls, "the callee must bind the acted-for account to the caller"


def _donation(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    donate = _primitive(
        "token",
        "transfer(address,uint256)",
        "donate",
        ATTACKER,
        "balanceOf(address(this)) is read by the candidate function",
    )
    if candidate.detector == "accounting.balance_as_deposit":
        calls = [donate, _call(function, "credit", ATTACKER, default=ATTACKER_CONTROLLED)]
        return "donate→credit", calls, "a donation must not be credited as a deposit"
    withdraw = _withdraw_like(model, function.contract, function.signature)
    if withdraw is None:
        return "no function that pays out was found in the contract"
    calls = [
        _call(function, "deposit", ATTACKER, default="minimum_amount"),
        donate,
        _call(function, "deposit", VICTIM, default="victim_amount"),
        _call(withdraw, "withdraw", ATTACKER, default="all_shares"),
    ]
    return "deposit→donate→withdraw", calls, "no actor may withdraw more than it deposited"


def _token_behavior(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    origin = candidate.fact("token_origin", "unknown")
    token_value = "fee_on_transfer_token" if "caller-chosen" in origin else UNCONSTRAINED
    overrides = {item.name: token_value for item in function.params if "IERC20" in item.type_name}
    calls = [_call(function, "deposit", ATTACKER, overrides, "amount_one")]
    withdraw = _withdraw_like(model, function.contract, function.signature)
    if withdraw is not None:
        calls.append(_call(withdraw, "withdraw", ATTACKER, default="credited_balance"))
    template = "deposit→withdraw" if withdraw is not None else "deposit"
    return template, calls, "credited balance must equal the amount received"


def _message(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    detector = candidate.detector
    if detector in {"message.replay_no_consumption", "message.nonce_not_in_digest"}:
        first = _call(function, "execute", ATTACKER, default="valid_message")
        again = replace(
            first,
            role="replay",
            arguments=tuple((n, "replay_of_call:0") for n, _ in first.arguments),
        )
        return "authorize→execute", [first, again], "an authorization must be consumable once"
    omitted = {
        name for name in candidate.fact("omitted_fields").split(",") if name and name != "none"
    }
    if detector == "message.missing_expiry":
        calls = [_call(function, "execute", ATTACKER, default="expired_message")]
        return "authorize→execute", calls, "an expired authorization must be refused"
    overrides = {name: "attacker_substituted" for name in omitted}
    calls = [_call(function, "execute", ATTACKER, overrides, "valid_message")]
    return "authorize→execute", calls, "an authorization must bind every effectful field"


def _oracle(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    if candidate.detector == "oracle.stale_source_accepted":
        shift = _primitive(
            "vm", "warp(uint256)", "age_price", ATTACKER, "the feed timestamp is not checked"
        )
    elif candidate.detector == "oracle.spot_price_sensitive_use":
        shift = _primitive(
            "market",
            "swap(uint256)",
            "move_spot_price",
            ATTACKER,
            "the price is read from pool reserves",
        )
    else:
        shift = _primitive(
            "oracle", "report(uint256)", "update_price", ATTACKER, "the source set is weak"
        )
    calls = [shift, _call(function, "valuation", ATTACKER, default=UNCONSTRAINED)]
    spender = _withdraw_like(model, function.contract, function.signature)
    if spender is not None:
        calls.append(_call(spender, "borrow_or_withdraw", ATTACKER, default="maximum"))
    return "oracle update→valuation→borrow", calls, "value must not depend on a weak price"


def _initializer(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    first = _call(function, "initialize", ATTACKER, default=ATTACKER)
    second = _call(function, "reinitialize", VICTIM, default="second_owner")
    return "initialize→reinitialize", [first, second], "an account initializes exactly once"


UNAUTHENTICATED_EXECUTION = frozenset(
    {
        "aa.unauthenticated_account_execution",
        "v4_hooks.callback_without_pool_manager_check",
        "bridge.receiver_without_endpoint_check",
    }
)


def _account_validation(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    if candidate.detector in UNAUTHENTICATED_EXECUTION:
        calls = [_call(function, "execute", ATTACKER, default=ATTACKER_CONTROLLED)]
        return "unauthenticated-execution", calls, "execution requires a valid operation"
    calls = [_call(function, "validate", ATTACKER, default="forged_operation")]
    for other in model.functions_of(function.contract):
        if other.exposed and other.signature != function.signature and "call{" in other.body:
            calls.append(_call(other, "execute", ATTACKER, default=ATTACKER_CONTROLLED))
            break
    return "AA validation→execution", calls, "execution requires a valid operation"


def _swallowed_call(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    """Bind a swallowed low-level call to a local failing callee when the source allows it.

    Possible false positives and ambiguous propagation are not turned into exploit
    sequences. A receiver that is not a caller-supplied address stays unknown.
    """
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    evidence = candidate.fact("evidence_class", "unknown")
    pattern = candidate.fact("pattern", "unknown")
    if evidence == "possible_false_positive":
        return f"possible false positive ({pattern}); not synthesized as an exploit sequence"
    if evidence == "ambiguous":
        return "failure propagation is ambiguous; a sequence is not synthesized"
    if evidence != "detector_emission":
        return "evidence class is unknown; a sequence is not synthesized"
    target = candidate.fact("target_parameter", "unknown")
    if target in {"", "unknown"}:
        return (
            "the call receiver is not a caller-supplied address parameter; "
            "a failing callee is not invented"
        )
    why = (
        f"failure {candidate.fact('failure_propagation', 'unknown')}; "
        f"gas guard {candidate.fact('gas_guard', 'unknown')}; "
        f"call kind {candidate.fact('call_kind', 'unknown')}; "
        f"receiver parameter {target}"
    )[:240]
    deploy = _primitive("callee", "deploy_reverting()", "failing_callee", ATTACKER, why)
    overrides = {target: "reverting_callee"}
    data_param = candidate.fact("calldata_parameter", "unknown")
    if data_param not in {"", "unknown", "empty"}:
        overrides[data_param] = "reverting_calldata"
    probe = _call(function, "probe", ATTACKER, overrides, UNCONSTRAINED)
    return (
        "failing-callee→swallowed-call",
        [deploy, probe],
        "behavioral probe of a reverting callee: outer success shows the failure was "
        "swallowed and is not by itself a security-property violation",
    )


def _boundary(model: ResearchModel, candidate: SemanticCandidate) -> Built:
    function = _own(model, candidate)
    if isinstance(function, str):
        return function
    calls = [_call(function, "batch", ATTACKER, default="boundary_value")]
    return "boundary-batch", calls, "totals must not wrap or truncate"


_BUILDERS: dict[str, Builder] = {
    "v4_hooks.callback_without_pool_manager_check": _account_validation,
    "bridge.receiver_without_endpoint_check": _account_validation,
    "caller_context.swallowed_call_failure": _swallowed_call,
    "caller_context.self_call_elevation": _dispatch,
    "caller_context.forwarded_sender_delegatecall": _dispatch,
    "caller_context.unrestricted_dispatch_with_approval_authority": _approval_dispatch,
    "caller_context.trusted_intermediary_actor_parameter": _actor_parameter,
    "accounting.donation_share_price": _donation,
    "accounting.balance_as_deposit": _donation,
    "accounting.fee_on_transfer_mismatch": _token_behavior,
    "accounting.unchecked_token_return": _token_behavior,
    "message.field_not_bound": _message,
    "message.missing_domain_separation": _message,
    "message.missing_expiry": _message,
    "message.replay_no_consumption": _message,
    "message.nonce_not_in_digest": _message,
    "oracle.insufficient_quorum": _oracle,
    "oracle.price_not_validated": _oracle,
    "oracle.stale_source_accepted": _oracle,
    "oracle.fallback_weaker": _oracle,
    "oracle.spot_price_sensitive_use": _oracle,
    "aa.unprotected_account_initializer": _initializer,
    "aa.signature_not_bound_to_userophash": _account_validation,
    "aa.signature_result_ignored": _account_validation,
    "aa.validate_without_entrypoint_binding": _account_validation,
    "aa.unauthenticated_account_execution": _account_validation,
    "arithmetic.batch_accumulation_overflow": _boundary,
    "arithmetic.unsafe_downcast_after_arithmetic": _boundary,
}


# ---- feedback-directed mutation -------------------------------------------------------------


def mutate(
    parents: Iterable[Vfcs],
    signals: Iterable[FeedbackSignal],
    *,
    limit: int = MAX_MUTATIONS,
) -> tuple[Vfcs, ...]:
    """Deterministic, bounded variants of known sequences. Every operator keeps real functions."""
    limit = max(0, min(limit, MAX_MUTATIONS))
    by_id = {item.sequence_id: item for item in parents}
    known = set(by_id)
    out: dict[str, Vfcs] = {}
    for signal in sorted(
        list(signals)[:MAX_FEEDBACK_SIGNALS], key=lambda s: (s.kind, s.sequence_id, s.call_index)
    ):
        parent = by_id.get(signal.sequence_id)
        if parent is None:
            continue
        for operator, calls in _variants(parent, signal):
            if not calls or len(calls) > MAX_VFCS_LENGTH:
                continue
            child_id = sequence_id_of(calls, parent.template, parent.identity)
            if child_id in known or child_id in out:
                continue
            if len(out) >= limit:
                return tuple(out.values())
            out[child_id] = replace(
                parent,
                sequence_id=child_id,
                origin=f"mutation:{signal.kind}:{operator}:{parent.sequence_id}",
                calls=tuple(calls),
            )
    return tuple(out.values())


def _variants(parent: Vfcs, signal: FeedbackSignal) -> Iterable[tuple[str, list[VfcsCall]]]:
    calls = list(parent.calls)
    index = signal.call_index if 0 <= signal.call_index < len(calls) else len(calls) - 1
    kind = signal.kind
    if kind == "symbolic_counterexample" and signal.values:
        seeded = dict(signal.values)
        target = calls[index]
        arguments = tuple((n, seeded.get(n, v)) for n, v in target.arguments)
        yield (
            "seed_arguments",
            [*calls[:index], replace(target, arguments=arguments), *calls[index + 1 :]],
        )
    if kind in {"near_miss", "oracle_mismatch", "coverage_gain"}:
        target = calls[index]
        for value in _BOUNDARY_VALUES:
            arguments = tuple(
                (n, value if v in {UNCONSTRAINED, ATTACKER_CONTROLLED} else v)
                for n, v in target.arguments
            )
            if arguments != target.arguments:
                yield (
                    f"boundary_{value}",
                    [*calls[:index], replace(target, arguments=arguments), *calls[index + 1 :]],
                )
    if kind in {"near_miss", "coverage_gain"} and index + 1 < len(calls):
        swapped = list(calls)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        yield "reorder", swapped
    if kind == "coverage_gain" and len(calls) < MAX_VFCS_LENGTH and not calls[index].primitive:
        yield "repeat_call", [*calls[: index + 1], calls[index], *calls[index + 1 :]]
    if kind in {"caller_context_mismatch", "contradiction"}:
        target = calls[index]
        if not target.primitive:
            other = VICTIM if target.actor == ATTACKER else ATTACKER
            yield "swap_actor", [*calls[:index], replace(target, actor=other), *calls[index + 1 :]]


# ---- minimization ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinimizationResult:
    """A reduction. It preserves one named property and it is not verification."""

    original: tuple[VfcsCall, ...]
    minimized: tuple[VfcsCall, ...]
    completed: bool
    reason: str
    attempts: int
    preserved_property: str
    evaluator: str
    inconclusive_attempts: int = 0
    verified: bool = False
    removed: tuple[str, ...] = field(default=())

    @property
    def reduced(self) -> bool:
        return len(self.minimized) < len(self.original)


Evaluator = Callable[[tuple[VfcsCall, ...]], bool | None]


def minimize(
    sequence: Vfcs,
    evaluator: Evaluator,
    *,
    evaluator_name: str,
    max_attempts: int = MAX_MINIMIZATION_ATTEMPTS,
) -> MinimizationResult:
    """Delta debugging over the calls. ``None`` from the evaluator means it could not tell."""
    budget = max(0, min(max_attempts, MAX_MINIMIZATION_ATTEMPTS))
    attempts = 0
    inconclusive = 0
    original = sequence.calls

    def test(candidate: tuple[VfcsCall, ...]) -> bool | None:
        nonlocal attempts, inconclusive
        attempts += 1
        try:
            outcome = evaluator(candidate)
        except Exception:
            outcome = None
        if outcome is None:
            inconclusive += 1
        return outcome

    def result(current: tuple[VfcsCall, ...], completed: bool, reason: str) -> MinimizationResult:
        kept = {c.identity for c in current}
        return MinimizationResult(
            original=original,
            minimized=current,
            completed=completed,
            reason=reason,
            attempts=attempts,
            preserved_property=sequence.property_under_test,
            evaluator=evaluator_name,
            inconclusive_attempts=inconclusive,
            removed=tuple(sorted({c.identity for c in original} - kept)),
        )

    if budget == 0:
        return result(original, False, "attempt_budget_exhausted")
    if test(original) is not True:
        return result(original, True, "original_not_reproduced")
    current = original
    granularity = 2
    while len(current) >= 2:
        size = max(1, len(current) // granularity)
        chunks = [current[i : i + size] for i in range(0, len(current), size)]
        reduced = False
        for index in range(len(chunks)):
            if attempts >= budget:
                return result(current, False, "attempt_budget_exhausted")
            complement = tuple(c for j, chunk in enumerate(chunks) if j != index for c in chunk)
            if complement and test(complement) is True:
                current = complement
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            return result(current, True, "one_minimal")
        granularity = min(len(current), granularity * 2)
    return result(current, True, "one_minimal")


# ---- stable call-instance identity and bounded feedback loop (Phase 51, owner Phase 5) -------
#
# A sequence may call the same (contract, function) more than once; a per-instance identity is
# needed so a fuzzer's feedback maps back to the exact call it refers to. The identity is derived
# from the deterministic sequence id plus the call's position and role, so it is stable across
# regeneration of the same sequence. Feedback is accepted only from the existing bounded fuzzers
# (Foundry, Echidna, Medusa, ItyFuzz); BugForge adds no fuzzer of its own.

ALLOWED_FEEDBACK_ENGINES = frozenset({"foundry", "echidna", "medusa", "ityfuzz"})


def call_instance_id(sequence: Vfcs, index: int) -> str:
    """A stable identity for one call position within a sequence."""
    call = sequence.calls[index]
    return "ci_" + digest(
        (sequence.sequence_id, index, call.contract, call.function, call.role, call.actor)
    )


def instance_identities(sequence: Vfcs) -> tuple[str, ...]:
    """Stable per-call-instance identities, one per call position."""
    return tuple(call_instance_id(sequence, index) for index in range(len(sequence.calls)))


def index_for_instance(sequence: Vfcs, instance_id: str) -> int:
    """Resolve a stable call-instance id to its position, or -1 when it does not belong."""
    for index in range(len(sequence.calls)):
        if call_instance_id(sequence, index) == instance_id:
            return index
    return -1


@dataclass(frozen=True)
class FeedbackOutcome:
    """The result of folding fuzzer feedback back into the research loop. Nothing verified."""

    children: tuple[Vfcs, ...]
    accepted: tuple[str, ...]
    refused: tuple[tuple[str, str], ...]  # (signal reference, reason)
    unavailable_engines: tuple[str, ...]
    verified: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "verified": False,
            "children": [
                {
                    "sequence_id": child.sequence_id,
                    "origin": child.origin,
                    "template": child.template,
                    "calls": [call.identity for call in child.calls],
                    "instances": list(instance_identities(child)),
                }
                for child in self.children
            ],
            "accepted": list(self.accepted),
            "refused": [{"signal": ref, "reason": reason} for ref, reason in self.refused],
            "unavailable_engines": list(self.unavailable_engines),
        }


def incorporate_feedback(
    parents: Iterable[Vfcs],
    signals: Iterable[FeedbackSignal],
    *,
    available_engines: frozenset[str] | None = None,
    limit: int = MAX_MUTATIONS,
) -> FeedbackOutcome:
    """Fold bounded fuzzer feedback into new candidate sequences.

    Only signals from the allowed fuzzers are considered. A signal whose engine is allowed
    but not currently available is reported as unavailable and ignored (never fabricated).
    A signal that names a stable call-instance has it resolved to the concrete call index.
    The actual expansion reuses the existing bounded ``mutate`` and stays within ``limit``.
    """

    parent_list = list(parents)
    by_id = {item.sequence_id: item for item in parent_list}
    usable: list[FeedbackSignal] = []
    refused: list[tuple[str, str]] = []
    unavailable: set[str] = set()
    for signal in signals:
        ref = f"{signal.engine or '?'}:{signal.kind}:{signal.sequence_id}"
        engine = (signal.engine or "").strip().lower()
        if engine not in ALLOWED_FEEDBACK_ENGINES:
            refused.append((ref, "feedback is accepted only from foundry/echidna/medusa/ityfuzz"))
            continue
        if available_engines is not None and engine not in available_engines:
            unavailable.add(engine)
            refused.append((ref, f"{engine} is unavailable; its feedback is not fabricated"))
            continue
        parent = by_id.get(signal.sequence_id)
        if parent is None:
            refused.append((ref, "signal does not match a known sequence"))
            continue
        resolved = signal
        if signal.call_instance:
            index = index_for_instance(parent, signal.call_instance)
            if index < 0:
                refused.append((ref, "call-instance id does not belong to this sequence"))
                continue
            resolved = replace(signal, call_index=index)
        usable.append(resolved)

    children = mutate(parent_list, usable, limit=limit)
    accepted = tuple(f"{s.engine}:{s.kind}:{s.sequence_id}:{s.call_index}" for s in usable)
    return FeedbackOutcome(
        children=children,
        accepted=accepted,
        refused=tuple(refused),
        unavailable_engines=tuple(sorted(unavailable)),
    )
