"""Evidence quality, deltas, contradictions, and identity-safe correlation.

Everything here reads engine results and produces typed records. Nothing here
can mark a finding verified, and a missing finding is never proof of safety.
"""

from __future__ import annotations

import json
from dataclasses import replace

from app.discovery.capabilities import ResultStatus
from app.discovery.correlation import detector_family
from app.discovery.orchestration.codec import digest
from app.discovery.orchestration.model import (
    CampaignIdentity,
    Contradiction,
    EvidenceDelta,
    EvidenceItem,
    EvidenceQuality,
    ExecutionPhase,
    FailureClass,
    NegativeEvidence,
)
from app.discovery.results import DynamicResult

MAX_ITEMS_PER_RESULT = 32
MAX_FINDINGS_PER_RESULT = 24
MAX_PATHS_PER_RESULT = 16
MAX_ATTR_CHARS = 160

RUNTIME_FAMILY = frozenset({"runtime_validation", "fork_validation", "differential_validation"})
_UNKNOWN_OBSERVATIONS = frozenset({"unknown", "incomplete", "unavailable", "inconclusive"})
_DISCRIMINATORS = ("differential_validation", "runtime_validation", "fork_validation")


def outcome_of(result: DynamicResult) -> tuple[ExecutionPhase, FailureClass]:
    """Map an engine result to execution truth. Missing tools and refusals are not failures of research."""
    status = result.status
    if result.provenance == "budget_exhausted":
        return ExecutionPhase.BLOCKED, FailureClass.POLICY
    if status is ResultStatus.UNAVAILABLE:
        return ExecutionPhase.UNAVAILABLE, FailureClass.UNAVAILABLE
    if status in {ResultStatus.NOT_IMPLEMENTED, ResultStatus.UNSUPPORTED, ResultStatus.PLANNED}:
        return ExecutionPhase.UNSUPPORTED, FailureClass.UNSUPPORTED
    if status is ResultStatus.TIMEOUT:
        return ExecutionPhase.TIMED_OUT, FailureClass.TIMEOUT
    if status in {ResultStatus.FAILED, ResultStatus.TOOL_FAILURE}:
        if result.executed and (result.crash or result.assertion or result.findings):
            return ExecutionPhase.COMPLETED, FailureClass.NONE
        failure = FailureClass.TOOL if result.executed else FailureClass.INFRASTRUCTURE
        return ExecutionPhase.FAILED, failure
    if not result.executed:
        return ExecutionPhase.UNKNOWN, FailureClass.UNKNOWN_OUTCOME
    return ExecutionPhase.COMPLETED, FailureClass.NONE


def is_positive(result: DynamicResult) -> bool:
    """A tool reported a candidate. This is a signal, not a verdict."""
    bound = str(result.metadata.get("bound", result.metadata.get("economic_bound", ""))).lower()
    return bool(
        result.findings
        or result.crash
        or result.assertion
        or result.sanitizer
        or (result.oracle_kind == "economic" and bound == "true")
    )


def polarity_of(result: DynamicResult, phase: ExecutionPhase) -> str:
    if phase is not ExecutionPhase.COMPLETED:
        return ""
    metadata = result.metadata
    classification = str(metadata.get("classification", ""))
    if classification == "deterministic divergence":
        return "divergent"
    if is_positive(result):
        return "positive"
    if metadata.get("transaction_outcome") == "reverted":
        return "negative"
    return "neutral"


def quality_of(result: DynamicResult, phase: ExecutionPhase) -> EvidenceQuality:
    if phase is ExecutionPhase.UNAVAILABLE:
        return EvidenceQuality.UNAVAILABLE
    if phase in {ExecutionPhase.UNSUPPORTED, ExecutionPhase.BLOCKED}:
        return EvidenceQuality.UNSUPPORTED
    if phase is not ExecutionPhase.COMPLETED:
        return EvidenceQuality.INCOMPLETE
    metadata = result.metadata
    observation = str(metadata.get("observation_status", ""))
    if observation in _UNKNOWN_OBSERVATIONS or metadata.get("document_attributed") == "false":
        return EvidenceQuality.UNKNOWN
    if metadata.get("transaction_outcome") == "unknown":
        return EvidenceQuality.UNKNOWN
    if is_positive(result):
        return EvidenceQuality.CANDIDATE
    return EvidenceQuality.OBSERVATION


def identity_matches(result: DynamicResult, identity: CampaignIdentity) -> bool:
    """A result that names a different snapshot or compiler cannot be attributed to this campaign."""
    pairs = (
        ("source_snapshot", identity.source_snapshot),
        ("compiler_configuration", identity.compiler_configuration),
        ("program_context", identity.program_context),
    )
    for name, expected in pairs:
        reported = str(result.metadata.get(name, ""))
        if expected and reported and reported != expected:
            return False
    return True


def _clip(value: object) -> str:
    return str(value)[:MAX_ATTR_CHARS]


def _category(result: DynamicResult, capability: str) -> str:
    chosen = str(result.metadata.get("evidence_class", ""))
    if chosen in {"economic", "protocol", "runtime", "differential"}:
        return chosen
    if capability == "cross_contract_analysis":
        return "protocol"
    if capability == "economic_simulation":
        return "economic"
    if capability in RUNTIME_FAMILY:
        return "differential" if capability == "differential_validation" else "runtime"
    return "summary"


def attribute(contract: str, function: str, identity: CampaignIdentity) -> str:
    """The identity key this observation belongs to, or empty when it cannot be attributed.

    A bare function name is never enough, and a name without a signature cannot
    select one overload of a campaign target that names a signature.
    """
    own_contract = contract
    own_function = function
    if "::" in own_function:
        tail = own_function.split("::", 1)[1]
        head, _, rest = tail.partition(".")
        own_contract = own_contract or head
        own_function = rest
    if not own_contract or not own_function:
        return ""
    target_contract, target_function = identity.contract, identity.function
    if target_contract and target_function:
        if own_contract == target_contract and _name(own_function) == _name(target_function):
            if "(" in target_function and own_function != target_function:
                return ""
            return identity.identity_key()
    return f"{own_contract}.{own_function}"


def _name(function: str) -> str:
    return function.split("(", 1)[0]


def _item(
    *,
    execution_key: str,
    kind: str,
    category: str,
    engine: str,
    capability: str,
    quality: EvidenceQuality,
    identity_key: str,
    polarity: str,
    attrs: dict[str, str],
) -> EvidenceItem:
    content = {
        "kind": kind,
        "category": category,
        "engine": engine,
        "capability": capability,
        "identity_key": identity_key,
        "polarity": polarity,
        "attrs": attrs,
    }
    return EvidenceItem(
        evidence_id=f"ev_{digest(content)}",
        kind=kind,
        quality=quality,
        engine=engine,
        capability=capability,
        identity_key=identity_key,
        execution_key=execution_key,
        polarity=polarity,
        attrs={**attrs, "category": category},
    )


def items_from_result(
    result: DynamicResult,
    *,
    identity: CampaignIdentity,
    capability: str,
    execution_key: str,
    phase: ExecutionPhase,
) -> tuple[EvidenceItem, ...]:
    """Typed, bounded evidence from one result. Raw tool output is never copied."""
    quality = quality_of(result, phase)
    matched = identity_matches(result, identity)
    if not matched:
        quality = EvidenceQuality.UNKNOWN
    polarity = polarity_of(result, phase) if matched else ""
    category = _category(result, capability)
    target_key = attribute(result.contract, result.function, identity)
    base_attrs = {
        "status": result.status.value,
        "executed": str(result.executed).lower(),
        "verified": "false",
    }
    if not matched:
        base_attrs["identity"] = "mismatch"
    for name in (
        "classification",
        "observation_status",
        "transaction_outcome",
        "runtime_mode",
        "request_identity",
        "deterministic",
    ):
        if result.metadata.get(name):
            base_attrs[name] = _clip(result.metadata[name])
    kind = result.to_evidence().kind.value
    items: list[EvidenceItem] = [
        _item(
            execution_key=execution_key,
            kind=kind,
            category=category,
            engine=result.engine,
            capability=capability,
            quality=quality,
            identity_key=target_key,
            polarity=polarity,
            attrs=base_attrs,
        )
    ]
    if phase is ExecutionPhase.COMPLETED and matched:
        for finding in result.findings[:MAX_FINDINGS_PER_RESULT]:
            key = attribute(finding.contract, finding.function, identity)
            items.append(
                _item(
                    execution_key=execution_key,
                    kind="finding",
                    category="finding",
                    engine=result.engine,
                    capability=capability,
                    quality=EvidenceQuality.CANDIDATE,
                    identity_key=key,
                    polarity="positive",
                    attrs={
                        "detector": _clip(finding.detector_id),
                        "family": detector_family(finding.detector_id),
                        "file": _clip(finding.file_path),
                        "line": str(finding.line),
                        "verified": "false",
                    },
                )
            )
        sequence_id = str(result.metadata.get("sequence_id", ""))
        if sequence_id:
            items.append(
                _item(
                    execution_key=execution_key,
                    kind="sequence",
                    category="sequence",
                    engine=result.engine,
                    capability=capability,
                    quality=EvidenceQuality.OBSERVATION,
                    identity_key=target_key,
                    polarity="neutral",
                    attrs={"sequence_id": _clip(sequence_id), "verified": "false"},
                )
            )
        snapshot = str(result.metadata.get("state_snapshot", ""))
        if snapshot:
            items.append(
                _item(
                    execution_key=execution_key,
                    kind="state_transition",
                    category="state_transition",
                    engine=result.engine,
                    capability=capability,
                    quality=EvidenceQuality.OBSERVATION,
                    identity_key=target_key,
                    polarity="neutral",
                    attrs={
                        "state_snapshot": _clip(snapshot),
                        "transaction_index": _clip(result.metadata.get("transaction_index", "")),
                        "verified": "false",
                    },
                )
            )
        for path_id in _path_ids(result)[:MAX_PATHS_PER_RESULT]:
            items.append(
                _item(
                    execution_key=execution_key,
                    kind="protocol_path",
                    category="protocol_path",
                    engine=result.engine,
                    capability=capability,
                    quality=EvidenceQuality.OBSERVATION,
                    identity_key="",
                    polarity="neutral",
                    attrs={"path_id": _clip(path_id), "verified": "false"},
                )
            )
    unique: dict[str, EvidenceItem] = {}
    for item in items:
        unique.setdefault(item.evidence_id, item)
    return tuple(unique.values())[:MAX_ITEMS_PER_RESULT]


def _path_ids(result: DynamicResult) -> list[str]:
    raw = result.metadata.get("path_ids", "")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


def negative_from(
    result: DynamicResult,
    *,
    identity: CampaignIdentity,
    capability: str,
    phase: ExecutionPhase,
) -> NegativeEvidence | None:
    """A bounded run that found nothing. It never proves safety."""
    if phase is not ExecutionPhase.COMPLETED or is_positive(result):
        return None
    if not identity_matches(result, identity):
        return None
    return NegativeEvidence(
        kind="not_observed",
        capability=capability,
        engine=result.engine,
        identity_key=attribute(result.contract, result.function, identity),
        statement=(
            f"{result.engine} ran {capability} within its bounds and reported no candidate; "
            "this is not evidence of safety"
        ),
        proves_safety=False,
    )


def corroborate(items: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
    """Mark candidates that two independent mechanisms report for one identity.

    Independent means a different engine and a different capability. A missing
    identity never corroborates. The result is still not verification.
    """
    base = tuple(
        replace(item, quality=EvidenceQuality.CANDIDATE)
        if item.quality is EvidenceQuality.CORROBORATED
        else item
        for item in items
    )
    groups: dict[tuple[str, str], list[EvidenceItem]] = {}
    for item in base:
        if (
            item.quality is EvidenceQuality.CANDIDATE
            and item.polarity == "positive"
            and item.identity_key
            and item.attrs.get("identity") != "mismatch"
        ):
            groups.setdefault((item.identity_key, item.attrs.get("family", "")), []).append(item)
    promoted: set[str] = set()
    for members in groups.values():
        engines = {item.engine for item in members}
        capabilities = {item.capability for item in members}
        if len(engines) >= 2 and len(capabilities) >= 2:
            promoted.update(item.evidence_id for item in members)
    return tuple(
        replace(item, quality=EvidenceQuality.CORROBORATED)
        if item.evidence_id in promoted
        else item
        for item in base
    )


def detect_contradictions(
    items: tuple[EvidenceItem, ...], exercised: frozenset[str]
) -> tuple[Contradiction, ...]:
    """Conflicts between evidence for one identity. Neither side is ever chosen.

    A contradiction is open until a discriminating capability has run and then
    stays unresolved if the conflict remains. It is never marked resolved here.
    """
    by_identity: dict[str, list[EvidenceItem]] = {}
    for item in items:
        if item.identity_key and item.attrs.get("identity") != "mismatch":
            by_identity.setdefault(item.identity_key, []).append(item)
    found: list[Contradiction] = []
    for identity_key in sorted(by_identity):
        members = by_identity[identity_key]
        positives = [
            i for i in members if i.polarity == "positive" and i.attrs.get("category") == "finding"
        ]
        negatives = [
            i
            for i in members
            if i.polarity == "negative" and i.capability in RUNTIME_FAMILY and i.kind != "finding"
        ]
        if positives and negatives:
            found.append(
                _build("candidate_not_reproduced", identity_key, positives + negatives, exercised)
            )
        divergent = [i for i in members if i.polarity == "divergent"]
        divergent_modes = {i.attrs.get("runtime_mode", "") for i in divergent}
        same = [
            i
            for i in members
            if i.attrs.get("classification") == "deterministic same result"
            and i.attrs.get("runtime_mode", "") not in divergent_modes
        ]
        if divergent and same:
            found.append(_build("runtime_divergence", identity_key, divergent + same, exercised))
    return tuple(sorted(found, key=lambda item: item.contradiction_id))


def _build(
    kind: str, identity_key: str, members: list[EvidenceItem], exercised: frozenset[str]
) -> Contradiction:
    ids = tuple(sorted({item.evidence_id for item in members}))
    engines = tuple(sorted({item.engine for item in members}))
    discriminators = tuple(
        item for item in _DISCRIMINATORS if item not in {m.capability for m in members}
    )
    status = "unresolved" if any(item in exercised for item in discriminators) else "open"
    return Contradiction(
        contradiction_id=f"ct_{digest((kind, identity_key, ids))}",
        kind=kind,
        identity_key=identity_key,
        evidence_ids=ids,
        engines=engines,
        discriminators=discriminators,
        status=status,
    )


def compute_delta(
    before: tuple[EvidenceItem, ...],
    after: tuple[EvidenceItem, ...],
    *,
    new_seeds: tuple[str, ...],
    new_coverage: bool,
    contradictions_before: tuple[Contradiction, ...],
    contradictions_after: tuple[Contradiction, ...],
    negatives_added: tuple[str, ...],
    uncertainty_before: tuple[str, ...],
    uncertainty_after: tuple[str, ...],
) -> EvidenceDelta:
    known = {item.evidence_id for item in before}
    added = sorted(
        (item for item in after if item.evidence_id not in known), key=lambda i: i.evidence_id
    )

    def ids(category: str) -> tuple[str, ...]:
        return tuple(i.evidence_id for i in added if i.attrs.get("category") == category)

    seen_contradictions = {item.contradiction_id for item in contradictions_before}
    return EvidenceDelta(
        new_evidence_ids=tuple(i.evidence_id for i in added),
        new_finding_ids=ids("finding"),
        new_protocol_paths=ids("protocol_path"),
        new_state_transitions=ids("state_transition"),
        new_sequences=ids("sequence"),
        new_runtime_observations=ids("runtime"),
        new_economic_observations=ids("economic"),
        new_differential_observations=ids("differential"),
        new_corpus_seeds=tuple(sorted(new_seeds)),
        new_contradictions=tuple(
            sorted(
                item.contradiction_id
                for item in contradictions_after
                if item.contradiction_id not in seen_contradictions
            )
        ),
        new_negative_evidence=tuple(sorted(negatives_added)),
        new_coverage=new_coverage,
        uncertainty_added=tuple(sorted(set(uncertainty_after) - set(uncertainty_before))),
        uncertainty_removed=tuple(sorted(set(uncertainty_before) - set(uncertainty_after))),
    )
