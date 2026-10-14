"""Authoritative reproduction, corroboration, and verification rules.

These predicates live in the domain so ``SecurityFinding.verify`` and the
lifecycle service cannot disagree. Evidence identity does not use object ids.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    EvidenceProvenance,
    evidence_contradicts,
    observation_identity,
)
from app.domain.trusted_evidence import is_trusted_observation

# Observations that can verify a finding. A reproduction record, log,
# screenshot, test failure, generated test, static analysis result, and AI
# text are absent on purpose.
INDEPENDENT_VERIFICATION_KINDS = frozenset(
    {
        EvidenceKind.HTTP_RESPONSE,
        EvidenceKind.BROWSER,
        EvidenceKind.SCANNER,
        EvidenceKind.API_TEST,
        EvidenceKind.REPLAY,
        EvidenceKind.FUZZING,
        EvidenceKind.PROXY,
    }
)

_CORROBORATING_KINDS = frozenset(
    {
        EvidenceKind.STATIC_ANALYSIS,
        EvidenceKind.SOURCE_CODE,
    }
)
# A single non-AI research observation can corroborate. It cannot verify.
_RESEARCH_OBSERVATION_KINDS = INDEPENDENT_VERIFICATION_KINDS | frozenset(
    {EvidenceKind.HTTP_REQUEST}
)

_FAILED_OUTCOMES = frozenset(
    {
        "failed",
        "not_reproduced",
        "blocked",
        "inconclusive",
        "intermittent",
        "error",
        "no_evidence",
        "environment_error",
        "timeout",
    }
)
_POSITIVE_OUTCOMES = frozenset(
    {
        "reproduced",
        "consistently_reproduced",
        "success",
        "exploited",
    }
)
_NEGATIVE_FLAGS = frozenset({"0", "false", "no"})
_POSITIVE_FLAGS = frozenset({"1", "true", "yes"})


def positive_reproduction(item: Evidence) -> bool:
    """True only for an explicit successful reproduction result.

    Kind ``REPRODUCTION`` is not enough. The record needs ``reproduced=true``
    or an outcome of ``reproduced`` or ``consistently_reproduced``. Failed,
    intermittent, inconclusive, blocked, and unknown outcomes are not success.
    This predicate does not check the semantic target. Lifecycle transitions
    use :func:`reproduction_for_target`.
    """
    if item.kind is not EvidenceKind.REPRODUCTION:
        return False
    if item.provenance is not EvidenceProvenance.REPRODUCTION:
        return False
    if evidence_contradicts(item):
        return False
    outcome = _meta(item, "outcome").lower()
    reproduced = _meta(item, "reproduced").lower()
    if outcome in _FAILED_OUTCOMES or reproduced in _NEGATIVE_FLAGS:
        return False
    if outcome in _POSITIVE_OUTCOMES:
        return True
    return reproduced in _POSITIVE_FLAGS and outcome in {"", *_POSITIVE_OUTCOMES}


def reproduction_for_target(item: Evidence, target_id: str) -> bool:
    """Successful reproduction bound to one semantic target.

    A positive record without ``observed_target``, or one bound to a different
    target, is historical only. It does not establish ``REPRODUCED`` or
    ``HUMAN_ACCEPTED`` for the current finding. This is not a server signature.
    """
    if not target_id or not positive_reproduction(item):
        return False
    return _meta(item, "observed_target") == target_id


def independent_verification_items(
    items: Sequence[Evidence],
    *,
    target_id: str = "",
    finding_id: str = "",
    finding_key: str = "",
    project_id: str = "",
) -> tuple[Evidence, ...]:
    """Trusted observations that verify one finding.

    An observation counts only when all of the following hold:

    * it is a ``ServerObservation`` whose signature matches the server secret
    * its kind is an observational verification kind
    * its finding id, finding key, and project id match this finding
    * its ``observed_target`` is the finding's current semantic target
    * its canonical identity is not a positive reproduction record
    * its execution id is not a reproduction execution id

    A matching semantic target is not enough. A new ``Evidence`` UUID is not
    independence. Unstamped HTTP, browser, scanner, API, replay, fuzzing, or
    proxy evidence does not verify. Client metadata, including
    ``attribution=server``, does not verify.
    """
    reproductions = [item for item in items if positive_reproduction(item)]
    reproduction_ids = {observation_identity(item) for item in reproductions}
    reproduction_executions = {
        _meta(item, "execution_id") for item in reproductions if _meta(item, "execution_id")
    }
    found: list[Evidence] = []
    for item in items:
        if evidence_contradicts(item) or not is_trusted_observation(item):
            continue
        if item.kind not in INDEPENDENT_VERIFICATION_KINDS:
            continue
        if not _bound_to_finding(
            item,
            target_id=target_id,
            finding_id=finding_id,
            finding_key=finding_key,
            project_id=project_id,
        ):
            continue
        if observation_identity(item) in reproduction_ids:
            continue
        execution = _meta(item, "execution_id")
        if execution and execution in reproduction_executions:
            continue
        found.append(item)
    return tuple(found)


def can_corroborate(
    evidence: EvidenceBundle | Sequence[Evidence],
    *,
    target_id: str = "",
    finding_id: str = "",
    finding_key: str = "",
    project_id: str = "",
) -> bool:
    """True when authorized non-AI observations support this finding's target.

    Two distinct static or source observations corroborate when each one is
    either unbound (legacy evidence with no ``observed_target``) or stamped
    with ``target_id``. A stamp for a different target does not count.
    One static record, two copies of that record, static plus AI text, and a
    contradictory record do not corroborate.

    A runtime observation corroborates only when it is a ``ServerObservation``
    whose finding id, finding key, project id, and ``observed_target`` match
    the current finding. The observation's own target is not used as the
    expected target. Plain client HTTP evidence and forged attribution do
    not corroborate. A trusted runtime observation still cannot verify.
    """
    items = evidence.items if isinstance(evidence, EvidenceBundle) else evidence
    seen: set[tuple[str, ...]] = set()
    research = False
    for item in items:
        if item.kind is EvidenceKind.AI_ANALYSIS or evidence_contradicts(item):
            continue
        if item.kind in _CORROBORATING_KINDS:
            if _static_matches_target(item, target_id):
                seen.add(observation_identity(item))
        elif (
            item.kind in _RESEARCH_OBSERVATION_KINDS
            and is_trusted_observation(item)
            and _bound_to_finding(
                item,
                target_id=target_id,
                finding_id=finding_id,
                finding_key=finding_key,
                project_id=project_id,
            )
        ):
            research = True
    return len(seen) >= 2 or research


def _static_matches_target(item: Evidence, target_id: str) -> bool:
    """Legacy static evidence has no target stamp and still counts.

    Analysis stamps ``observed_target`` when the finding is created. A later
    material target change leaves that stamp behind, so the old record stays
    stored without corroborating the new target.
    """
    bound = _meta(item, "observed_target")
    if not bound:
        return True
    return bool(target_id) and bound == target_id


def _bound_to_finding(
    item: Evidence,
    *,
    target_id: str,
    finding_id: str,
    finding_key: str,
    project_id: str,
) -> bool:
    if not target_id or not finding_id:
        return False
    if _meta(item, "observed_target") != target_id:
        return False
    if _meta(item, "finding_id") != finding_id:
        return False
    if _meta(item, "finding_key") != finding_key:
        return False
    return _meta(item, "project_id") == project_id


def _meta(item: Evidence, key: str) -> str:
    value = item.metadata.get(key)
    return "" if value is None else str(value)
