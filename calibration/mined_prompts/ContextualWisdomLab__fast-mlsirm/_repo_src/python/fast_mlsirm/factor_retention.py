"""Govern already-computed factor-retention evidence without doing psychometric math.

This module is intentionally a validation and orchestration boundary. Numerical
factor-retention methods remain owned by the existing Rust-backed implementation
paths. The contract records which supported method supplied each candidate count,
prevents duplicate-method double counting, and represents agreement or
uncertainty without selecting a structural measurement model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

MAX_FACTOR_CANDIDATE_COUNT = 10_000
"""Largest candidate factor count accepted by the governed transport contract."""


class FactorRetentionMethod(str, Enum):
    """Supported identities for already-computed factor-retention evidence."""

    PARALLEL_ANALYSIS = "parallel_analysis"
    VELICER_MAP = "velicer_map"
    LIKELIHOOD_INFORMATION_CRITERION = "likelihood_information_criterion"
    BOOTSTRAP_LR = "bootstrap_lr"
    PREDICTIVE = "predictive"
    EXTERNAL_SUPPORTED = "external_supported"


class FactorRetentionDecision(str, Enum):
    """Conservative decision states for the governed evidence set."""

    CONSENSUS = "consensus"
    DISAGREEMENT = "disagreement"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _require_method(value: object) -> None:
    """Admit only the closed package-owned factor-retention method enum."""
    if type(value) is not FactorRetentionMethod:
        raise TypeError("method must be a FactorRetentionMethod")


def _require_candidate_count(value: object) -> None:
    """Validate one candidate count without invoking caller numeric protocols."""
    if type(value) is not int:
        raise ValueError("candidate_count must be a positive integer")
    if value <= 0:
        raise ValueError("candidate_count must be a positive integer")
    if value > MAX_FACTOR_CANDIDATE_COUNT:
        raise ValueError(
            "candidate_count exceeds maximum governed factor candidate count"
        )


@dataclass(frozen=True, slots=True)
class FactorRetentionEvidence:
    """One supported method's already-computed candidate factor count."""

    method: FactorRetentionMethod
    candidate_count: int

    def __post_init__(self) -> None:
        """Fail closed on untyped methods and unbounded candidate counts."""
        _require_method(self.method)
        _require_candidate_count(self.candidate_count)


@dataclass(frozen=True, slots=True)
class FactorRetentionResult:
    """Deterministic governed summary over a factor-retention evidence set."""

    decision: FactorRetentionDecision
    retained_count: int | None
    candidate_range: tuple[int, int] | None
    evidence: tuple[FactorRetentionEvidence, ...]

    @property
    def evidence_count(self) -> int:
        """Return the number of distinct supported method records."""
        return len(self.evidence)


def govern_factor_retention(
    evidence: Iterable[FactorRetentionEvidence],
) -> FactorRetentionResult:
    """Summarize supported retention evidence without computing factor statistics.

    Two or more distinct methods that return the same candidate count produce a
    ``consensus`` result. Two or more distinct methods that disagree produce a
    ``disagreement`` result whose range spans the observed supported candidates.
    Zero or one method is ``insufficient_evidence``. Evidence is consumed only
    until a duplicate makes further input invalid, so caller-controlled iterables
    are not exhausted unnecessarily. Because method identity is a closed enum and
    duplicates fail immediately, accepted evidence cardinality is intrinsically
    bounded by the supported method set. No branch selects bifactor, higher-order,
    testlet, faceted, or latent-space structure.
    """

    iterable_message = "evidence must be an iterable of FactorRetentionEvidence"
    try:
        iterator = iter(evidence)
    except MemoryError:
        raise
    except Exception:
        raise ValueError(iterable_message) from None

    records: list[FactorRetentionEvidence] = []
    seen_methods: set[FactorRetentionMethod] = set()
    while True:
        try:
            record = next(iterator)
        except StopIteration:
            break
        except MemoryError:
            raise
        except Exception:
            raise ValueError(iterable_message) from None
        if type(record) is not FactorRetentionEvidence:
            raise TypeError("evidence entries must be FactorRetentionEvidence")
        method = record.method
        candidate_count = record.candidate_count
        _require_method(method)
        _require_candidate_count(candidate_count)
        if method in seen_methods:
            raise ValueError("duplicate factor-retention method evidence")
        seen_methods.add(method)
        records.append(FactorRetentionEvidence(method, candidate_count))

    ordered = tuple(sorted(records, key=lambda record: record.method.value))
    if not ordered:
        return FactorRetentionResult(
            decision=FactorRetentionDecision.INSUFFICIENT_EVIDENCE,
            retained_count=None,
            candidate_range=None,
            evidence=ordered,
        )

    counts = tuple(record.candidate_count for record in ordered)
    candidate_range = (min(counts), max(counts))
    if len(ordered) < 2:
        return FactorRetentionResult(
            decision=FactorRetentionDecision.INSUFFICIENT_EVIDENCE,
            retained_count=None,
            candidate_range=candidate_range,
            evidence=ordered,
        )

    if candidate_range[0] == candidate_range[1]:
        return FactorRetentionResult(
            decision=FactorRetentionDecision.CONSENSUS,
            retained_count=candidate_range[0],
            candidate_range=candidate_range,
            evidence=ordered,
        )

    return FactorRetentionResult(
        decision=FactorRetentionDecision.DISAGREEMENT,
        retained_count=None,
        candidate_range=candidate_range,
        evidence=ordered,
    )


__all__ = [
    "MAX_FACTOR_CANDIDATE_COUNT",
    "FactorRetentionDecision",
    "FactorRetentionEvidence",
    "FactorRetentionMethod",
    "FactorRetentionResult",
    "govern_factor_retention",
]
