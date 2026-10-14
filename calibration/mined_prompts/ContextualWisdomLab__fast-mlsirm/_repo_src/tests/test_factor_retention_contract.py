"""Fail-first contract for the governed factor-retention evidence surface."""

from __future__ import annotations

import importlib.util

import pytest


def test_factor_retention_contract_module_exists() -> None:
    """Issue #608 requires a dedicated factor-retention contract namespace."""
    assert importlib.util.find_spec("fast_mlsirm.factor_retention") is not None


def _surface():
    """Import the planned surface only after the fail-first existence assertion."""
    from fast_mlsirm.factor_retention import (
        MAX_FACTOR_CANDIDATE_COUNT,
        FactorRetentionDecision,
        FactorRetentionEvidence,
        FactorRetentionMethod,
        govern_factor_retention,
    )

    return (
        MAX_FACTOR_CANDIDATE_COUNT,
        FactorRetentionDecision,
        FactorRetentionEvidence,
        FactorRetentionMethod,
        govern_factor_retention,
    )


def test_consensus_returns_one_supported_count() -> None:
    """Independent supported methods may agree without declaring one method best."""
    _, Decision, Evidence, Method, govern = _surface()

    result = govern(
        (
            Evidence(Method.PARALLEL_ANALYSIS, 3),
            Evidence(Method.VELICER_MAP, 3),
            Evidence(Method.PREDICTIVE, 3),
        )
    )

    assert result.decision is Decision.CONSENSUS
    assert result.retained_count == 3
    assert result.candidate_range == (3, 3)
    assert result.evidence_count == 3
    assert tuple(item.method for item in result.evidence) == (
        Method.PARALLEL_ANALYSIS,
        Method.PREDICTIVE,
        Method.VELICER_MAP,
    )


def test_disagreement_preserves_conservative_candidate_range() -> None:
    """Conflicting supported methods must not be collapsed to one retained count."""
    _, Decision, Evidence, Method, govern = _surface()

    result = govern(
        (
            Evidence(Method.PARALLEL_ANALYSIS, 2),
            Evidence(Method.BOOTSTRAP_LR, 4),
            Evidence(Method.LIKELIHOOD_INFORMATION_CRITERION, 3),
        )
    )

    assert result.decision is Decision.DISAGREEMENT
    assert result.retained_count is None
    assert result.candidate_range == (2, 4)


def test_single_method_is_insufficient_even_with_a_candidate() -> None:
    """One method can bound a candidate but cannot establish cross-method consensus."""
    _, Decision, Evidence, Method, govern = _surface()

    result = govern((Evidence(Method.EXTERNAL_SUPPORTED, 5),))

    assert result.decision is Decision.INSUFFICIENT_EVIDENCE
    assert result.retained_count is None
    assert result.candidate_range == (5, 5)
    assert result.evidence_count == 1


def test_empty_evidence_is_insufficient_without_a_range() -> None:
    """No supplied evidence yields no fabricated candidate count or range."""
    _, Decision, _, _, govern = _surface()

    result = govern(())

    assert result.decision is Decision.INSUFFICIENT_EVIDENCE
    assert result.retained_count is None
    assert result.candidate_range is None
    assert result.evidence_count == 0


@pytest.mark.parametrize("candidate_count", [True, False, 0, -1, 1.5, "2", None])
def test_candidate_count_must_be_a_positive_integer(candidate_count: object) -> None:
    """Transport validation rejects booleans and non-positive/non-integer counts."""
    _, _, Evidence, Method, _ = _surface()

    with pytest.raises(ValueError, match="positive integer"):
        Evidence(Method.PARALLEL_ANALYSIS, candidate_count)  # type: ignore[arg-type]


def test_candidate_count_has_a_fixed_resource_ceiling() -> None:
    """Caller-controlled retention counts cannot grow without a package bound."""
    maximum, _, Evidence, Method, _ = _surface()

    assert Evidence(Method.PARALLEL_ANALYSIS, maximum).candidate_count == maximum
    with pytest.raises(ValueError, match="exceeds maximum"):
        Evidence(Method.PARALLEL_ANALYSIS, maximum + 1)


def test_duplicate_method_evidence_is_rejected() -> None:
    """One method cannot be double-counted as independent retention evidence."""
    _, _, Evidence, Method, govern = _surface()

    with pytest.raises(ValueError, match="duplicate factor-retention method"):
        govern(
            (
                Evidence(Method.PARALLEL_ANALYSIS, 2),
                Evidence(Method.PARALLEL_ANALYSIS, 3),
            )
        )


def test_duplicate_method_is_rejected_before_outer_iterable_overread() -> None:
    """Duplicate detection must not first exhaust an attacker-controlled iterable."""
    _, _, Evidence, Method, govern = _surface()

    def evidence_stream():
        yield Evidence(Method.PARALLEL_ANALYSIS, 2)
        yield Evidence(Method.PARALLEL_ANALYSIS, 3)
        raise AssertionError("governance boundary over-read duplicate evidence")

    with pytest.raises(ValueError, match="duplicate factor-retention method"):
        govern(evidence_stream())


def test_constructor_rejects_untyped_method_identity() -> None:
    """Method identity is a closed governed enum rather than arbitrary text."""
    _, _, Evidence, _, _ = _surface()

    with pytest.raises(TypeError, match="FactorRetentionMethod"):
        Evidence("parallel_analysis", 2)  # type: ignore[arg-type]


def test_governance_rejects_non_evidence_entries() -> None:
    """The aggregation boundary accepts only package-owned evidence records."""
    _, _, _, _, govern = _surface()

    with pytest.raises(TypeError, match="FactorRetentionEvidence"):
        govern((object(),))
