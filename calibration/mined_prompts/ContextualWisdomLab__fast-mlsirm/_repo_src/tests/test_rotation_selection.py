"""Criterion-neutral rotation-selection tests."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rotation_selection import (
    RotationCandidateEvidence,
    RotationSelectionResult,
    select_rotation_criterion,
)


def _reference() -> np.ndarray:
    """Return a mixed two-factor matrix with a recoverable simple structure."""
    return np.asarray(
        [
            [0.72, 0.39],
            [0.65, 0.35],
            [0.60, 0.31],
            [-0.31, 0.70],
            [-0.28, 0.64],
            [-0.25, 0.58],
        ],
        dtype=np.float64,
    )


def _bootstraps(count: int) -> np.ndarray:
    """Return deterministic perturbed loading matrices."""
    base = _reference()
    outputs = []
    for replicate in range(count):
        offsets = np.arange(base.size, dtype=np.float64).reshape(base.shape) - 5.0
        outputs.append(base + (replicate + 1) * 0.0002 * offsets)
    return np.stack(outputs)


def test_selector_returns_complete_single_sample_evidence() -> None:
    """Selection returns every candidate, a Pareto flag, and explicit warning."""
    result = select_rotation_criterion(
        _reference(),
        ["varimax", "quartimax", "equamax"],
        mode="orthogonal",
        policy="fully-exploratory",
        n_starts=4,
        seed=9,
        max_threads=1,
    )
    assert isinstance(result, RotationSelectionResult)
    assert result.evidence_grade == "single_sample_diagnostic"
    assert result.bootstrap_replicates == 0
    assert "one loading matrix" in result.warning
    assert len(result.candidates) == 3
    assert isinstance(result.selected, RotationCandidateEvidence)
    assert result.selected.criterion == result.selected_criterion
    assert result.selected_index in range(3)
    assert any(candidate.pareto_optimal for candidate in result.candidates)
    assert all(np.isfinite(candidate.policy_score) for candidate in result.candidates)
    assert all(candidate.solution.mode == "orthogonal" for candidate in result.candidates)


def test_selector_uses_bootstrap_congruence_and_theory_target() -> None:
    """Twenty aligned bootstrap matrices produce supported target evidence."""
    target = np.asarray(
        [
            [0.8, 0.0],
            [0.7, 0.0],
            [0.6, 0.0],
            [0.0, 0.8],
            [0.0, 0.7],
            [0.0, 0.6],
        ],
        dtype=np.float64,
    )
    result = select_rotation_criterion(
        _reference(),
        ["varimax", "quartimax"],
        mode="orthogonal",
        policy="theory_guided",
        bootstrap_loadings=_bootstraps(20),
        theory_target=target,
        n_starts=3,
        seed=11,
        max_threads=1,
    )
    assert result.evidence_grade == "bootstrap_supported"
    assert result.bootstrap_replicates == 20
    assert "not a universal" in result.warning
    for candidate in result.candidates:
        assert 0.0 <= candidate.bootstrap_congruence <= 1.0
        assert 0.0 <= candidate.bootstrap_min_congruence <= 1.0
        assert candidate.target_rmse >= 0.0
        assert 0.0 <= candidate.factor_balance <= 1.0
        assert 0.0 <= candidate.convergence_rate <= 1.0
        assert 0.0 <= candidate.basin_support_rate <= 1.0


@pytest.mark.parametrize(
    "policy",
    [
        "interpretability_first",
        "stability_first",
        "recovery_first",
        "sparse_simple_structure",
    ],
)
def test_supported_policy_names_execute(policy: str) -> None:
    """Every general-purpose policy is executable without theory metadata."""
    result = select_rotation_criterion(
        _reference(),
        ["varimax", "quartimax"],
        mode="orthogonal",
        policy=policy,
        n_starts=2,
        max_iter=200,
        max_threads=1,
    )
    assert result.policy == policy


def test_selection_validation_is_fail_closed() -> None:
    """Malformed candidate, bootstrap, target, and policy contracts are rejected."""
    reference = _reference()
    with pytest.raises(ValueError, match="exact list or tuple"):
        select_rotation_criterion(reference, "varimax")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        select_rotation_criterion(reference, ["varimax"])
    with pytest.raises(ValueError, match="unique"):
        select_rotation_criterion(reference, ["varimax", "varimax"])
    with pytest.raises(ValueError, match="non-empty"):
        select_rotation_criterion(reference, ["varimax", ""], mode="orthogonal")
    with pytest.raises(ValueError, match="policy"):
        select_rotation_criterion(reference, ["varimax", "quartimax"], policy="magic")
    with pytest.raises(ValueError, match="non-empty"):
        select_rotation_criterion(reference, ["varimax", "quartimax"], policy="")
    with pytest.raises(ValueError, match="shape"):
        select_rotation_criterion(
            reference,
            ["varimax", "quartimax"],
            mode="orthogonal",
            bootstrap_loadings=np.zeros((2, 4, 2)),
        )
    with pytest.raises(ValueError, match="at least one"):
        select_rotation_criterion(
            reference,
            ["varimax", "quartimax"],
            mode="orthogonal",
            bootstrap_loadings=np.empty((0, 6, 2)),
        )
    invalid_bootstrap = np.zeros((2, 6, 2))
    invalid_bootstrap[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_rotation_criterion(
            reference,
            ["varimax", "quartimax"],
            mode="orthogonal",
            bootstrap_loadings=invalid_bootstrap,
        )
    with pytest.raises(ValueError, match="shape"):
        select_rotation_criterion(
            reference,
            ["varimax", "quartimax"],
            mode="orthogonal",
            theory_target=np.zeros((4, 2)),
        )
    infinite_target = np.zeros_like(reference)
    infinite_target[0, 0] = np.inf
    with pytest.raises(ValueError, match="not infinity"):
        select_rotation_criterion(
            reference,
            ["varimax", "quartimax"],
            mode="orthogonal",
            theory_target=infinite_target,
        )
    with pytest.raises(ValueError, match="at least one"):
        select_rotation_criterion(
            reference,
            ["varimax", "quartimax"],
            mode="orthogonal",
            theory_target=np.full_like(reference, np.nan),
        )
    with pytest.raises(ValueError, match="requires theory_target"):
        select_rotation_criterion(
            reference,
            ["varimax", "quartimax"],
            mode="orthogonal",
            policy="theory_guided",
        )


def test_selection_target_weight_validation_delegates_consistently() -> None:
    """Candidate-specific target metadata uses the same validation as rotation."""
    reference = _reference()
    negative = -np.ones_like(reference)
    with pytest.raises(ValueError, match="non-negative"):
        select_rotation_criterion(
            reference,
            ["pst", "geomin"],
            target=np.zeros_like(reference),
            weights=negative,
        )
    with pytest.raises(ValueError, match="requires target"):
        select_rotation_criterion(reference, ["target", "geomin"])
