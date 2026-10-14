"""Criterion-neutral empirical selection of exploratory rotations.

Criterion objectives have incompatible scales. The selector therefore compares
solutions through common diagnostics and an explicit decision policy. Its
choice is conditional on the candidate set, extraction model, bootstrap design,
and policy, never a universally optimal criterion claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rotation_core_loader import rotation_core
from .rotation import (
    RotationSolution,
    _matrix,
    _method_name,
    _mode_name,
    _optional_matrix,
    _optional_trusted_integer,
    _optional_trusted_real,
    _solution_from_core,
    _trusted_boolean,
    _trusted_integer,
    _trusted_real,
)


@dataclass(frozen=True)
class RotationCandidateEvidence:
    """Criterion-neutral metrics for one candidate rotation."""

    criterion: str
    solution: RotationSolution
    row_complexity: float
    factor_balance: float
    max_factor_correlation: float
    convergence_rate: float
    basin_support_rate: float
    bootstrap_congruence: float
    bootstrap_min_congruence: float
    target_rmse: float
    policy_score: float
    pareto_optimal: bool


@dataclass(frozen=True)
class RotationSelectionResult:
    """Policy-conditional criterion choice and complete candidate evidence."""

    selected_index: int
    selected_criterion: str
    policy: str
    candidates: tuple[RotationCandidateEvidence, ...]
    bootstrap_replicates: int
    evidence_grade: str
    warning: str

    @property
    def selected(self) -> RotationCandidateEvidence:
        """Return evidence for the selected criterion."""

        return self.candidates[self.selected_index]


_POLICIES = frozenset(
    {
        "interpretability_first",
        "stability_first",
        "theory_guided",
        "fully_exploratory",
        "recovery_first",
        "sparse_simple_structure",
        "bifactor_discovery",
    }
)


def _policy_name(value: str) -> str:
    """Normalize and validate a decision policy identifier."""

    if type(value) is not str or not value.strip():
        raise ValueError("policy must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in _POLICIES:
        raise ValueError(f"policy must be one of {sorted(_POLICIES)}")
    return normalized


def _candidate_names(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return at least two distinct names from an inert candidate container."""

    if type(values) not in (list, tuple):
        raise ValueError("candidates must be an exact list or tuple of criterion names")
    normalized = tuple(_method_name(value) for value in values)
    if len(normalized) < 2:
        raise ValueError("criterion selection requires at least two candidates")
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidate criterion names must be unique")
    return normalized


def _bootstrap_array(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Validate optional ``(replicates, rows, factors)`` loading matrices."""

    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.ndim != 3 or tuple(array.shape[1:]) != shape:
        raise ValueError(
            f"bootstrap_loadings must have shape (replicates, {shape[0]}, {shape[1]})"
        )
    if array.shape[0] == 0:
        raise ValueError("bootstrap_loadings must contain at least one replicate")
    if not np.all(np.isfinite(array)):
        raise ValueError("bootstrap_loadings must contain only finite values")
    return array


def _theory_target(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Validate a finite-or-NaN theory target matrix."""

    if value is None:
        return None
    target = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if target.shape != shape:
        raise ValueError(f"theory_target must have shape {shape}")
    if np.any(np.isinf(target)):
        raise ValueError("theory_target may contain finite values or NaN, not infinity")
    if np.all(np.isnan(target)):
        raise ValueError("theory_target must specify at least one cell")
    return target


def _result_from_core(value: dict[str, object]) -> RotationSelectionResult:
    """Convert the Rust selection dictionary into immutable evidence objects."""

    raw_candidates = value["candidates"]
    candidates = tuple(
        RotationCandidateEvidence(
            criterion=str(candidate["criterion"]),
            solution=_solution_from_core(candidate["solution"]),
            row_complexity=float(candidate["row_complexity"]),
            factor_balance=float(candidate["factor_balance"]),
            max_factor_correlation=float(candidate["max_factor_correlation"]),
            convergence_rate=float(candidate["convergence_rate"]),
            basin_support_rate=float(candidate["basin_support_rate"]),
            bootstrap_congruence=float(candidate["bootstrap_congruence"]),
            bootstrap_min_congruence=float(candidate["bootstrap_min_congruence"]),
            target_rmse=float(candidate["target_rmse"]),
            policy_score=float(candidate["policy_score"]),
            pareto_optimal=bool(candidate["pareto_optimal"]),
        )
        for candidate in raw_candidates
    )
    return RotationSelectionResult(
        selected_index=int(value["selected_index"]),
        selected_criterion=str(value["selected_criterion"]),
        policy=str(value["policy"]),
        candidates=candidates,
        bootstrap_replicates=int(value["bootstrap_replicates"]),
        evidence_grade=str(value["evidence_grade"]),
        warning=str(value["warning"]),
    )


def select_rotation_criterion(
    loadings: np.ndarray,
    candidates: list[str] | tuple[str, ...],
    *,
    mode: str = "oblique",
    policy: str = "fully_exploratory",
    bootstrap_loadings: np.ndarray | None = None,
    theory_target: np.ndarray | None = None,
    normalize: bool = False,
    n_starts: int = 32,
    seed: int = 1,
    max_iter: int = 2_000,
    tolerance: float = 1e-5,
    function_window: int = 10,
    max_line_search: int = 20,
    basin_tolerance: float = 1e-8,
    max_threads: int = 0,
    kappa: float | None = None,
    gamma: float | None = None,
    delta: float | None = None,
    simplimax_zeros: int | None = None,
    target: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> RotationSelectionResult:
    """Select a criterion using neutral, policy-weighted empirical evidence.

    Candidate objective values are never compared directly. At least 20
    bootstrap loading matrices receive the ``bootstrap_supported`` evidence
    grade; fewer are explicitly exploratory.
    """

    names = _candidate_names(candidates)
    resolved_mode = _mode_name("geomin", mode)
    resolved_policy = _policy_name(policy)
    marshaled_normalize = _trusted_boolean(normalize, "normalize")
    marshaled_n_starts = _trusted_integer(n_starts, "n_starts")
    marshaled_seed = _trusted_integer(seed, "seed")
    marshaled_max_iter = _trusted_integer(max_iter, "max_iter")
    marshaled_tolerance = _trusted_real(tolerance, "tolerance")
    marshaled_function_window = _trusted_integer(function_window, "function_window")
    marshaled_max_line_search = _trusted_integer(max_line_search, "max_line_search")
    marshaled_basin_tolerance = _trusted_real(basin_tolerance, "basin_tolerance")
    marshaled_max_threads = _trusted_integer(max_threads, "max_threads")
    marshaled_kappa = _optional_trusted_real(kappa, "kappa")
    marshaled_gamma = _optional_trusted_real(gamma, "gamma")
    marshaled_delta = _optional_trusted_real(delta, "delta")
    marshaled_simplimax_zeros = _optional_trusted_integer(
        simplimax_zeros, "simplimax_zeros"
    )

    matrix = _matrix(loadings, "loadings")
    bootstraps = _bootstrap_array(bootstrap_loadings, matrix.shape)
    target_matrix = _optional_matrix(target, "target", matrix.shape, allow_nan=True)
    weight_matrix = _optional_matrix(weights, "weights", matrix.shape)
    if weight_matrix is not None and np.any(weight_matrix < 0.0):
        raise ValueError("weights must be non-negative")
    theory = _theory_target(theory_target, matrix.shape)
    result = rotation_core().select_rotation_criterion(
        matrix,
        list(names),
        resolved_mode,
        resolved_policy,
        marshaled_normalize,
        marshaled_n_starts,
        marshaled_seed,
        marshaled_max_iter,
        marshaled_tolerance,
        marshaled_function_window,
        marshaled_max_line_search,
        marshaled_basin_tolerance,
        marshaled_max_threads,
        marshaled_kappa,
        marshaled_gamma,
        marshaled_delta,
        marshaled_simplimax_zeros,
        target_matrix,
        weight_matrix,
        bootstraps,
        theory,
    )
    return _result_from_core(result)
