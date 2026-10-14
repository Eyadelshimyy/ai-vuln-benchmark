"""End-to-end tests for the joint MAP hierarchical CT-AR Rasch slice."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from fast_mlsirm._multilevel_core_loader import multilevel_core
from fast_mlsirm.multilevel import estimation as estimation_module
from fast_mlsirm.multilevel import (
    LongitudinalStateKind,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
    fit_hierarchical_longitudinal_irt,
    simulate_hierarchical_longitudinal_irt,
)


def _occasion(respondent: str, occasion: str, sequence: int, offset: int):
    """Build one sealed occasion with a unique revision identity."""
    occasion_id = f"occasion_{occasion}"
    return build_temporal_occasion(
        respondent_id=respondent,
        occasion_id=occasion_id,
        sequence_index=sequence,
        time_offset_milliseconds=offset,
        occasion_revision_fingerprint=(occasion_id + " revision")
        .encode()
        .hex()
        .ljust(64, "0")[:64],
    )


def _design(n_persons: int, n_occasions: int, irregular: bool = False):
    """Return a sealed longitudinal design with optional irregular gaps."""
    occasions = []
    for person in range(n_persons):
        elapsed = 0
        for occasion in range(n_occasions):
            if irregular and occasion == 2:
                elapsed += 2
            elif occasion > 0:
                elapsed += 1
            occasions.append(
                _occasion(
                    f"respondent_{person}",
                    f"{person}_{occasion}",
                    occasion,
                    elapsed * 86_400_000,
                )
            )
    return build_longitudinal_design(
        occasions=occasions,
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        ),
    )


class _OversizedItemSequence(Sequence[float]):
    """Expose an oversized length without allowing item iteration."""

    def __getitem__(self, index: int) -> float:
        raise AssertionError(f"item sequence was materialized at index {index}")

    def __len__(self) -> int:
        return 4_097


def test_public_fit_recovers_states_across_seeds_and_is_worker_invariant() -> None:
    """Multi-seed recovery stays inside honest MAP RMSE/coverage bounds."""
    design = _design(8, 3, irregular=True)
    items = np.array([-0.6, -0.2, 0.2, 0.6], dtype=np.float64)
    state_sse = 0.0
    state_count = 0.0
    covered = 0.0
    mean_err = 0.0
    for seed in (11, 23, 41):
        simulated = simulate_hierarchical_longitudinal_irt(
            design,
            item_intercepts=items,
            population_mean=0.0,
            population_sd=0.7,
            decay_rate=0.35,
            seed=seed,
        )
        result = fit_hierarchical_longitudinal_irt(
            design,
            simulated["responses"],
            item_ids=["a", "b", "c", "d"],
            worker_count=3,
            max_iter=80,
            tolerance=1e-4,
        )
        single = fit_hierarchical_longitudinal_irt(
            design,
            simulated["responses"],
            item_ids=["a", "b", "c", "d"],
            worker_count=1,
            max_iter=80,
            tolerance=1e-4,
        )
        np.testing.assert_allclose(result["state"], single["state"], atol=1e-8)
        assert result["estimand_scope"] == "joint_map_hierarchical_ctar_rasch"
        assert result["transition_kind"] == "continuous_time_ar1_ou"
        assert result["interval_kind"] == (
            "wald_conditional_hyperparameter_observed_information"
        )
        assert result["engine"] == "rust_cpu_multithreaded"
        assert result["population_random_effects_estimated"] is True
        assert result["ar_coefficient_estimated"] is True
        assert result["ar_coefficient_source"] == "joint_map"
        assert result["multiple_membership_estimated"] is False
        assert result["gpu_parity"] is False
        assert result["item_ids"] == ["a", "b", "c", "d"]
        assert result["estimand_scope"] != "independent_respondent_ols_trend"
        assert result["state_intervals_identified"] is True
        truth = np.asarray(simulated["state"], dtype=np.float64)
        state_sse += float(np.sum((result["state"] - truth) ** 2))
        state_count += truth.size
        covered += float(
            np.sum(
                (result["state_lower"] <= truth) & (truth <= result["state_upper"])
            )
        )
        mean_err += (float(result["population_mean"]) - 0.0) ** 2
    state_rmse = (state_sse / state_count) ** 0.5
    coverage = covered / state_count
    mean_rmse = (mean_err / 3.0) ** 0.5
    assert state_rmse < 0.85, state_rmse
    assert coverage > 0.80, coverage
    assert mean_rmse < 0.35, mean_rmse


def test_missing_responses_and_irregular_gaps_are_honored() -> None:
    """NaN responses are excluded and longer gaps shrink the CT-AR weight."""
    design = _design(1, 3, irregular=True)
    responses = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [np.nan, np.nan, 0.0, 1.0],
            [0.0, 1.0, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    result = fit_hierarchical_longitudinal_irt(
        design,
        responses,
        worker_count=1,
        max_iter=60,
        tolerance=1e-4,
    )
    assert result["observed_count"] == 8
    assert result["transition_count"] == 2
    assert result["unit_time_ar_coefficient"] < 1.0
    assert result["item_ids"] == ["item_0", "item_1", "item_2", "item_3"]


def test_public_fit_rejects_invalid_controls_and_foreign_designs() -> None:
    """Python marshalling fails closed before native dispatch."""
    design = _design(1, 2)
    responses = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="worker_count"):
        fit_hierarchical_longitudinal_irt(design, responses, worker_count=0)
    with pytest.raises(ValueError, match="max_iter"):
        fit_hierarchical_longitudinal_irt(design, responses, max_iter=0)
    with pytest.raises(ValueError, match="tolerance"):
        fit_hierarchical_longitudinal_irt(design, responses, tolerance=0.0)
    with pytest.raises(ValueError, match="hessian_step"):
        fit_hierarchical_longitudinal_irt(design, responses, hessian_step=-1.0)
    with pytest.raises(
        ValueError, match="at least one respondent must have two or more occasions"
    ):
        fit_hierarchical_longitudinal_irt(
            _design(1, 1),
            np.array([[1.0, 0.0]], dtype=np.float64),
        )

    class ForeignDesign:
        """Represent an object that was not produced by the package factory."""

    with pytest.raises(ValueError, match="LongitudinalDesign"):
        fit_hierarchical_longitudinal_irt(ForeignDesign(), responses)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload, match",
    [
        (True, "NumPy ndarray"),
        (np.array([1.0, 0.0]), "two-dimensional"),
        (np.array([[1.0, 0.0]], dtype=np.float64), "align with the sealed"),
        (np.array([[1.0], [0.0]], dtype=np.float64), "at least two items"),
        (np.array([[True, False], [False, True]]), "Boolean"),
        (np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64), "0, 1, or NaN"),
        (np.array([["a", "b"], ["c", "d"]], dtype=object), "float64 safely"),
    ],
)
def test_public_fit_rejects_invalid_response_matrices(
    payload: object, match: str
) -> None:
    """Hostile or malformed response matrices raise package-owned errors."""
    with pytest.raises(ValueError, match=match):
        fit_hierarchical_longitudinal_irt(_design(1, 2), payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "item_ids, match",
    [
        ("item_a", "sequence of item identifiers"),
        (["a"], "length must equal"),
        (["a", ""], "non-empty strings"),
        (["a", 1], "non-empty strings"),
    ],
)
def test_public_fit_rejects_invalid_item_labels(
    item_ids: object, match: str
) -> None:
    """Item labels are validated before the native kernel is invoked."""
    responses = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match=match):
        fit_hierarchical_longitudinal_irt(
            _design(1, 2),
            responses,
            item_ids=item_ids,  # type: ignore[arg-type]
        )


def test_simulate_accepts_list_and_ndarray_item_intercepts() -> None:
    """Generating intercepts may be a Python sequence or a NumPy vector."""
    design = _design(1, 2)
    listed = simulate_hierarchical_longitudinal_irt(
        design, item_intercepts=[-0.2, 0.2], seed=2
    )
    arrayed = simulate_hierarchical_longitudinal_irt(
        design, item_intercepts=np.array([-0.2, 0.2], dtype=np.float64), seed=2
    )
    np.testing.assert_array_equal(listed["responses"], arrayed["responses"])
    np.testing.assert_array_equal(listed["state"], arrayed["state"])


def test_simulate_and_fit_reject_invalid_generating_controls() -> None:
    """The recovery simulator validates generating parameters locally."""
    design = _design(1, 2)
    with pytest.raises(ValueError, match="LongitudinalDesign"):
        simulate_hierarchical_longitudinal_irt(
            object(),  # type: ignore[arg-type]
            item_intercepts=[-0.2, 0.2],
        )
    with pytest.raises(ValueError, match="sequence of real numbers"):
        simulate_hierarchical_longitudinal_irt(design, item_intercepts="ab")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        simulate_hierarchical_longitudinal_irt(design, item_intercepts=[0.1])
    with pytest.raises(ValueError, match="finite"):
        simulate_hierarchical_longitudinal_irt(
            design, item_intercepts=[0.1, float("nan")]
        )
    with pytest.raises(ValueError, match="sum to zero"):
        simulate_hierarchical_longitudinal_irt(design, item_intercepts=[0.1, 0.2])
    for decay_rate in (float("nan"), 0.0, -0.1):
        with pytest.raises(ValueError, match="decay_rate"):
            simulate_hierarchical_longitudinal_irt(
                _design(1, 1), item_intercepts=[-0.2, 0.2], decay_rate=decay_rate
            )
    with pytest.raises(ValueError, match="non-negative"):
        simulate_hierarchical_longitudinal_irt(
            design, item_intercepts=[-0.2, 0.2], seed=-1
        )
    with pytest.raises(ValueError, match="converted safely"):
        simulate_hierarchical_longitudinal_irt(
            design, item_intercepts=[object(), object()]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("population_mean", 10**400),
        ("population_sd", 10**400),
        ("decay_rate", 10**400),
        ("seed", 10**400),
    ],
)
def test_simulator_rejects_unrepresentable_controls(name: str, value: object) -> None:
    """Simulator controls fail with package errors before native dispatch."""
    with pytest.raises(ValueError, match=name):
        simulate_hierarchical_longitudinal_irt(
            _design(1, 2),
            item_intercepts=[-0.2, 0.2],
            **{name: value},
        )


def test_raw_binding_bounds_hierarchical_axes() -> None:
    """The raw extension bounds occasion and item axes before native work."""
    core = multilevel_core()
    with pytest.raises(ValueError, match="row_offsets exceeds"):
        core.fit_hierarchical_ctar_rasch(
            np.zeros(100_002, dtype=np.uint64),
            np.array([0], dtype=np.int64),
            np.zeros((1, 2), dtype=np.float64),
            1,
            1,
            1e-4,
            1e-3,
        )
    with pytest.raises(ValueError, match="occasion axis exceeds"):
        core.fit_hierarchical_ctar_rasch(
            np.array([0, 1], dtype=np.uint64),
            np.array([0], dtype=np.int64),
            np.zeros((100_001, 2), dtype=np.float64),
            1,
            1,
            1e-4,
            1e-3,
        )
    with pytest.raises(ValueError, match="time offsets exceed"):
        core.fit_hierarchical_ctar_rasch(
            np.array([0, 1], dtype=np.uint64),
            np.zeros(100_001, dtype=np.int64),
            np.zeros((1, 2), dtype=np.float64),
            1,
            1,
            1e-4,
            1e-3,
        )
    with pytest.raises(ValueError, match="item axis exceeds"):
        core.fit_hierarchical_ctar_rasch(
            np.array([0, 1], dtype=np.uint64),
            np.array([0], dtype=np.int64),
            np.zeros((1, 4_097), dtype=np.float64),
            1,
            1,
            1e-4,
            1e-3,
        )
    with pytest.raises(ValueError, match="item_intercepts exceeds"):
        core.simulate_hierarchical_ctar_rasch(
            np.array([0, 1], dtype=np.uint64),
            np.array([0], dtype=np.int64),
            np.zeros(4_097, dtype=np.float64),
            0.0,
            0.5,
            0.4,
            1,
        )
    with pytest.raises(ValueError, match="row_offsets exceeds"):
        core.simulate_hierarchical_ctar_rasch(
            np.zeros(100_002, dtype=np.uint64),
            np.array([0], dtype=np.int64),
            np.array([-0.2, 0.2], dtype=np.float64),
            0.0,
            0.5,
            0.4,
            1,
        )
    with pytest.raises(ValueError, match="time offsets exceed"):
        core.simulate_hierarchical_ctar_rasch(
            np.array([0, 100_001], dtype=np.uint64),
            np.zeros(100_001, dtype=np.int64),
            np.array([-0.2, 0.2], dtype=np.float64),
            0.0,
            0.5,
            0.4,
            1,
        )


def test_python_wrapper_bounds_before_copying_or_materializing() -> None:
    """Python wrappers enforce native dimensions before allocation or iteration."""
    with pytest.raises(ValueError, match="occasion axis exceeds"):
        estimation_module._validate_binary_response_matrix(
            np.empty((100_001, 2), dtype=np.float64), 100_001
        )
    with pytest.raises(ValueError, match="item axis exceeds"):
        estimation_module._validate_binary_response_matrix(
            np.empty((1, 4_097), dtype=np.float64), 1
        )
    with pytest.raises(ValueError, match="item_intercepts exceeds"):
        simulate_hierarchical_longitudinal_irt(
            _design(1, 1), item_intercepts=_OversizedItemSequence()
        )
