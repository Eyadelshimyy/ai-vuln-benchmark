"""End-to-end tests for the Rust longitudinal state estimator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from fast_mlsirm.multilevel import (
    LongitudinalStateKind,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
    fit_longitudinal_state,
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


def _single_occasion_design():
    """Return the smallest sealed design for public-boundary validation tests."""
    return build_longitudinal_design(
        occasions=[_occasion("respondent_a", "guard", 0, 0)],
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        ),
    )


def test_rust_state_fit_recovers_slopes_and_missing_values() -> None:
    """A two-person synthetic recovery fixture exercises multithreading."""
    occasions = [
        _occasion("respondent_a", "a0", 0, 0),
        _occasion("respondent_a", "a1", 1, 86_400_000),
        _occasion("respondent_a", "a2", 2, 172_800_000),
        _occasion("respondent_b", "b0", 0, 0),
        _occasion("respondent_b", "b1", 1, 86_400_000),
        _occasion("respondent_b", "b2", 2, 172_800_000),
    ]
    design = build_longitudinal_design(
        occasions=occasions,
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        ),
    )
    values = {
        "occasion_a0": 2.0,
        "occasion_a1": 3.5,
        "occasion_a2": 5.0,
        "occasion_b0": -1.0,
        "occasion_b1": -3.0,
    }
    result = fit_longitudinal_state(design, values, worker_count=4)
    single = fit_longitudinal_state(design, values, worker_count=1)
    np.testing.assert_allclose(result["intercepts"], [2.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(result["slopes"], [1.5, -2.0], atol=1e-12)
    np.testing.assert_array_equal(single["state"], result["state"])
    assert single["rmse"] == result["rmse"]
    assert result["observed_count"] == 5
    assert result["engine"] == "rust_cpu_multithreaded"
    assert result["state_kind"] == "random_intercept_slope"
    assert result["estimand_scope"] == "independent_respondent_ols_trend"
    assert result["population_random_effects_estimated"] is False
    assert result["ar_coefficient_estimated"] is False
    assert result["ar_coefficient_source"] == "not_applicable"
    assert len(result["design_fingerprint"]) == 64
    assert result["occasion_records"][0]["sequence_index"] == 0


def test_rust_state_fit_recovers_true_ols_parameters_with_rmse() -> None:
    """Known intercepts and slopes are recovered with bounded RMSE."""
    true_intercepts = (1.25, -0.5, 0.0)
    true_slopes = (0.75, -1.0, 0.25)
    occasions = []
    values: dict[str, float] = {}
    for respondent_index, respondent in enumerate(
        ("respondent_r0", "respondent_r1", "respondent_r2")
    ):
        intercept = true_intercepts[respondent_index]
        slope = true_slopes[respondent_index]
        for occasion_index in range(5):
            occasion_id = f"{respondent}{occasion_index}"
            occasions.append(
                _occasion(
                    respondent,
                    occasion_id,
                    occasion_index,
                    occasion_index * 86_400_000,
                )
            )
            values[f"occasion_{occasion_id}"] = intercept + slope * occasion_index
    design = build_longitudinal_design(
        occasions=occasions,
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        ),
    )
    result = fit_longitudinal_state(design, values, worker_count=3)
    intercept_rmse = float(
        np.sqrt(np.mean((np.asarray(result["intercepts"]) - true_intercepts) ** 2))
    )
    slope_rmse = float(np.sqrt(np.mean((np.asarray(result["slopes"]) - true_slopes) ** 2)))
    assert intercept_rmse < 1e-12
    assert slope_rmse < 1e-12
    assert result["rmse"] < 1e-12
    assert result["observed_count"] == 15


def test_all_missing_respondent_has_zero_rmse() -> None:
    """An all-missing design uses the intercept-only empty-observation branch."""
    design = build_longitudinal_design(
        occasions=[
            _occasion("respondent_a", "a0", 0, 0),
            _occasion("respondent_a", "a1", 1, 86_400_000),
        ],
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE,
        ),
    )
    empty = fit_longitudinal_state(design, {}, worker_count=3)
    assert empty["observed_count"] == 0
    assert empty["rmse"] == 0.0
    np.testing.assert_allclose(empty["intercepts"], [0.0])
    np.testing.assert_allclose(empty["slopes"], [0.0])


def test_rust_state_fit_preserves_discrete_ar_and_irregular_time() -> None:
    """An AR fixture verifies the explicit discrete-step contract."""
    design = build_longitudinal_design(
        occasions=[
            _occasion("respondent_a", "a0", 0, 0),
            _occasion("respondent_a", "a1", 1, 86_400_000),
            _occasion("respondent_a", "a2", 4, 259_200_000),
        ],
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.STATIONARY_AUTOREGRESSIVE,
            autoregressive_coefficient=0.5,
        ),
    )
    result = fit_longitudinal_state(
        design,
        {"occasion_a0": 1.0, "occasion_a1": 0.5, "occasion_a2": 0.125},
    )
    assert result["ar_coefficient"] == 0.5
    assert result["transition_count"] == 2
    assert result["estimand_scope"] == "discrete_ar_state_prediction"
    assert result["population_random_effects_estimated"] is False
    assert result["ar_coefficient_estimated"] is False
    assert result["ar_coefficient_source"] == "caller_supplied"
    # The third declared occasion is sequence three steps after the second;
    # the discrete-step AR(1) therefore predicts 0.5**3 * 0.5 = 0.0625.
    np.testing.assert_allclose(result["state"], [1.0, 0.5, 0.0625], atol=1e-12)


def test_rust_state_fit_recovers_true_ar_predictions_with_rmse() -> None:
    """A caller-supplied AR series is recovered with near-zero prediction RMSE."""
    phi = 0.4
    start = 1.6
    values = {"occasion_a0": start}
    occasions = [_occasion("respondent_a", "a0", 0, 0)]
    current = start
    for step in range(1, 6):
        current *= phi
        occasion_id = f"a{step}"
        occasions.append(_occasion("respondent_a", occasion_id, step, step * 86_400_000))
        values[f"occasion_{occasion_id}"] = current
    design = build_longitudinal_design(
        occasions=occasions,
        state_spec=build_longitudinal_state_spec(
            state_kind=LongitudinalStateKind.STATIONARY_AUTOREGRESSIVE,
            autoregressive_coefficient=phi,
        ),
    )
    result = fit_longitudinal_state(design, values)
    assert result["rmse"] < 1e-12
    assert result["transition_count"] == 5
    np.testing.assert_allclose(result["state"][0], start, atol=1e-12)


def test_state_fit_rejects_invalid_worker_count_and_foreign_design() -> None:
    """The public estimator validates execution controls before Rust dispatch."""
    design = _single_occasion_design()
    with pytest.raises(ValueError, match="worker_count"):
        fit_longitudinal_state(design, {}, worker_count=0)
    with pytest.raises(ValueError, match="worker_count"):
        fit_longitudinal_state(design, {}, worker_count=10**400)

    class ForeignDesign:
        """Represent an object that was not produced by the package factory."""

    with pytest.raises(ValueError, match="LongitudinalDesign"):
        fit_longitudinal_state(ForeignDesign(), {})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, np.bool_(True), 1 + 2j, "not-a-number"])
def test_state_fit_rejects_non_real_observation_values(value: object) -> None:
    """Caller-controlled observations fail with the package-owned exception."""
    with pytest.raises(ValueError, match="must be a real number"):
        fit_longitudinal_state(
            _single_occasion_design(),
            {"occasion_guard": value},  # type: ignore[dict-item]
        )


def test_state_fit_translates_hostile_mapping_reads_to_value_error() -> None:
    """A hostile mapping cannot leak its implementation exception."""

    class ExplodingMapping(Mapping[str, float]):
        """Raise from every read path to model an adversarial mapping."""

        def __getitem__(self, key: str) -> float:
            raise RuntimeError(f"blocked read: {key}")

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

        def get(self, key: str, default=None):
            raise RuntimeError(f"blocked get: {key}")

    with pytest.raises(ValueError, match="plain read-only mapping"):
        fit_longitudinal_state(_single_occasion_design(), ExplodingMapping())


@pytest.mark.parametrize("value", [2, np.int64(2), 2.5, np.float64(2.5)])
def test_state_fit_accepts_real_numeric_scalars(value: object) -> None:
    """Supported Python and NumPy real scalars convert through the helper."""
    result = fit_longitudinal_state(
        _single_occasion_design(),
        {"occasion_guard": value},  # type: ignore[dict-item]
    )
    np.testing.assert_allclose(result["intercepts"], [float(value)])
    assert result["observed_count"] == 1


@pytest.mark.parametrize("value", [10**400, float("inf"), float("-inf")])
def test_state_fit_rejects_unrepresentable_or_infinite_observations(value: object) -> None:
    """Overflow and infinities cannot cross the longitudinal numeric boundary."""
    with pytest.raises(ValueError, match="finite"):
        fit_longitudinal_state(
            _single_occasion_design(),
            {"occasion_guard": value},  # type: ignore[dict-item]
        )
