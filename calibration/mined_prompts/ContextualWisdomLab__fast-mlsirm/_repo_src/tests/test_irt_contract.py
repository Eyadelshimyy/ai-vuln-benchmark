"""Cross-component IRT response shape contract."""

from __future__ import annotations

import re

import numpy as np
import pytest
from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit as fit_binary
from fast_mlsirm.grm import fit_grm
from fast_mlsirm.irt_contract import (
    fit_irt_experiment,
    validate_irt_experiment_readiness,
    validate_irt_response_matrix,
)
from fast_mlsirm.models import confirmatory
from fast_mlsirm.polytomous import fit_polytomous
from fast_mlsirm.rsm import fit_rsm
from fast_mlsirm.twopl import fit_2pl


def test_dichotomous_contract_requires_multiple_items() -> None:
    matrix = validate_irt_response_matrix(
        [[0, 1], [1, np.nan]],
        "dichotomous",
    )
    assert matrix.shape == (2, 2)
    with pytest.raises(ValueError, match="at least two item columns"):
        validate_irt_response_matrix([[1]], "dichotomous")


def test_dichotomous_contract_rejects_non_binary_observations() -> None:
    with pytest.raises(ValueError, match="0, 1"):
        validate_irt_response_matrix([[0, 2]], "dichotomous")
    with pytest.raises(ValueError, match="integer"):
        validate_irt_response_matrix([[0.5, 1]], "dichotomous")


def test_polytomous_contract_requires_explicit_categories_and_multiple_items() -> None:
    matrix = validate_irt_response_matrix(
        [[0, 2], [1, np.nan]],
        "polytomous",
        n_categories=3,
    )
    assert matrix.shape == (2, 2)
    with pytest.raises(ValueError, match="n_categories"):
        validate_irt_response_matrix([[0, 1]], "polytomous")
    with pytest.raises(ValueError, match="at least two item columns"):
        validate_irt_response_matrix([[1]], "polytomous", n_categories=3)


def test_polytomous_contract_rejects_invalid_categories_and_shape() -> None:
    with pytest.raises(ValueError, match=re.escape("0..2")):
        validate_irt_response_matrix([[0, 3]], "polytomous", n_categories=3)
    with pytest.raises(ValueError, match="n_categories is only valid"):
        validate_irt_response_matrix([[0, 1]], "dichotomous", n_categories=2)
    with pytest.raises(ValueError, match=re.escape("2..")):
        validate_irt_response_matrix([[0, 1]], "polytomous", n_categories=1)
    with pytest.raises(ValueError, match="2-D"):
        validate_irt_response_matrix([0, 1], "polytomous", n_categories=3)
    with pytest.raises(ValueError, match="finite"):
        validate_irt_response_matrix([[0, np.inf]], "polytomous", n_categories=3)

    class _ForgedInt(int):
        def __le__(self, other):
            return True

        def __ge__(self, other):
            return True

    with pytest.raises(ValueError, match="n_categories"):
        validate_irt_response_matrix(
            [[0, 1], [1, 0]],
            "polytomous",
            n_categories=_ForgedInt(10**100),
        )
    with pytest.raises(ValueError, match="item_type"):
        validate_irt_response_matrix([[0, 1], [1, 0]], [], n_categories=2)


def test_irt_experiment_readiness_enforces_min_persons_items_and_variation() -> None:
    matrix = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    assert validate_irt_experiment_readiness(matrix, "dichotomous", min_persons=5).shape == (
        5,
        2,
    )
    with pytest.raises(ValueError, match=r"at least .* persons"):
        validate_irt_experiment_readiness(
            [[0, 1], [1, 0]],
            "dichotomous",
            min_persons=5,
        )
    with pytest.raises(ValueError, match=r"at least .* non-missing"):
        validate_irt_experiment_readiness(
            [[np.nan, 1], [np.nan, 0], [1, 1], [np.nan, np.nan], [0, 1]],
            "dichotomous",
            min_observed_per_item=3,
        )


def test_irt_experiment_readiness_requires_observed_item_variation() -> None:
    with pytest.raises(ValueError, match="distinct observed"):
        validate_irt_experiment_readiness(
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
            "dichotomous",
            min_persons=5,
        )


def test_polytomous_readiness_requires_declared_category_occupancy() -> None:
    with pytest.raises(ValueError, match=r"missing categories \[2\]"):
        validate_irt_experiment_readiness(
            [[0, 0], [1, 1], [0, 2], [1, 0], [0, 1]],
            "polytomous",
            n_categories=3,
            min_persons=5,
        )

    ready = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]]
    assert validate_irt_experiment_readiness(
        ready,
        "polytomous",
        n_categories=3,
        min_persons=5,
    ).shape == (5, 2)


def test_irt_experiment_readiness_checks_factor_coverage() -> None:
    matrix = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]
    validate_irt_experiment_readiness(
        matrix,
        "dichotomous",
        min_persons=6,
        factor_ids=("g", "f"),
        min_items_per_factor=1,
    )
    with pytest.raises(ValueError, match="factor_ids"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=6,
            factor_ids=("g",),
        )
    with pytest.raises(ValueError, match="under-covered"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=6,
            factor_ids=("g", "g"),
            min_items_per_factor=3,
        )


def test_fit_irt_experiment_blocks_before_call_and_passes_ready_matrix() -> None:
    called: list[np.ndarray] = []

    def fake_fit(responses, **kwargs):
        called.append(responses)
        return kwargs["result"]

    with pytest.raises(ValueError, match="distinct observed"):
        fit_irt_experiment(
            fake_fit,
            np.zeros((5, 2)),
            "dichotomous",
            factor_ids=(0, 0),
            result="unreachable",
        )
    assert called == []

    ready = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    assert (
        fit_irt_experiment(
            fake_fit,
            ready,
            "dichotomous",
            factor_ids=(0, 0),
            result="ready",
        )
        == "ready"
    )
    assert len(called) == 1
    np.testing.assert_array_equal(called[0], ready)


def test_fit_irt_experiment_supports_polytomous_multi_item_results() -> None:
    ready = np.array([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]])
    result = fit_irt_experiment(
        lambda responses, **_: responses.shape,
        ready,
        "polytomous",
        n_categories=3,
        factor_ids=((0,), (0,)),
    )
    assert result == (5, 2)


def test_fit_irt_experiment_normalizes_missing_semantics_before_readiness() -> None:
    called: list[np.ndarray] = []

    def fake_fit(responses, **kwargs):
        called.append(responses)
        return kwargs["result"]

    raw = np.array(
        [[0, 1], [1, 0], [0, 1], [-1, 0], [1, 1]],
        dtype=float,
    )
    mask = np.array(
        [[True, True], [True, True], [True, True], [True, True], [False, True]],
        dtype=bool,
    )
    assert (
        fit_irt_experiment(
            fake_fit,
            raw,
            "dichotomous",
            factor_ids=(0, 0),
            mask=mask,
            result="normalized",
        )
        == "normalized"
    )
    expected = np.array([[0, 1], [1, 0], [0, 1], [np.nan, 0], [np.nan, 1]])
    np.testing.assert_array_equal(np.isnan(called[0]), np.isnan(expected))
    np.testing.assert_array_equal(
        np.nan_to_num(called[0], nan=0), np.nan_to_num(expected, nan=0)
    )


def test_fit_irt_experiment_consumes_raw_mask_before_fitter() -> None:
    """The numerical fitter receives normalized NaNs, never the raw mask object."""
    received_kwargs: list[dict[str, object]] = []

    def fake_fit(_responses, **kwargs):
        received_kwargs.append(kwargs)
        return "normalized"

    raw = np.array(
        [[0, 1], [1, 0], [0, 1], [-1, 0], [1, 1]],
        dtype=float,
    )
    mask = np.array(
        [[True, True], [True, True], [True, True], [True, True], [False, True]],
        dtype=bool,
    )

    assert (
        fit_irt_experiment(
            fake_fit,
            raw,
            "dichotomous",
            factor_ids=(0, 0),
            mask=mask,
            result="normalized",
        )
        == "normalized"
    )
    assert received_kwargs == [{"result": "normalized"}]


def test_fit_irt_experiment_reports_configured_distinct_value_threshold() -> None:
    with pytest.raises(ValueError, match="at least 3 distinct observed"):
        validate_irt_experiment_readiness(
            [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
            "dichotomous",
            min_persons=5,
            min_item_distinct_values=3,
        )


def test_irt_experiment_readiness_rejects_unsafe_controls_and_factor_shapes() -> None:
    matrix = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    with pytest.raises(TypeError, match="min_items_per_factor"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=5,
            factor_ids=np.array(["a", "b"]),
            min_items_per_factor=True,
        )
    with pytest.raises(ValueError, match="factor_ids"):
        validate_irt_experiment_readiness(
            matrix,
            "dichotomous",
            min_persons=5,
            factor_ids="ab",
        )
    with pytest.raises(ValueError, match="cannot exceed n_categories"):
        validate_irt_experiment_readiness(
            matrix,
            "polytomous",
            n_categories=2,
            min_persons=5,
            min_item_distinct_values=3,
        )


def test_loading_pattern_memberships_are_checked_before_native_fit(monkeypatch) -> None:
    def native_must_not_run():
        raise AssertionError("native core must not run for an unready experiment")

    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", native_must_not_run)
    binary = np.array(
        [[0, 1], [1, 0], [0, 1], [1, 0]],
        dtype=float,
    )
    with pytest.raises(ValueError, match=r"at least .* persons"):
        fit_irt_experiment(
            fit_2pl,
            binary,
            "dichotomous",
            factor_ids=(0, 0),
            q=7,
        )

    binary_with_undercovered_factor = np.tile(
        np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=float), (3, 1)
    )
    pattern = confirmatory(
        np.array(
            [[1, 0], [1, 0], [1, 0], [0, 1]],
            dtype=np.int64,
        )
    )
    with pytest.raises(ValueError, match="under-covered"):
        fit_irt_experiment(
            fit_2pl,
            binary_with_undercovered_factor,
            "dichotomous",
            factor_ids=((0,), (0,), (0,), (1,)),
            model=pattern,
            q=7,
        )


def test_mlsirm_readiness_uses_missing_mask_before_fit() -> None:
    responses = np.array(
        [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        dtype=float,
    )
    responses[:3, 0] = np.nan
    with pytest.raises(ValueError, match="non-missing"):
        fit_irt_experiment(
            fit_binary,
            responses,
            "dichotomous",
            factor_ids=(0, 0),
            factor_id=np.array([0, 0], dtype=np.int64),
            config=FitConfig(max_iter=1),
        )


@pytest.mark.parametrize(
    "fit_call",
    [
        lambda y: fit_binary(y, np.array([0]), FitConfig(max_iter=1)),
        lambda y: fit_2pl(y, q=7, max_iter=1, model=1, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15),
        lambda y: fit_grm(y, n_cat=2, q=7, max_iter=1, model=1, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15),
        lambda y: fit_polytomous(y, n_cat=2, model="grm", q_theta=7, max_iter=1, tol=1e-6),
        lambda y: fit_rsm(y, n_cat=2, q_theta=7, max_iter=1, tol=1e-6),
    ],
)
def test_public_irt_fitters_reject_one_item_results(fit_call) -> None:
    with pytest.raises(ValueError, match="at least two item columns"):
        fit_call(np.array([[0.0], [1.0], [0.0]]))


if __name__ == "__main__":
    test_dichotomous_contract_requires_multiple_items()
    test_dichotomous_contract_rejects_non_binary_observations()
    test_polytomous_contract_requires_explicit_categories_and_multiple_items()
    test_polytomous_contract_rejects_invalid_categories_and_shape()
    print("ok")
