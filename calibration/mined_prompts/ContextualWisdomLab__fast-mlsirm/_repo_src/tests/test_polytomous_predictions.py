"""Public Rust-owned GRM/GPCM prediction contract."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm._polytomous_prediction_admission as prediction_admission
import fast_mlsirm.polytomous as polytomous_module
from fast_mlsirm import (
    PolytomousFit,
    polytomous_category_probabilities,
    polytomous_expected_response,
)


class _ArrayProvider:
    def __array__(self, *args, **kwargs):
        raise AssertionError("caller __array__ callback executed")


@pytest.mark.parametrize(
    ("model", "cat_params"),
    [
        ("grm", np.array([[1.0, -1.0], [0.5, -0.5]])),
        ("gpcm", np.array([[0.2, -0.4], [-0.1, 0.3]])),
    ],
)
def test_public_polytomous_predictions_are_normalized_and_consistent(model, cat_params):
    fit = PolytomousFit(model, np.array([1.0, 0.8]), cat_params, 0.0, 0)
    theta = np.array([-1000.0, 0.0, 1000.0])

    probabilities = polytomous_category_probabilities(fit, theta)
    expected = polytomous_expected_response(fit, theta)

    assert probabilities.shape == (3, 2, 3)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=2), 1.0)
    assert np.allclose(expected, probabilities @ np.arange(3, dtype=np.float64))


def test_prediction_admission_preserves_fit_metadata_when_normalizing_evidence():
    """Admission must not erase package-owned fit metadata while normalizing arrays."""

    captured = {}

    def original(fit, theta):
        captured["fit"] = fit
        captured["theta"] = theta
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    module = SimpleNamespace(
        PolytomousFit=PolytomousFit,
        _polytomous_predictions=original,
    )
    prediction_admission.install(module)

    loglik_trace = np.array([-10.0, -8.5, -8.0], dtype=np.float64)
    thresholds = np.array([[0.25]], dtype=np.float64)
    fit = PolytomousFit(
        "gpcm",
        [np.float32(1.0)],
        [[np.float32(0.0)]],
        -8.0,
        7,
        converged=True,
        termination_reason="tolerance",
        loglik_trace=loglik_trace,
        final_delta=1e-7,
        stopping_tolerance=1e-6,
        thresholds=thresholds,
    )

    module._polytomous_predictions(fit, [np.float32(0.0)])
    trusted = captured["fit"]

    assert trusted.converged is True
    assert trusted.termination_reason == "tolerance"
    assert trusted.loglik_trace is loglik_trace
    assert trusted.final_delta == pytest.approx(1e-7)
    assert trusted.stopping_tolerance == pytest.approx(1e-6)
    assert trusted.thresholds is thresholds
    assert trusted.slope.dtype == np.float64
    assert trusted.cat_params.dtype == np.float64
    assert captured["theta"].dtype == np.float64


def test_public_polytomous_predictions_reject_invalid_grm_thresholds():
    fit = PolytomousFit("grm", np.array([1.0]), np.array([[-1.0, 1.0]]), 0.0, 0)
    with pytest.raises(ValueError, match="strictly decreasing"):
        polytomous_category_probabilities(fit, np.array([0.0]))


def test_public_polytomous_predictions_reject_category_count_above_fitter_ceiling_before_numpy(
    monkeypatch,
):
    """A manually constructed fit cannot exceed the fitter's 64-category domain."""

    fit = PolytomousFit("gpcm", np.array([1.0]), np.zeros((1, 64)), 0.0, 0)

    def unexpected_numpy_materialization(*args, **kwargs):
        raise AssertionError("out-of-domain category count reached NumPy materialization")

    def unexpected_core_discovery():
        raise AssertionError("out-of-domain category count reached compiled-core discovery")

    monkeypatch.setattr(prediction_admission.np, "asarray", unexpected_numpy_materialization)
    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match=r"n_cat must be in 2\.\.=64"):
        polytomous_category_probabilities(fit, np.array([0.0]))


def test_public_polytomous_predictions_bound_output_before_native_dispatch(monkeypatch):
    """Reject oversized logical prediction grids before Rust/output allocation."""

    fit = PolytomousFit("grm", np.array([1.0]), np.array([[0.0]]), 0.0, 0)
    theta = np.broadcast_to(np.array([0.0]), (10_000_001,))

    def unexpected_core_discovery():
        raise AssertionError("oversized prediction grid reached compiled-core discovery")

    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="20,000,000.*prediction"):
        polytomous_category_probabilities(fit, theta)


def test_public_polytomous_predictions_bound_output_before_dense_copy(monkeypatch):
    """Output-cell admission must precede contiguous materialization of broadcast theta."""

    fit = PolytomousFit("grm", np.array([1.0]), np.array([[0.0]]), 0.0, 0)
    theta = np.broadcast_to(np.array([0.0]), (10_000_001,))

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("oversized prediction grid reached contiguous materialization")

    def unexpected_core_discovery():
        raise AssertionError("oversized prediction grid reached compiled-core discovery")

    monkeypatch.setattr(prediction_admission.np, "ascontiguousarray", unexpected_copy)
    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="20,000,000.*prediction"):
        polytomous_category_probabilities(fit, theta)


@pytest.mark.parametrize("field", ["theta", "slope", "cat_params"])
def test_public_polytomous_predictions_reject_array_providers_before_callbacks(
    monkeypatch, field
):
    """Caller array protocols must not run while prediction evidence is admitted."""

    theta = np.array([0.0])
    slope = np.array([1.0])
    cat_params = np.array([[0.0]])
    if field == "theta":
        theta = _ArrayProvider()
    elif field == "slope":
        slope = _ArrayProvider()
    else:
        cat_params = _ArrayProvider()
    fit = PolytomousFit("grm", slope, cat_params, 0.0, 0)

    def unexpected_core_discovery():
        raise AssertionError("untrusted evidence reached compiled-core discovery")

    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="trusted NumPy array or built-in sequence"):
        polytomous_category_probabilities(fit, theta)


@pytest.mark.parametrize(
    ("field", "over_rank", "expected_rank"),
    [
        ("theta", [[0.0]], 1),
        ("slope", [[1.0]], 1),
        ("cat_params", [[[0.0]]], 2),
    ],
)
def test_public_polytomous_predictions_reject_over_rank_before_numpy_materialization(
    monkeypatch, field, over_rank, expected_rank
):
    """Known over-rank evidence must fail before NumPy can materialize it."""

    theta = np.array([0.0])
    slope = np.array([1.0])
    cat_params = np.array([[0.0]])
    if field == "theta":
        theta = over_rank
    elif field == "slope":
        slope = over_rank
    else:
        cat_params = over_rank
    fit = PolytomousFit("grm", slope, cat_params, 0.0, 0)

    def unexpected_numpy_materialization(*args, **kwargs):
        raise AssertionError("over-rank evidence reached NumPy materialization")

    def unexpected_core_discovery():
        raise AssertionError("over-rank evidence reached compiled-core discovery")

    monkeypatch.setattr(prediction_admission.np, "asarray", unexpected_numpy_materialization)
    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match=rf"{field} must be at most {expected_rank}-D"):
        polytomous_category_probabilities(fit, theta)


def test_public_polytomous_predictions_preserve_exact_numpy_row_sequences():
    """A trusted NumPy row nested in the 2-D category matrix remains compatible."""

    fit = PolytomousFit("gpcm", [1.0], [np.array([0.0])], 0.0, 0)
    probabilities = polytomous_category_probabilities(fit, [0.0])
    assert probabilities.shape == (1, 1, 2)
    assert np.allclose(probabilities.sum(axis=2), 1.0)


def test_public_polytomous_predictions_reject_complex_before_native_dispatch(monkeypatch):
    """Do not silently project complex theta evidence onto the real line."""

    fit = PolytomousFit("gpcm", np.array([1.0]), np.array([[0.0]]), 0.0, 0)

    def unexpected_core_discovery():
        raise AssertionError("complex evidence reached compiled-core discovery")

    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="real-valued"):
        polytomous_expected_response(fit, np.array([0.0 + 1.0j]))


def test_public_polytomous_predictions_reject_mixed_integer_precision_loss_before_numpy(
    monkeypatch,
):
    """Mixed sequence promotion must not erase integer identity before validation."""

    fit = PolytomousFit("gpcm", [1.0], [[0.0]], 0.0, 0)
    theta = [9_007_199_254_740_993, 0.0]

    def unexpected_numpy_materialization(*args, **kwargs):
        raise AssertionError("lossy mixed evidence reached NumPy materialization")

    def unexpected_core_discovery():
        raise AssertionError("lossy mixed evidence reached compiled-core discovery")

    monkeypatch.setattr(prediction_admission.np, "asarray", unexpected_numpy_materialization)
    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="exactly representable as float64"):
        polytomous_expected_response(fit, theta)


def test_public_polytomous_predictions_bound_joint_grid_before_float64_conversion(
    monkeypatch,
):
    """Joint output budget must be decided from shapes before float64 conversion."""

    theta = np.broadcast_to(np.array([0.0], dtype=np.float32), (2,))
    slope = np.broadcast_to(np.array([1.0], dtype=np.float32), (10_000_001,))
    cat_params = np.broadcast_to(
        np.array([[0.0]], dtype=np.float32), (10_000_001, 1)
    )
    fit = PolytomousFit("gpcm", slope, cat_params, 0.0, 0)

    def unexpected_numpy_materialization(*args, **kwargs):
        raise AssertionError("oversized joint grid reached NumPy materialization")

    def unexpected_core_discovery():
        raise AssertionError("oversized joint grid reached compiled-core discovery")

    monkeypatch.setattr(prediction_admission.np, "asarray", unexpected_numpy_materialization)
    monkeypatch.setattr(polytomous_module, "_core_module", unexpected_core_discovery)
    with pytest.raises(ValueError, match="20,000,000.*prediction"):
        polytomous_category_probabilities(fit, theta)
