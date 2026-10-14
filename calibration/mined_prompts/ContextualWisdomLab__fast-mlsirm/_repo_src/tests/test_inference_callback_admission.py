"""Regression coverage for callback-free inference evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm as fml
import fast_mlsirm._inference_admission_safety as admission_safety
from fast_mlsirm import _core
from fast_mlsirm.inference import (
    observed_information,
    second_order_test,
    standard_errors_from_vcov,
    vcov_from_hessian,
)


class _HostileArrayProvider:
    """Array provider that must never run during package admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller __array__ executed during inference admission")


class _HostileFloatProvider:
    """Real-scalar provider that must never run during control admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller __float__ executed during inference admission")


def _unexpected_native_dispatch(*args, **kwargs):
    raise AssertionError("invalid inference evidence reached Rust")


@pytest.mark.parametrize(
    ("native_name", "public_fn", "kwargs", "message"),
    [
        ("second_order_test", second_order_test, {}, "hessian"),
        ("vcov_from_hessian", vcov_from_hessian, {}, "hessian"),
        ("standard_errors_from_vcov", standard_errors_from_vcov, {}, "vcov"),
    ],
)
def test_inference_rejects_array_providers_without_callbacks(
    monkeypatch, native_name, public_fn, kwargs, message
):
    """Caller array protocols must not choose curvature/covariance evidence."""
    monkeypatch.setattr(_core, native_name, _unexpected_native_dispatch)
    evidence = _HostileArrayProvider()

    with pytest.raises(ValueError, match=message):
        public_fn(evidence, **kwargs)

    assert evidence.calls == 0


@pytest.mark.parametrize(
    ("native_name", "public_fn", "control_name"),
    [
        ("second_order_test", second_order_test, "tol"),
        ("vcov_from_hessian", vcov_from_hessian, "rcond"),
    ],
)
def test_inference_rejects_real_control_providers_before_matrix_work(
    monkeypatch, native_name, public_fn, control_name
):
    """Semantic controls must be sealed before caller matrix materialization."""
    monkeypatch.setattr(_core, native_name, _unexpected_native_dispatch)
    matrix = _HostileArrayProvider()
    control = _HostileFloatProvider()

    with pytest.raises(ValueError, match=control_name):
        public_fn(matrix, **{control_name: control})

    assert control.calls == 0
    assert matrix.calls == 0


@pytest.mark.parametrize(
    ("public_fn", "control_name", "control_value"),
    [
        (second_order_test, "tol", True),
        (second_order_test, "tol", np.bool_(True)),
        (vcov_from_hessian, "rcond", True),
        (vcov_from_hessian, "rcond", np.bool_(True)),
        (second_order_test, "tol", -1.0),
        (vcov_from_hessian, "rcond", -1.0),
    ],
)
def test_inference_rejects_invalid_real_controls_before_matrix_work(
    public_fn, control_name, control_value
):
    """Boolean/negative controls fail before any caller evidence protocol."""
    matrix = _HostileArrayProvider()

    with pytest.raises(ValueError, match=control_name):
        public_fn(matrix, **{control_name: control_value})

    assert matrix.calls == 0


@pytest.mark.parametrize(
    ("native_name", "public_fn"),
    [
        ("second_order_test", second_order_test),
        ("vcov_from_hessian", vcov_from_hessian),
        ("standard_errors_from_vcov", standard_errors_from_vcov),
    ],
)
def test_inference_rejects_oversized_numpy_matrix_before_dense_copy(
    monkeypatch, native_name, public_fn
):
    """Logical matrix size must be bounded before a broadcast view is copied."""
    side = 4_473  # side**2 == 20,007,729 cells, above the 20M support envelope.
    evidence = np.broadcast_to(np.array(1.0, dtype=np.float64), (side, side))
    monkeypatch.setattr(_core, native_name, _unexpected_native_dispatch)

    def _unexpected_dense_copy(*args, **kwargs):
        raise AssertionError("oversized inference evidence reached dense float64 materialization")

    monkeypatch.setattr(admission_safety.np, "ascontiguousarray", _unexpected_dense_copy)

    with pytest.raises(ValueError, match="resource limit"):
        public_fn(evidence)


@pytest.mark.parametrize(
    "public_fn",
    [second_order_test, vcov_from_hessian, standard_errors_from_vcov],
)
def test_inference_rejects_oversized_builtin_dimension_before_row_replay(public_fn):
    """An over-budget square dimension is knowable before built-in row traversal."""
    side = 4_473
    evidence = [()] * side

    with pytest.raises(ValueError, match="resource limit"):
        public_fn(evidence)


@pytest.mark.parametrize(
    ("native_name", "public_fn"),
    [
        ("second_order_test", second_order_test),
        ("vcov_from_hessian", vcov_from_hessian),
        ("standard_errors_from_vcov", standard_errors_from_vcov),
    ],
)
@pytest.mark.parametrize(
    "evidence",
    [
        [[2**53 + 1, 0], [0, 1]],
        np.array([[2**53 + 1, 0], [0, 1]], dtype=np.uint64),
    ],
)
def test_inference_rejects_lossy_matrix_normalization_before_rust(
    monkeypatch, native_name, public_fn, evidence
):
    """Curvature/covariance entries must keep exact identity through Rust f64 marshalling."""
    monkeypatch.setattr(_core, native_name, _unexpected_native_dispatch)

    with pytest.raises(ValueError, match="losslessly representable as float64"):
        public_fn(evidence)


def test_second_order_preserves_trusted_sequence_and_numpy_scalar_compatibility(monkeypatch):
    """Trusted inert evidence reaches Rust as package-owned float64 primitives."""
    captured: dict[str, object] = {}

    def _capture(matrix, tol):
        captured["matrix"] = matrix
        captured["tol"] = tol
        return {
            "passed": True,
            "min_eigenvalue": 1.0,
            "eigenvalues": np.array([1.0, 2.0], dtype=np.float64),
        }

    monkeypatch.setattr(_core, "second_order_test", _capture)
    result = second_order_test(
        [[np.int16(2), np.float32(0.0)], [0, np.uint8(1)]],
        tol=np.float32(1e-6),
    )

    matrix = captured["matrix"]
    assert type(matrix) is np.ndarray
    assert matrix.dtype == np.float64
    assert matrix.shape == (2, 2)
    assert type(captured["tol"]) is float
    assert result["passed"] is True


def test_inference_preserves_exact_zero_dim_numpy_real_controls(monkeypatch):
    """Exact 0-D numeric arrays remain inert scalar controls at the Rust boundary."""
    second_captured: dict[str, object] = {}
    vcov_captured: dict[str, object] = {}

    def _capture_second(matrix, tol):
        second_captured["matrix"] = matrix
        second_captured["tol"] = tol
        return {
            "passed": True,
            "min_eigenvalue": 1.0,
            "eigenvalues": np.array([1.0, 2.0], dtype=np.float64),
        }

    def _capture_vcov(matrix, rcond):
        vcov_captured["matrix"] = matrix
        vcov_captured["rcond"] = rcond
        return [1.0, 0.0, 0.0, 1.0]

    monkeypatch.setattr(_core, "second_order_test", _capture_second)
    monkeypatch.setattr(_core, "vcov_from_hessian", _capture_vcov)

    matrix = np.eye(2, dtype=np.float64)
    second_order_test(matrix, tol=np.array(np.float32(1e-6)))
    vcov_from_hessian(matrix, rcond=np.array(np.float64(1e-10)))

    assert type(second_captured["tol"]) is float
    assert second_captured["tol"] == float(np.float32(1e-6))
    assert type(vcov_captured["rcond"]) is float
    assert vcov_captured["rcond"] == float(np.float64(1e-10))


def test_package_level_inference_exports_use_the_guarded_callables():
    """Historical package aliases must not bypass installed inference admission."""
    assert fml.observed_information is observed_information
    assert fml.second_order_test is second_order_test
    assert fml.vcov_from_hessian is vcov_from_hessian
    assert fml.standard_errors_from_vcov is standard_errors_from_vcov
