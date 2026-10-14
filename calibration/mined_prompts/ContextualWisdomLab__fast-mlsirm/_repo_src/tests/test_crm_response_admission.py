"""Data-integrity regressions for continuous-response-model response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import crm


def _result() -> dict[str, object]:
    """Return a minimal shape-consistent trusted-core CRM result."""

    return {
        "slope": [1.0],
        "intercept": [0.0],
        "resid_sd": [1.0],
        "discrimination": [1.0],
        "difficulty": [0.0],
        "theta": [0.0, 0.0],
        "loglik_trace": [0.0],
        "n_iter": 1,
        "converged": True,
        "n_parameters": 3,
        "termination_reason": "tolerance",
        "final_delta": 0.0,
        "stopping_tolerance": 1e-6,
    }


def test_complex_responses_fail_before_lossy_cast_or_native_discovery(monkeypatch):
    """A non-zero imaginary response must never be projected onto the real axis."""

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered after lossy complex coercion")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    responses = np.array([[0.25 + 0.0j], [0.75 + 1.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        crm.fit_crm(responses)


def test_object_complex_responses_use_package_error_before_native_discovery(monkeypatch):
    """Object storage containing complex evidence must fail at package admission."""

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before object-complex rejection")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    responses = np.array([[0.25], [0.75 + 1.0j]], dtype=object)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        crm.fit_crm(responses)


def test_array_provider_rejected_without_protocol_or_native_execution(monkeypatch):
    """Caller array protocols cannot synthesize the observed CRM response matrix."""

    callbacks = 0

    class HostileArrayProvider:
        def __array__(self, dtype=None):
            del dtype
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller __array__ executed during CRM admission")

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before provider rejection")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)

    with pytest.raises(ValueError, match="trusted NumPy array or built-in response matrix"):
        crm.fit_crm(HostileArrayProvider())

    assert callbacks == 0


def test_numeric_subclass_rejected_without_conversion_or_native_execution(monkeypatch):
    """Caller numeric subclasses cannot execute conversion while ratings are admitted."""

    callbacks = 0

    class HostileFloat(float):
        def __float__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller __float__ executed during CRM admission")

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before numeric-subclass rejection")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)

    with pytest.raises(ValueError, match="trusted NumPy array or built-in response matrix"):
        crm.fit_crm([[HostileFloat(0.25)], [0.75]])

    assert callbacks == 0


def test_oversized_exact_response_array_fails_before_value_scan_or_native_discovery(monkeypatch):
    """Logical size is bounded before value-wise NumPy work on broadcast evidence."""

    responses = np.broadcast_to(np.array([[0.5]], dtype=np.float64), (20_000_001, 1))

    def unexpected_value_scan(*args, **kwargs):
        del args, kwargs
        raise AssertionError("value-wise NumPy work ran before CRM resource admission")

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before CRM resource admission")

    monkeypatch.setattr(crm.np, "iscomplexobj", unexpected_value_scan)
    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)

    with pytest.raises(ValueError, match="20,000,000"):
        crm.fit_crm(responses)


def test_oversized_nested_numpy_rows_fail_before_sequence_materialization(monkeypatch):
    """Exact NumPy leaves are charged by logical occurrence before stacking."""

    row = np.broadcast_to(np.array([0.5], dtype=np.float64), (10_000_001,))
    responses = [row, row]

    def unexpected_asarray(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NumPy sequence materialization ran before CRM resource admission")

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before CRM resource admission")

    monkeypatch.setattr(crm.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)

    with pytest.raises(ValueError, match="20,000,000"):
        crm.fit_crm(responses)


def test_zero_cell_fanout_is_bounded_before_sequence_materialization(monkeypatch):
    """Malformed empty-container fan-out is bounded independently of scalar cells."""

    responses = [[] for _ in range(9)]

    def unexpected_asarray(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NumPy materialization ran before CRM structural admission")

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before CRM structural admission")

    monkeypatch.setattr(crm, "_MAX_CRM_RESPONSE_STRUCTURAL_NODES", 8)
    monkeypatch.setattr(crm.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)

    with pytest.raises(ValueError, match="structural traversal budget"):
        crm.fit_crm(responses)


def test_valid_builtin_matrix_fits_tight_structural_budget(monkeypatch):
    """The 2N structural envelope preserves every valid non-empty 2-D matrix."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_crm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(crm, "_MAX_CRM_RESPONSE_CELLS", 2)
    monkeypatch.setattr(crm, "_MAX_CRM_RESPONSE_STRUCTURAL_NODES", 4)
    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())

    fitted = crm.fit_crm([[0.25], [0.75]], max_iter=1)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.25, 0.75], dtype=np.float64))
    np.testing.assert_array_equal(args[1], np.array([True, True], dtype=bool))
    assert args[2:4] == (2, 1)
    assert fitted.n_parameters == 3


def test_builtin_response_matrix_preserves_trusted_numpy_scalar_marshalling(monkeypatch):
    """Ordinary built-in rows with concrete NumPy reals remain compatible."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_crm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = crm.fit_crm([[np.float32(0.25)], [np.float64(0.75)]], max_iter=1)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.25, 0.75], dtype=np.float64))
    np.testing.assert_array_equal(args[1], np.array([True, True], dtype=bool))
    assert args[2:4] == (2, 1)
    assert fitted.n_parameters == 3


def test_builtin_matrix_preserves_exact_numpy_row_compatibility(monkeypatch):
    """Exact numeric ndarray rows remain valid inside an inert built-in matrix."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_crm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = [
        np.array([0.25], dtype=np.float32),
        np.array([0.75], dtype=np.float64),
    ]
    fitted = crm.fit_crm(responses, max_iter=1)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.25, 0.75], dtype=np.float64))
    np.testing.assert_array_equal(args[1], np.array([True, True], dtype=bool))
    assert args[2:4] == (2, 1)
    assert fitted.n_parameters == 3


def test_text_response_storage_rejected_before_native_discovery(monkeypatch):
    """Textual numerics are evidence strings, not continuous response values."""

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered before text-storage rejection")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)

    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        crm.fit_crm(np.array([["0.25"], ["0.75"]]))


@pytest.mark.parametrize("infinite", [np.inf, -np.inf])
def test_infinite_responses_are_not_reclassified_as_missing(monkeypatch, infinite):
    """Only NaN is missing; either sign of infinity is invalid observed evidence."""

    def unexpected_core() -> object:
        raise AssertionError("compiled core discovered after infinity was treated as missing")

    monkeypatch.setattr(fitstats, "_core_module", unexpected_core)
    responses = np.array([[0.25], [infinite]], dtype=np.float64)

    with pytest.raises(ValueError, match="responses may only use NaN for missing values"):
        crm.fit_crm(responses)


def test_real_responses_preserve_existing_native_marshalling(monkeypatch):
    """Ordinary real-valued response arrays keep the existing Rust dispatch shape."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        def fit_crm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = crm.fit_crm(np.array([[0.25], [0.75]], dtype=np.float32), max_iter=1)

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0.25, 0.75], dtype=np.float64))
    np.testing.assert_array_equal(args[1], np.array([True, True], dtype=bool))
    assert args[2:4] == (2, 1)
    assert fitted.n_parameters == 3
