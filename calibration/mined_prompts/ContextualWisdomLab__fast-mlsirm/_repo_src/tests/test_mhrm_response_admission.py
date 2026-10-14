"""Data-integrity regressions for MH-RM public response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import mhrm


def _unexpected_core() -> object:
    """Fail if native capability discovery happens before response admission."""

    raise AssertionError("compiled core discovered before MH-RM response admission completed")


def _result() -> dict[str, object]:
    """Return a minimal shape-consistent trusted-core 2PL MH-RM result."""

    return {
        "loading": [1.0, 1.0],
        "intercept": [0.0, 0.0],
        "step": [],
        "n_cat": 2,
        "theta": [0.0, 0.0],
        "corr": [1.0],
        "se_loading": [],
        "se_intercept": [],
        "se_step": [],
        "acceptance_rate": 0.25,
        "n_cycles": 2,
        "converged": False,
        "termination_reason": "max_cycles_reached",
        "final_param_change": 0.1,
        "n_parameters": 4,
    }


def test_complex_responses_fail_before_lossy_cast_or_native_discovery(monkeypatch):
    """A non-zero imaginary response must never be projected onto a real category."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array(
        [[0.0 + 0.0j, 1.0 + 1.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="responses must be real-valued"):
        mhrm.fit_mhrm(responses, 1, max_cycles=2, burn_in=1, mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)


def test_array_provider_rejected_without_protocol_or_native_execution(monkeypatch):
    """Caller array protocols cannot synthesize the observed MH-RM response matrix."""

    callbacks = 0

    class HostileArrayProvider:
        def __array__(self, dtype=None):
            del dtype
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller __array__ executed during MH-RM response admission")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="trusted NumPy array or built-in response matrix"):
        mhrm.fit_mhrm(HostileArrayProvider(), 1, max_cycles=2, burn_in=1, mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)

    assert callbacks == 0


def test_numeric_subclass_rejected_without_conversion_or_native_execution(monkeypatch):
    """Caller numeric subclasses cannot execute conversion while responses are admitted."""

    callbacks = 0

    class HostileFloat(float):
        def __float__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller __float__ executed during MH-RM response admission")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="trusted NumPy array or built-in response matrix"):
        mhrm.fit_mhrm(
            [[HostileFloat(0.0), 1.0], [1.0, 0.0]],
            1,
            max_cycles=2,
            burn_in=1,
            mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
        )

    assert callbacks == 0


def test_object_and_text_storage_fail_before_conversion_or_native_discovery(monkeypatch):
    """Object/text storage is not implicitly converted into MH-RM category evidence."""

    callbacks = 0

    class HostileCell:
        def __float__(self):
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller __float__ executed during object response admission")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        mhrm.fit_mhrm(
            np.array([[0.0, 1.0], [1.0, HostileCell()]], dtype=object),
            1,
            max_cycles=2,
            burn_in=1,
            mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
        )
    assert callbacks == 0

    with pytest.raises(ValueError, match="responses must be a real numeric array"):
        mhrm.fit_mhrm(
            np.array([["0", "1"], ["1", "0"]]),
            1,
            max_cycles=2,
            burn_in=1,
            mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
        )


def test_builtin_response_matrix_preserves_trusted_numpy_scalar_marshalling(monkeypatch):
    """Built-in rows with concrete NumPy real scalars retain the integer Rust payload."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        """Capture trusted MH-RM arguments without running stochastic arithmetic."""

        def fit_mhrm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    fitted = mhrm.fit_mhrm(
        [[np.float32(0.0), np.int16(1)], [np.uint8(1), np.float64(0.0)]],
        1,
        max_cycles=2,
        burn_in=1,
        mh_steps=1,
        estimate_se=False, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
    )

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0, 1, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(args[1], np.array([True, True, True, True]))
    assert args[3:6] == (2, 2, 1)
    assert fitted.loading.shape == (2, 1)


def test_builtin_matrix_preserves_exact_numpy_row_compatibility(monkeypatch):
    """Exact numeric ndarray rows remain valid inside an inert built-in matrix."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        """Capture trusted MH-RM arguments without running stochastic arithmetic."""

        def fit_mhrm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = [
        np.array([0, 1], dtype=np.int16),
        np.array([1, 0], dtype=np.uint8),
    ]
    fitted = mhrm.fit_mhrm(
        responses,
        1,
        max_cycles=2,
        burn_in=1,
        mh_steps=1,
        estimate_se=False, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
    )

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0, 1, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(args[1], np.array([True, True, True, True]))
    assert args[3:6] == (2, 2, 1)
    assert fitted.loading.shape == (2, 1)


def test_real_responses_preserve_existing_native_marshalling(monkeypatch):
    """Ordinary real response matrices keep their existing integer Rust payload."""

    captured: dict[str, tuple[object, ...]] = {}

    class CapturingCore:
        """Capture trusted MH-RM arguments without running stochastic arithmetic."""

        def fit_mhrm(self, *args):
            captured["args"] = args
            return _result()

    monkeypatch.setattr(fitstats, "_core_module", lambda: CapturingCore())
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    fitted = mhrm.fit_mhrm(
        responses,
        1,
        max_cycles=2,
        burn_in=1,
        mh_steps=1,
        estimate_se=False, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15
    )

    args = captured["args"]
    np.testing.assert_array_equal(args[0], np.array([0, 1, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(args[1], np.array([True, True, True, True]))
    assert args[3:6] == (2, 2, 1)
    assert fitted.loading.shape == (2, 1)
    assert fitted.theta.shape == (2, 1)


def test_oversized_exact_response_view_fails_before_model_or_native_work(monkeypatch):
    """The Rust 200M-cell response ceiling is replayed before downstream dense work."""

    responses = np.broadcast_to(
        np.array([[0.0]], dtype=np.float64),
        (2, 100_000_001),
    )

    def unexpected_model(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model resolution reached before MH-RM response resource admission")

    monkeypatch.setattr(mhrm, "_resolve_model", unexpected_model)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="200,000,000"):
        mhrm.fit_mhrm(responses, 1, max_cycles=2, burn_in=1, mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)


def test_oversized_exact_numpy_row_fails_before_sequence_materialization(monkeypatch):
    """A huge exact ndarray row is bounded before a trusted built-in matrix is stacked."""

    row = np.broadcast_to(np.array([0.0], dtype=np.float64), (200_000_001,))

    def unexpected_asarray(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NumPy materialization reached before MH-RM response resource admission")

    monkeypatch.setattr(mhrm.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="200,000,000"):
        mhrm.fit_mhrm([row], 1, max_cycles=2, burn_in=1, mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)


def test_empty_container_fanout_fails_before_sequence_materialization(monkeypatch):
    """Malformed empty-container fan-out cannot bypass bounded MH-RM preflight work."""

    monkeypatch.setattr(mhrm, "_MAX_RESPONSE_CELLS", 2, raising=False)

    def unexpected_asarray(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NumPy materialization ran after unbounded MH-RM traversal")

    monkeypatch.setattr(mhrm.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = [[[]] for _ in range(7)]

    with pytest.raises(ValueError, match="structural traversal budget"):
        mhrm.fit_mhrm(responses, 1, max_cycles=2, burn_in=1, mh_steps=1, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)
