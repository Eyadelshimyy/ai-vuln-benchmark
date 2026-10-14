"""Trust-boundary regressions for cognitive-diagnosis native controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import (
    fit_cdm,
    fit_gdina,
    fit_ho_cdm,
    fit_ho_gdina,
    fit_seq_gdina,
    fit_seq_gdina_qr,
    gdina_wald_selection,
    validate_q_matrix,
)


class _HostileInt(int):
    """Integer subclass whose coercion/comparison hooks must remain unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return int.__int__(self)

    def __index__(self):
        type(self).calls += 1
        return int.__index__(self)

    def __lt__(self, other):
        type(self).calls += 1
        return int.__lt__(self, other)

    def __le__(self, other):
        type(self).calls += 1
        return int.__le__(self, other)


class _HostileFloat(float):
    """Floating subclass whose conversion/comparison hooks must remain unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __float__(self):
        type(self).calls += 1
        return float.__float__(self)

    def __lt__(self, other):
        type(self).calls += 1
        return float.__lt__(self, other)

    def __le__(self, other):
        type(self).calls += 1
        return float.__le__(self, other)


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass whose hooks must remain unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return super().__int__()

    def __index__(self):
        type(self).calls += 1
        return super().__index__()


class _HostileNumpyFloat(np.float64):
    """NumPy floating subclass whose hooks must remain unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __float__(self):
        type(self).calls += 1
        return super().__float__()


class _NumberProvider:
    """Arbitrary numeric protocol provider that must never be normalized."""

    calls = 0

    def __init__(self, value):
        self.value = value

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __int__(self):
        type(self).calls += 1
        return int(self.value)

    def __index__(self):
        type(self).calls += 1
        return int(self.value)

    def __float__(self):
        type(self).calls += 1
        return float(self.value)

    def __lt__(self, other):
        type(self).calls += 1
        return self.value < other

    def __le__(self, other):
        type(self).calls += 1
        return self.value <= other


class _HostileStr(str):
    """String subclass whose representation hook must remain unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __str__(self):
        type(self).calls += 1
        return str.__str__(self)


_INTEGER_HOSTILES = (_HostileInt, _HostileNumpyInt, _NumberProvider)
_FLOAT_HOSTILES = (_HostileFloat, _HostileNumpyFloat, _NumberProvider)
_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def _binary_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return a small complete binary matrix and a matching Q-matrix."""
    responses = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    q_matrix = np.array([[1], [1], [1]], dtype=np.int64)
    return responses, q_matrix


def _three_attribute_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return valid inputs for higher-order CDM dispatch."""
    responses = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    q_matrix = np.eye(3, dtype=np.int64)
    return responses, q_matrix


def _sequential_qr_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid one-step-per-item sequential restricted-Q inputs."""
    responses, q_matrix = _binary_inputs()
    return responses, q_matrix, np.ones(responses.shape[1], dtype=np.int64)


def _entry_points() -> tuple[Callable[..., object], ...]:
    """Return every public CDM adapter that consumes shared stopping controls."""
    responses, q_matrix = _binary_inputs()
    ho_responses, ho_q = _three_attribute_inputs()
    seq_responses, step_q, n_steps = _sequential_qr_inputs()
    return (
        lambda **kw: fit_cdm(responses, q_matrix, **kw),
        lambda **kw: fit_gdina(responses, q_matrix, **kw),
        lambda **kw: validate_q_matrix(responses, q_matrix, **kw),
        lambda **kw: gdina_wald_selection(responses, q_matrix, **kw),
        lambda **kw: fit_ho_cdm(ho_responses, ho_q, **kw),
        lambda **kw: fit_ho_gdina(ho_responses, ho_q, **kw),
        lambda **kw: fit_seq_gdina(responses, q_matrix, **kw),
        lambda **kw: fit_seq_gdina_qr(seq_responses, step_q, n_steps, **kw),
    )


def _unexpected_core_discovery():
    """Fail if rejected public input reaches native-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid CDM input")


@pytest.mark.parametrize("factory", _INTEGER_HOSTILES)
def test_max_iter_rejects_untrusted_integer_controls_without_callbacks(
    monkeypatch, factory
):
    """All CDM adapters reject hostile iteration controls before native discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    for call in _entry_points():
        factory.reset()
        with pytest.raises(ValueError, match="max_iter must be an integer between"):
            call(max_iter=factory(5))
        assert factory.calls == 0


@pytest.mark.parametrize("factory", _FLOAT_HOSTILES)
def test_tol_rejects_untrusted_numeric_controls_without_callbacks(monkeypatch, factory):
    """All CDM adapters reject hostile tolerances before native discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    for call in _entry_points():
        factory.reset()
        with pytest.raises(ValueError, match="tol must be a finite number > 0"):
            call(tol=factory(1e-6))
        assert factory.calls == 0


@pytest.mark.parametrize(
    ("call", "value"),
    (
        (lambda y, q, value: fit_cdm(y, q, model=value), _HostileStr("dina")),
        (lambda y, q, value: fit_ho_cdm(y, q, model=value), _HostileStr("dino")),
    ),
)
def test_model_rejects_string_subclasses_without_callbacks(monkeypatch, call, value):
    """DINA/DINO selectors accept exact built-in strings only."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    value.reset()
    responses, q_matrix = _three_attribute_inputs()

    with pytest.raises(ValueError, match="model must be 'dina' or 'dino'"):
        call(responses, q_matrix, value)

    assert value.calls == 0


@pytest.mark.parametrize("model", ["other", "DINA", "dina ", ""])
def test_model_rejects_unknown_exact_strings_before_native_discovery(monkeypatch, model):
    """Python preserves the Rust selector vocabulary instead of coercing strings."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses, q_matrix = _binary_inputs()

    with pytest.raises(ValueError, match="model must be 'dina' or 'dino'"):
        fit_cdm(responses, q_matrix, model=model)


@pytest.mark.parametrize("bad", [True, np.bool_(True)])
def test_boolean_stopping_controls_fail_before_native_discovery(monkeypatch, bad):
    """Boolean integer/numeric aliases remain outside the trusted control domain."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses, q_matrix = _binary_inputs()

    with pytest.raises(ValueError, match="max_iter must be an integer between"):
        fit_gdina(responses, q_matrix, max_iter=bad)
    with pytest.raises(ValueError, match="tol must be a finite number > 0"):
        fit_gdina(responses, q_matrix, tol=bad)


@pytest.mark.parametrize("bad", [0, -1, 10**12])
def test_iteration_bounds_fail_before_native_discovery(monkeypatch, bad):
    """Iteration work remains positively bounded by the repository cap."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses, q_matrix = _binary_inputs()

    with pytest.raises(ValueError, match="max_iter must be an integer between"):
        fit_gdina(responses, q_matrix, max_iter=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf, 10**1000])
def test_tolerance_domain_fails_before_native_discovery(monkeypatch, bad):
    """Tolerance must be finite and strictly positive before native discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses, q_matrix = _binary_inputs()

    with pytest.raises(ValueError, match="tol must be a finite number > 0"):
        fit_gdina(responses, q_matrix, tol=bad)


def test_malformed_response_shape_fails_before_native_discovery(monkeypatch):
    """Existing structural validation precedes native-loader access after hardening."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    q_matrix = np.ones((2, 1), dtype=np.int64)

    with pytest.raises(ValueError, match="responses must be a 2-D persons x items array"):
        fit_gdina(np.zeros(2), q_matrix)


@pytest.mark.parametrize("numpy_type", _NUMPY_INTEGER_TYPES)
def test_exact_numpy_iteration_controls_reach_native_boundary(monkeypatch, numpy_type):
    """Supported exact NumPy integer scalars retain compatibility."""
    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses, q_matrix = _binary_inputs()

    with pytest.raises(RuntimeError, match="fit_gdina requires the compiled Rust core"):
        fit_gdina(responses, q_matrix, max_iter=numpy_type(5))

    assert calls == 1


@pytest.mark.parametrize(
    ("numpy_type", "value"),
    tuple((numpy_type, 1) for numpy_type in _NUMPY_INTEGER_TYPES)
    + tuple((numpy_type, 1e-4) for numpy_type in _NUMPY_FLOAT_TYPES),
)
def test_exact_numpy_tolerance_controls_reach_native_boundary(
    monkeypatch, numpy_type, value
):
    """Supported exact NumPy real scalars retain tolerance compatibility."""
    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses, q_matrix = _binary_inputs()

    with pytest.raises(RuntimeError, match="fit_gdina requires the compiled Rust core"):
        fit_gdina(responses, q_matrix, tol=numpy_type(value))

    assert calls == 1


@pytest.mark.parametrize("model", ["dina", "dino"])
def test_exact_model_strings_reach_native_boundary(monkeypatch, model):
    """Both existing DINA/DINO selector values remain accepted."""
    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses, q_matrix = _three_attribute_inputs()

    with pytest.raises(RuntimeError, match="fit_ho_cdm requires the compiled Rust core"):
        fit_ho_cdm(responses, q_matrix, model=model)

    assert calls == 1
