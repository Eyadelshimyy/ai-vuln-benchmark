"""Trust-boundary regressions for cognitive-diagnosis response admission."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.cdm as cdm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import fit_cdm, fit_gdina


class _ArraySentinel:
    """Caller array provider that must remain untouched for invalid controls."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __array__(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("caller array materialization executed")


class _HostileFloat:
    """Object-array element whose real conversion must never execute."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller numeric conversion executed")


class _HostileInt(int):
    """Integer subclass that must not cross the package trust boundary."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller integer conversion executed")

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller float conversion executed")


def _unexpected_core_discovery():
    """Fail if invalid public evidence reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid CDM input")


def _q_matrix() -> np.ndarray:
    """Return a minimal valid item-by-attribute design."""

    return np.array([[1], [1]], dtype=np.int64)


def _higher_order_q_matrix() -> np.ndarray:
    """Return a minimal three-attribute design for higher-order entry points."""

    return np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int64)


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_complex_responses_before_lossy_coercion(monkeypatch, fit):
    """Imaginary observed evidence cannot be projected onto binary real data."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit(responses, _q_matrix())


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_object_storage_before_element_coercion(monkeypatch, fit):
    """Object response storage fails before caller numeric conversion callbacks."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileFloat.reset()
    responses = np.array([[_HostileFloat(), 1], [1, 0]], dtype=object)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        fit(responses, _q_matrix())

    assert _HostileFloat.calls == 0


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_callback_bearing_response_providers(monkeypatch, fit):
    """Top-level and nested array protocols cannot synthesize observed evidence."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    for responses in (_ArraySentinel(), [[_ArraySentinel(), 1], [1, 0]]):
        _ArraySentinel.reset()
        with pytest.raises(ValueError, match="trusted NumPy array or built-in sequence"):
            fit(responses, _q_matrix())
        assert _ArraySentinel.calls == 0


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_numeric_subclasses_without_callbacks(monkeypatch, fit):
    """Caller-defined numeric identities are not trusted as inert response cells."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileInt.reset()
    responses = [[_HostileInt(0), 1], [1, 0]]

    with pytest.raises(ValueError, match="trusted NumPy array or built-in sequence"):
        fit(responses, _q_matrix())

    assert _HostileInt.calls == 0


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_preserve_trusted_sequence_compatibility(monkeypatch, fit):
    """Inert ndarray rows, shared rows, and NumPy booleans remain supported."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    shared_row = [0, 1]
    cases = (
        [np.array([0, 1], dtype=np.int8), np.array([1, 0], dtype=np.uint8)],
        [shared_row, shared_row],
        [[np.bool_(False), np.int8(1)], [np.uint8(1), np.bool_(False)]],
    )

    for responses in cases:
        with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
            fit(responses, _q_matrix())

    assert calls == len(cases)


def test_cdm_module_reload_preserves_callback_safe_response_guard():
    """Reload validation runs in isolation so class identities remain stable here."""

    script = r'''
import importlib
import numpy as np
import fast_mlsirm.cdm as cdm
import fast_mlsirm.fitstats as fitstats


class HostileList(list):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("caller container iteration executed")


def missing_core():
    return None


fitstats._core_module = missing_core
reloaded = importlib.reload(cdm)
q_matrix = np.array([[1], [1]], dtype=np.int64)

hostile = HostileList([[0, 1], [1, 0]])
try:
    reloaded.fit_cdm(hostile, q_matrix)
except ValueError as exc:
    assert "trusted NumPy array or built-in sequence" in str(exc)
else:
    raise AssertionError("hostile list subclass was accepted after reload")
assert HostileList.calls == 0

shared_row = [0, 1]
try:
    reloaded.fit_cdm([shared_row, shared_row], q_matrix)
except RuntimeError as exc:
    assert "requires the compiled Rust core" in str(exc)
else:
    raise AssertionError("shared acyclic rows did not reach native dispatch boundary")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_extended_precision_that_collapses_to_binary(monkeypatch, fit):
    """A non-binary long-double value cannot round into accepted float64 evidence."""

    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        pytest.skip("platform longdouble does not exceed float64 precision")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    almost_one = np.nextafter(np.longdouble(1), np.longdouble(0))
    responses = np.array([[almost_one, 1], [1, 0]], dtype=np.longdouble)

    with pytest.raises(ValueError, match="exactly representable as float64"):
        fit(responses, _q_matrix())


@pytest.mark.parametrize(
    ("fit", "kwargs", "message"),
    [
        (fit_cdm, {"model": "other"}, "model must be 'dina' or 'dino'"),
        (fit_cdm, {"max_iter": 0}, "max_iter must be an integer between"),
        (fit_cdm, {"tol": 0.0}, "tol must be a finite number > 0"),
        (fit_gdina, {"max_iter": 0}, "max_iter must be an integer between"),
        (fit_gdina, {"tol": 0.0}, "tol must be a finite number > 0"),
    ],
)
def test_invalid_controls_fail_before_response_materialization(
    monkeypatch, fit: Callable[..., object], kwargs: dict[str, object], message: str
):
    """Semantic controls are rejected before caller response array protocols."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _ArraySentinel.reset()

    with pytest.raises(ValueError, match=message):
        fit(_ArraySentinel(), _q_matrix(), **kwargs)

    assert _ArraySentinel.calls == 0


@pytest.mark.parametrize(
    ("entrypoint", "args", "kwargs", "message"),
    [
        (
            cdm.validate_q_matrix,
            (_q_matrix(),),
            {"max_iter": 0},
            "max_iter must be an integer between",
        ),
        (
            cdm.validate_q_matrix,
            (_q_matrix(),),
            {"tol": 0.0},
            "tol must be a finite number > 0",
        ),
        (
            cdm.gdina_wald_selection,
            (_q_matrix(),),
            {"max_iter": 0},
            "max_iter must be an integer between",
        ),
        (
            cdm.gdina_wald_selection,
            (_q_matrix(),),
            {"tol": 0.0},
            "tol must be a finite number > 0",
        ),
        (
            cdm.fit_ho_cdm,
            (_higher_order_q_matrix(),),
            {"model": "other"},
            "model must be 'dina' or 'dino'",
        ),
        (
            cdm.fit_ho_cdm,
            (_higher_order_q_matrix(),),
            {"max_iter": 0},
            "max_iter must be an integer between",
        ),
        (
            cdm.fit_ho_cdm,
            (_higher_order_q_matrix(),),
            {"tol": 0.0},
            "tol must be a finite number > 0",
        ),
        (
            cdm.fit_ho_gdina,
            (_higher_order_q_matrix(),),
            {"max_iter": 0},
            "max_iter must be an integer between",
        ),
        (
            cdm.fit_ho_gdina,
            (_higher_order_q_matrix(),),
            {"tol": 0.0},
            "tol must be a finite number > 0",
        ),
        (
            cdm.fit_seq_gdina,
            (_q_matrix(),),
            {"max_iter": 0},
            "max_iter must be an integer between",
        ),
        (
            cdm.fit_seq_gdina,
            (_q_matrix(),),
            {"tol": 0.0},
            "tol must be a finite number > 0",
        ),
        (
            cdm.fit_seq_gdina_qr,
            (np.array([[1], [1]], dtype=np.int64), np.array([1, 1], dtype=np.int64)),
            {"max_iter": 0},
            "max_iter must be an integer between",
        ),
        (
            cdm.fit_seq_gdina_qr,
            (np.array([[1], [1]], dtype=np.int64), np.array([1, 1], dtype=np.int64)),
            {"tol": 0.0},
            "tol must be a finite number > 0",
        ),
    ],
)
def test_sibling_semantic_controls_fail_before_response_materialization(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Sibling CDM controls must fail before caller response protocols execute."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _ArraySentinel.reset()

    with pytest.raises(ValueError, match=message):
        entrypoint(_ArraySentinel(), *args, **kwargs)

    assert _ArraySentinel.calls == 0


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_valid_real_evidence_reaches_native_dispatch_boundary(monkeypatch, fit):
    """Ordinary real binary responses retain the existing Rust dispatch contract."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.array([[0.0, 1.0], [1.0, np.nan]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
        fit(responses, _q_matrix())

    assert calls == 1


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_callback_bearing_q_matrix_providers(monkeypatch, fit):
    """Q-matrix array protocols cannot synthesize item-to-attribute evidence."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)
    for q_matrix in (_ArraySentinel(), [[_ArraySentinel()], [1]]):
        _ArraySentinel.reset()
        with pytest.raises(ValueError, match="q_matrix must be a trusted NumPy array or built-in sequence"):
            fit(responses, q_matrix)
        assert _ArraySentinel.calls == 0


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_preserve_trusted_q_matrix_sequence_compatibility(monkeypatch, fit):
    """Inert NumPy rows and concrete NumPy scalar Q evidence remain supported."""

    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(fitstats, "_core_module", missing_core)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)
    q_matrices = (
        [np.array([1], dtype=np.int8), np.array([1], dtype=np.uint8)],
        [[np.bool_(True)], [np.int16(1)]],
    )

    for q_matrix in q_matrices:
        with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
            fit(responses, q_matrix)

    assert calls == len(q_matrices)
