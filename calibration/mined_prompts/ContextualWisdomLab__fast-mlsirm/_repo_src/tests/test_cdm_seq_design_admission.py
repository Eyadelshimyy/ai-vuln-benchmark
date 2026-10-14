"""Sequential G-DINA design-admission regressions."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import fast_mlsirm.cdm as cdm
import fast_mlsirm.fitstats as fitstats


class _ArrayProvider:
    """Fail if NumPy is allowed to invoke caller-owned array conversion."""

    callbacks = 0

    def __array__(self, dtype=None):  # pragma: no cover - must never execute
        type(self).callbacks += 1
        raise AssertionError("caller array protocol executed during design admission")


class _ListSubclass(list):
    """Caller-defined sequence identity that must not be traversed implicitly."""


class _IntSubclass(int):
    """Caller-defined numeric identity that must not be normalized implicitly."""


class _CoreReached(RuntimeError):
    """Signal that validated evidence reached the Rust dispatch boundary."""


class _CaptureCore:
    def __init__(self, captured: dict[str, object]) -> None:
        self._captured = captured

    def fit_seq_gdina_qr(self, *args):
        self._captured["step_q"] = args[2]
        self._captured["n_steps"] = args[3]
        raise _CoreReached


def _responses() -> np.ndarray:
    return np.array([[0.0], [1.0]], dtype=np.float64)


def _step_q() -> np.ndarray:
    return np.array([[1]], dtype=np.int64)


def test_seq_gdina_qr_rejects_top_level_n_steps_provider_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("compiled core discovered")),
    )
    _ArrayProvider.callbacks = 0

    with pytest.raises(ValueError, match="n_steps must be a trusted 1-D integer"):
        cdm.fit_seq_gdina_qr(_responses(), _step_q(), _ArrayProvider())

    assert _ArrayProvider.callbacks == 0


def test_seq_gdina_qr_rejects_nested_n_steps_provider_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("compiled core discovered")),
    )
    _ArrayProvider.callbacks = 0

    with pytest.raises(ValueError, match="n_steps must be a trusted 1-D integer"):
        cdm.fit_seq_gdina_qr(_responses(), _step_q(), [_ArrayProvider()])

    assert _ArrayProvider.callbacks == 0


@pytest.mark.parametrize(
    "n_steps",
    [
        _ListSubclass([1]),
        [_IntSubclass(1)],
        np.array([1], dtype=np.int64).view(type("_ArraySubclass", (np.ndarray,), {})),
    ],
)
def test_seq_gdina_qr_rejects_caller_defined_n_steps_identities_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    n_steps: object,
) -> None:
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("compiled core discovered")),
    )

    with pytest.raises(ValueError, match="n_steps must be a trusted 1-D integer"):
        cdm.fit_seq_gdina_qr(_responses(), _step_q(), n_steps)


def test_seq_gdina_qr_enforces_native_step_cap_before_step_q_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("compiled core discovered")),
    )
    _ArrayProvider.callbacks = 0

    with pytest.raises(ValueError, match="n_steps entries must be <= 50"):
        cdm.fit_seq_gdina_qr(_responses(), _ArrayProvider(), [51])

    assert _ArrayProvider.callbacks == 0


def test_seq_gdina_qr_rejects_step_q_provider_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("compiled core discovered")),
    )
    _ArrayProvider.callbacks = 0

    with pytest.raises(ValueError, match="step_q must be a trusted NumPy array"):
        cdm.fit_seq_gdina_qr(_responses(), _ArrayProvider(), [1])

    assert _ArrayProvider.callbacks == 0


def test_seq_gdina_qr_rejects_nested_step_q_provider_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("compiled core discovered")),
    )
    _ArrayProvider.callbacks = 0

    with pytest.raises(ValueError, match="step_q must be a trusted NumPy array"):
        cdm.fit_seq_gdina_qr(_responses(), [[_ArrayProvider()]], [1])

    assert _ArrayProvider.callbacks == 0


@pytest.mark.parametrize(
    "n_steps",
    [
        np.array([1], dtype=np.int32),
        np.array([1], dtype=np.uint16),
        [np.int16(1)],
        (np.uint8(1),),
    ],
)
def test_seq_gdina_qr_preserves_trusted_integer_step_controls(
    monkeypatch: pytest.MonkeyPatch,
    n_steps: object,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(fitstats, "_core_module", lambda: _CaptureCore(captured))

    with pytest.raises(_CoreReached):
        cdm.fit_seq_gdina_qr(_responses(), _step_q(), n_steps)

    assert captured["n_steps"] == [1]
    assert all(type(value) is int for value in captured["n_steps"])
    np.testing.assert_array_equal(captured["step_q"], np.array([1], dtype=np.int64))


@pytest.mark.parametrize(
    "n_steps",
    [
        np.array([True], dtype=np.bool_),
        np.array([1.0], dtype=np.float64),
        np.array(["1"]),
        [True],
        [1.0],
        ["1"],
        [0],
    ],
)
def test_seq_gdina_qr_rejects_invalid_step_controls_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    n_steps: object,
) -> None:
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: (_ for _ in ()).throw(AssertionError("compiled core discovered")),
    )

    with pytest.raises(ValueError, match="n_steps entries must be positive integers"):
        cdm.fit_seq_gdina_qr(_responses(), _step_q(), n_steps)


def test_seq_gdina_qr_design_admission_survives_direct_module_reload() -> None:
    """Reloading cdm cannot restore callback-bearing design materialization."""

    script = r'''
import importlib
import numpy as np
import fast_mlsirm.cdm as cdm
import fast_mlsirm.fitstats as fitstats


class ArraySentinel:
    calls = 0

    def __array__(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("caller sequential-design array protocol executed")


fitstats._core_module = lambda: None
reloaded = importlib.reload(cdm)
responses = np.array([[0.0], [1.0]], dtype=np.float64)
step_q = np.array([[1]], dtype=np.int64)

try:
    reloaded.fit_seq_gdina_qr(responses, step_q, ArraySentinel())
except ValueError as exc:
    assert "n_steps must be a trusted 1-D integer" in str(exc)
else:
    raise AssertionError("callback-bearing n_steps provider was accepted after reload")
assert ArraySentinel.calls == 0

try:
    reloaded.fit_seq_gdina_qr(responses, ArraySentinel(), np.array([1], dtype=np.uint16))
except ValueError as exc:
    assert "step_q must be a trusted NumPy array" in str(exc)
else:
    raise AssertionError("callback-bearing step_q provider was accepted after reload")
assert ArraySentinel.calls == 0

try:
    reloaded.fit_seq_gdina_qr(responses, step_q, (np.uint8(1),))
except RuntimeError as exc:
    assert "requires the compiled Rust core" in str(exc)
else:
    raise AssertionError("trusted sequential design did not reach native dispatch")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
