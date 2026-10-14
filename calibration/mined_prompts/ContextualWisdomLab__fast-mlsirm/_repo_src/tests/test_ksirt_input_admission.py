"""Trust-boundary regressions for the public KSIRT analysis wrapper."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import ksirt


class _HostileKernel(str):
    """String subclass whose semantic callbacks must never run."""

    def __eq__(self, other: object) -> bool:
        raise AssertionError("kernel equality callback executed")

    def __hash__(self) -> int:
        raise AssertionError("kernel hash callback executed")


class _HostileInteger(int):
    """Integer subclass whose conversion callbacks must never run."""

    def __int__(self) -> int:
        raise AssertionError("nevalpoints int callback executed")

    def __index__(self) -> int:
        raise AssertionError("nevalpoints index callback executed")


class _HostileNumber:
    """Object-dtype numeric provider whose conversion must never execute."""

    def __float__(self) -> float:
        raise AssertionError("object numeric conversion callback executed")


class _HostileArrayProvider:
    """Arbitrary NumPy protocol provider that package admission must reject."""

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("caller __array__ callback executed")


class _ResponseSentinel:
    """Response provider that proves invalid controls fail before data work."""

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("responses were materialized")


def _forbid_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if compiled-core discovery occurs before Python admission."""

    def _unexpected_core() -> object:
        raise AssertionError("compiled core was discovered")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)


def _core_fixture(captured: dict[str, object]) -> object:
    """Return a minimal Rust-boundary stand-in for accepted-input tests."""

    class _Core:
        def ksirt_occ(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            kernel: str,
            nevalpoints: int,
            bandwidth: list[float] | None,
        ) -> dict[str, object]:
            captured.update(
                responses=responses,
                n_persons=n_persons,
                n_items=n_items,
                kernel=kernel,
                nevalpoints=nevalpoints,
                bandwidth=bandwidth,
            )
            return {
                "theta": [-0.5, 0.5],
                "grid": [-1.0, 0.0, 1.0],
                "bandwidth": [0.5],
                "options": [[0.0, 1.0]],
                "occ": [[0.8, 0.5, 0.2, 0.2, 0.5, 0.8]],
                "expected": [[0.2, 0.5, 0.8]],
                "expected_total": [0.2, 0.5, 0.8],
            }

    return _Core()


def test_ksirt_rejects_hostile_kernel_before_data_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel semantics are admitted without caller callbacks or data work."""
    _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="kernel must be gaussian, quadratic, or uniform"):
        ksirt.analyze_ksirt(_ResponseSentinel(), kernel=_HostileKernel("gaussian"), nevalpoints=51)


def test_ksirt_rejects_hostile_nevalpoints_before_data_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation-grid size is admitted without implicit integer coercion."""
    _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="nevalpoints must be an integer"):
        ksirt.analyze_ksirt(_ResponseSentinel(), nevalpoints=_HostileInteger(51))


def test_ksirt_rejects_array_provider_responses_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response evidence cannot be synthesized through caller array protocols."""
    _forbid_core(monkeypatch)

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        ksirt.analyze_ksirt(_HostileArrayProvider(), nevalpoints=51)


def test_ksirt_rejects_array_provider_bandwidth_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bandwidth evidence cannot be synthesized through caller array protocols."""
    _forbid_core(monkeypatch)
    responses = np.array([[0.0], [1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="bandwidth must be a numeric array"):
        ksirt.analyze_ksirt(responses, nevalpoints=51, bandwidth=_HostileArrayProvider())


def test_ksirt_rejects_complex_responses_before_real_narrowing_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imaginary response evidence cannot be discarded by float64 coercion."""
    _forbid_core(monkeypatch)
    responses = np.array([[0.0 + 1.0j], [1.0 + 0.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        ksirt.analyze_ksirt(responses, nevalpoints=51)


def test_ksirt_rejects_complex_bandwidth_before_real_narrowing_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imaginary bandwidth evidence cannot be discarded by float64 coercion."""
    _forbid_core(monkeypatch)
    responses = np.array([[0.0], [1.0]], dtype=np.float64)
    bandwidth = np.array([0.5 + 1.0j], dtype=np.complex128)

    with pytest.raises(ValueError, match="bandwidth must be real-valued"):
        ksirt.analyze_ksirt(responses, nevalpoints=51, bandwidth=bandwidth)


def test_ksirt_rejects_object_responses_without_numeric_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object arrays cannot execute element conversion during admission."""
    _forbid_core(monkeypatch)
    responses = np.array(
        [[_HostileNumber()], [_HostileNumber()]],
        dtype=object,
    )

    with pytest.raises(ValueError, match="responses must be a numeric array"):
        ksirt.analyze_ksirt(responses, nevalpoints=51)


def test_ksirt_rejects_object_bandwidth_without_numeric_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object bandwidths cannot execute element conversion during admission."""
    _forbid_core(monkeypatch)
    responses = np.array([[0.0], [1.0]], dtype=np.float64)
    bandwidth = np.array([_HostileNumber()], dtype=object)

    with pytest.raises(ValueError, match="bandwidth must be a numeric array"):
        ksirt.analyze_ksirt(responses, nevalpoints=51, bandwidth=bandwidth)


def test_ksirt_preserves_concrete_numpy_integer_grid_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported NumPy integer controls normalize to inert built-in integers."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(fitstats, "_core_module", lambda: _core_fixture(captured))

    result = ksirt.analyze_ksirt(
        np.array([[0.0], [1.0]], dtype=np.float32),
        nevalpoints=np.int64(3),
        bandwidth=np.array([0.5], dtype=np.float32),
    )

    assert captured["nevalpoints"] == 3
    assert type(captured["nevalpoints"]) is int
    assert captured["kernel"] == "gaussian"
    assert result.grid.shape == (3,)


def test_ksirt_preserves_builtin_sequence_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain built-in sequence evidence remains compatible after sealing callbacks."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(fitstats, "_core_module", lambda: _core_fixture(captured))

    result = ksirt.analyze_ksirt(
        [[np.int16(0)], [np.float32(1.0)]],
        nevalpoints=3,
        bandwidth=(np.float32(0.5),),
    )

    assert captured["n_persons"] == 2
    assert captured["n_items"] == 1
    assert captured["bandwidth"] == [0.5]
    assert result.grid.shape == (3,)
