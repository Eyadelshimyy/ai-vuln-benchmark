"""Compatibility and callback-safety contracts for NumPy D-study controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


class _HostileIntegerArray(np.ndarray):
    """Array subclass whose iteration must never run during control admission."""

    def __new__(cls, values):
        obj = np.asarray(values, dtype=np.int64).view(cls)
        obj.iter_calls = 0
        return obj

    def __array_finalize__(self, obj) -> None:
        self.iter_calls = getattr(obj, "iter_calls", 0)

    def __iter__(self):
        self.iter_calls += 1
        raise RuntimeError("GTHEORY_NUMPY_CONTROL_ITER_SENTINEL")


class _RecordingCore:
    """Minimal Rust-boundary stub recording normalized D-study controls."""

    def __init__(self) -> None:
        self.primes: list[int] | None = None
        self.pairs: list[tuple[int, int]] | None = None

    def gtheory_pi(self, data, n_p, n_i, primes):
        self.primes = primes
        return {
            "df": [1.0, 1.0, 1.0],
            "ss": [1.0, 1.0, 1.0],
            "ms": [1.0, 1.0, 1.0],
            "var_raw": [1.0, 1.0, 1.0],
            "var": [1.0, 1.0, 1.0],
            "d_study": [],
        }

    def gtheory_pio(self, data, n_p, n_i, n_o, pairs):
        self.pairs = pairs
        return {
            "df": [1.0] * 7,
            "ss": [1.0] * 7,
            "ms": [1.0] * 7,
            "var_raw": [1.0] * 7,
            "var": [1.0] * 7,
            "d_study": [],
        }

    def phi_lambda(self, data, n_p, n_i, cut, primes):
        self.primes = primes
        return {
            "grand_mean": 0.5,
            "var": [1.0, 1.0, 1.0],
            "var_xbar": 1.0,
            "signal": 0.5,
            "phi": [0.5 for _ in primes],
        }


def _pi_data() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _pio_data() -> np.ndarray:
    return np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )


def test_exact_numpy_integer_dstudy_controls_remain_supported(monkeypatch) -> None:
    core = _RecordingCore()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: core)

    gtheory.gtheory_pi(_pi_data(), n_i_prime=np.array([2, 3], dtype=np.int64))
    assert core.primes == [2, 3]
    assert all(type(value) is int for value in core.primes)

    gtheory.gtheory_pio(
        _pio_data(),
        n_prime=np.array([[2, 2], [3, 4]], dtype=np.uint16),
    )
    assert core.pairs == [(2, 2), (3, 4)]
    assert all(type(value) is int for pair in core.pairs for value in pair)

    gtheory.phi_lambda(
        _pi_data(),
        0.5,
        n_i_prime=np.array([4, 5], dtype=np.int32),
    )
    assert core.primes == [4, 5]


def test_exact_range_one_facet_dstudy_controls_remain_supported(monkeypatch) -> None:
    """The documented Sequence[int] surface retains inert built-in ranges."""
    core = _RecordingCore()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: core)

    gtheory.gtheory_pi(_pi_data(), n_i_prime=range(2, 5))
    assert core.primes == [2, 3, 4]
    assert all(type(value) is int for value in core.primes)

    gtheory.phi_lambda(_pi_data(), 0.5, n_i_prime=range(5, 7))
    assert core.primes == [5, 6]


def test_numpy_dstudy_control_subclasses_fail_before_iteration(monkeypatch) -> None:
    hostile_vector = _HostileIntegerArray([2, 3])
    hostile_pairs = _HostileIntegerArray([[2, 2], [3, 4]])
    monkeypatch.setattr(
        gtheory,
        "_core_or_raise",
        lambda name: pytest.fail("invalid D-study controls reached Rust"),
    )

    with pytest.raises(ValueError):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=hostile_vector)
    with pytest.raises(ValueError):
        gtheory.gtheory_pio(_pio_data(), n_prime=hostile_pairs)

    assert hostile_vector.iter_calls == 0
    assert hostile_pairs.iter_calls == 0


@pytest.mark.parametrize(
    "control",
    [
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([True, False], dtype=np.bool_),
        np.array(["2", "3"]),
        np.array([[2, 3]], dtype=np.int64),
    ],
)
def test_one_facet_numpy_dstudy_controls_reject_non_integer_or_wrong_rank(
    monkeypatch, control: np.ndarray
) -> None:
    monkeypatch.setattr(
        gtheory,
        "_core_or_raise",
        lambda name: pytest.fail("invalid D-study controls reached Rust"),
    )

    with pytest.raises(ValueError):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=control)


@pytest.mark.parametrize(
    "control",
    [
        np.array([2, 3], dtype=np.int64),
        np.array([[2, 3, 4]], dtype=np.int64),
        np.array([[2.0, 3.0]], dtype=np.float64),
        np.array([[True, False]], dtype=np.bool_),
    ],
)
def test_two_facet_numpy_dstudy_controls_reject_wrong_shape_or_dtype(
    monkeypatch, control: np.ndarray
) -> None:
    monkeypatch.setattr(
        gtheory,
        "_core_or_raise",
        lambda name: pytest.fail("invalid D-study controls reached Rust"),
    )

    with pytest.raises(ValueError):
        gtheory.gtheory_pio(_pio_data(), n_prime=control)
