"""Fail-first safety contracts for G-theory public numeric controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


class _HostileInt:
    """Integer-like probe whose conversion hooks must not run in validation."""

    def __init__(self) -> None:
        self.int_calls = 0
        self.repr_calls = 0

    def __int__(self) -> int:
        self.int_calls += 1
        raise RuntimeError("GTHEORY_INT_COERCION_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("GTHEORY_INT_REPR_SENTINEL")


class _HostileFloat:
    """Real-like probe whose conversion hooks must not run in validation."""

    def __init__(self) -> None:
        self.float_calls = 0
        self.repr_calls = 0

    def __float__(self) -> float:
        self.float_calls += 1
        raise RuntimeError("GTHEORY_FLOAT_COERCION_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("GTHEORY_FLOAT_REPR_SENTINEL")


class _HostileControlList(list):
    """List subclass whose iterator must never run during control admission."""

    def __init__(self, values) -> None:
        super().__init__(values)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        raise RuntimeError("GTHEORY_CONTROL_ITER_SENTINEL")


class _HostileControlPair(tuple):
    """Pair subclass whose iterator must never run during D-study admission."""

    def __new__(cls, values):
        obj = super().__new__(cls, values)
        obj.iter_calls = 0
        return obj

    def __iter__(self):
        self.iter_calls += 1
        raise RuntimeError("GTHEORY_PAIR_ITER_SENTINEL")


class _NoNumericCore:
    """Core stub proving invalid controls fail before Rust numerical work."""

    def gtheory_pi(self, *args, **kwargs):
        raise AssertionError("invalid G-theory controls reached Rust")

    def gtheory_pio(self, *args, **kwargs):
        raise AssertionError("invalid G-theory controls reached Rust")

    def phi_lambda(self, *args, **kwargs):
        raise AssertionError("invalid Phi(lambda) controls reached Rust")


class _RecordingCore:
    """Minimal successful core used to inspect trusted scalar marshalling."""

    def __init__(self) -> None:
        self.primes: list[int] | None = None
        self.pairs: list[tuple[int, int]] | None = None
        self.cut: float | None = None

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
        self.cut = cut
        self.primes = primes
        return {
            "grand_mean": 0.5,
            "var": [1.0, 1.0, 1.0],
            "var_xbar": 1.0,
            "signal": 0.5,
            "phi": [0.5 for _ in primes],
        }


def _pi_data() -> np.ndarray:
    """Return the smallest non-degenerate one-facet score matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _pio_data() -> np.ndarray:
    """Return the smallest non-degenerate two-facet score tensor."""
    return np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )


def test_pi_rejects_hostile_dstudy_size_before_integer_coercion(monkeypatch) -> None:
    """One-facet D-study sizes must be validated without caller ``__int__``."""
    hostile = _HostileInt()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=[hostile])

    assert hostile.int_calls == 0
    assert hostile.repr_calls == 0


def test_pio_rejects_hostile_dstudy_size_before_integer_coercion(monkeypatch) -> None:
    """Two-facet D-study sizes must reject arbitrary integer-like objects."""
    hostile = _HostileInt()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(
        ValueError,
        match=r"n_prime entries must be pairs of positive integers",
    ):
        gtheory.gtheory_pio(_pio_data(), n_prime=[(hostile, 2)])

    assert hostile.int_calls == 0
    assert hostile.repr_calls == 0


def test_pi_rejects_hostile_dstudy_container_before_iteration(monkeypatch) -> None:
    """One-facet D-study containers must be inert before iteration."""
    hostile = _HostileControlList([2, 3])
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_i_prime must be a list or tuple"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=hostile)

    assert hostile.iter_calls == 0


def test_phi_lambda_rejects_hostile_dstudy_container_before_iteration(monkeypatch) -> None:
    """Phi(lambda) D-study containers must be inert before iteration."""
    hostile = _HostileControlList([2, 3])
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_i_prime must be a list or tuple"):
        gtheory.phi_lambda(_pi_data(), 0.5, n_i_prime=hostile)

    assert hostile.iter_calls == 0


def test_pio_rejects_hostile_outer_dstudy_container_before_iteration(monkeypatch) -> None:
    """Two-facet outer D-study containers must not execute caller iteration."""
    hostile = _HostileControlList([(2, 3)])
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_prime must be a list or tuple of pairs"):
        gtheory.gtheory_pio(_pio_data(), n_prime=hostile)

    assert hostile.iter_calls == 0


def test_pio_rejects_hostile_dstudy_pair_before_iteration(monkeypatch) -> None:
    """Two-facet D-study pairs must not execute caller iteration."""
    hostile = _HostileControlPair((2, 3))
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(
        ValueError,
        match=r"n_prime entries must be pairs of positive integers",
    ):
        gtheory.gtheory_pio(_pio_data(), n_prime=[hostile])

    assert hostile.iter_calls == 0


def test_phi_lambda_rejects_hostile_cut_before_float_coercion(monkeypatch) -> None:
    """Mastery cut validation must not execute caller-defined ``__float__``."""
    hostile = _HostileFloat()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), hostile, n_i_prime=[2])

    assert hostile.float_calls == 0
    assert hostile.repr_calls == 0


@pytest.mark.parametrize("value", [True, np.bool_(False), 0, -1, 1.5, "2"])
def test_pi_rejects_non_positive_or_non_integer_dstudy_controls(
    monkeypatch, value: object
) -> None:
    """D-study sizes accept only positive trusted integer scalar types."""
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=[value])


@pytest.mark.parametrize("cut", [True, np.bool_(False), np.nan, np.inf, "0.5"])
def test_phi_lambda_rejects_non_finite_or_non_real_cuts(monkeypatch, cut: object) -> None:
    """Mastery cut accepts only finite trusted Python/NumPy real scalars."""
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), cut, n_i_prime=[2])


def test_dstudy_controls_preserve_builtin_and_numpy_scalar_types(monkeypatch) -> None:
    """Trusted Python/NumPy integer controls remain accepted after hardening."""
    core = _RecordingCore()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: core)

    gtheory.gtheory_pi(_pi_data(), n_i_prime=[2, np.int64(3)])
    gtheory.gtheory_pio(
        _pio_data(),
        n_prime=((np.int32(2), 2), [3, np.uint8(2)]),
    )

    assert core.primes == [2, 3]
    assert core.pairs == [(2, 2), (3, 2)]


def test_phi_lambda_preserves_numpy_real_and_integer_controls(monkeypatch) -> None:
    """Trusted NumPy real and integer controls marshal to ordinary scalars."""
    core = _RecordingCore()
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: core)

    result = gtheory.phi_lambda(
        _pi_data(), np.float64(0.5), n_i_prime=[np.int32(2)]
    )

    assert core.cut == 0.5
    assert core.primes == [2]
    assert result.phi == [0.5]


@pytest.mark.parametrize(
    "call",
    [
        lambda: gtheory.gtheory_pi(_pi_data(), n_i_prime=[gtheory.MAX_GTHEORY_PRIME_SIZE + 1]),
        lambda: gtheory.gtheory_pio(
            _pio_data(), n_prime=[(gtheory.MAX_GTHEORY_PRIME_SIZE + 1, 2)]
        ),
        lambda: gtheory.phi_lambda(
            _pi_data(), 0.5, n_i_prime=[gtheory.MAX_GTHEORY_PRIME_SIZE + 1]
        ),
    ],
)
def test_public_dstudy_sizes_are_bounded_before_core(monkeypatch, call) -> None:
    """D-study sizes above the output-safety bound never reach Rust."""
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"values must be <="):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: gtheory.gtheory_pi(np.array([[0.0, np.nan], [1.0, 0.0]])),
        lambda: gtheory.gtheory_pio(
            np.array(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                    [[1.0, 0.0], [0.0, np.inf]],
                ]
            )
        ),
        lambda: gtheory.phi_lambda(
            np.array([[0.0, 1.0], [np.inf, 0.0]]), 0.5
        ),
    ],
)
def test_public_score_arrays_are_finite_before_core(monkeypatch, call) -> None:
    """Non-finite score arrays fail at the Python boundary before Rust."""
    monkeypatch.setattr(gtheory, "_core_or_raise", lambda name: _NoNumericCore())

    with pytest.raises(ValueError, match=r"data must contain only finite real values"):
        call()


@pytest.mark.skipif(
    np.finfo(np.longdouble).maxexp <= 1024,
    reason="platform longdouble is not wider than binary64",
)
def test_huge_finite_longdouble_control_raises_value_error_not_overflow() -> None:
    """A finite longdouble beyond float64 range fails as ValueError, never OverflowError.

    Some NumPy versions raise ``OverflowError`` from ``float(np.longdouble)`` when the
    value exceeds binary64 range; the public control contract must normalize that to
    the package-owned ``ValueError`` in both the estimator control path and the
    rubric-pilot mastery-cut path.
    """

    huge = np.longdouble(np.finfo(np.longdouble).max)
    assert np.isfinite(huge)

    with pytest.raises(ValueError):
        gtheory._finite_real_control(huge, "cut must be a finite number")

    from fast_mlsirm.rubric import gtheory_pilot

    with pytest.raises(ValueError):
        gtheory_pilot._finite_cut(huge)
