"""Lossless-Rust-f64 regressions for Brennan-Kane mastery cuts."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory
from fast_mlsirm.rubric import gtheory_pilot


def _valid_scores() -> np.ndarray:
    """Return a small complete balanced persons-by-items score matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def test_phi_lambda_rejects_lossy_integer_cut_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mastery cut cannot change integer identity at the Rust f64 boundary."""
    core_calls: list[str] = []

    def forbidden_core(_name: str):
        core_calls.append("core")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(gtheory, "_core_or_raise", forbidden_core)

    with pytest.raises(ValueError, match="cut must be exactly representable as float64"):
        gtheory.phi_lambda(_valid_scores(), 2**53 + 1, n_i_prime=(2,))

    assert core_calls == []


def test_phi_lambda_rejects_lossy_longdouble_cut_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extended-precision mastery cuts cannot be silently rounded to Rust f64."""
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble has no precision beyond float64")

    core_calls: list[str] = []

    def forbidden_core(_name: str):
        core_calls.append("core")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(gtheory, "_core_or_raise", forbidden_core)
    cut = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    assert np.longdouble(float(cut)) != cut

    with pytest.raises(ValueError, match="cut must be exactly representable as float64"):
        gtheory.phi_lambda(_valid_scores(), cut, n_i_prime=(2,))

    assert core_calls == []


@pytest.mark.parametrize("cut", [2**53 + 1, np.uint64(2**53 + 1)])
def test_rubric_handoff_rejects_lossy_integer_cut(cut: object) -> None:
    """The provenance handoff cannot advertise a cut changed by Rust f64 normalization."""
    with pytest.raises(ValueError, match="cut must be exactly representable as float64"):
        gtheory_pilot._finite_cut(cut)


def test_lossless_numpy_cut_reaches_rust_as_builtin_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly representable NumPy controls preserve the established public API."""
    captured: dict[str, object] = {}

    class _Core:
        def phi_lambda(self, data, n_p, n_i, cut, primes):
            captured["data"] = np.array(data, copy=True)
            captured["shape"] = (n_p, n_i)
            captured["cut"] = cut
            captured["primes"] = primes
            return {
                "grand_mean": 0.5,
                "var": [0.1, 0.1, 0.1],
                "var_xbar": 0.05,
                "signal": 0.2,
                "phi": [0.8],
            }

    monkeypatch.setattr(gtheory, "_core_or_raise", lambda _name: _Core())

    result = gtheory.phi_lambda(
        _valid_scores(),
        np.longdouble(0.5),
        n_i_prime=(np.int16(2),),
    )

    assert result.phi == [0.8]
    assert type(captured["cut"]) is float
    assert captured["cut"] == 0.5
    assert captured["shape"] == (2, 2)
    assert captured["primes"] == [2]
    assert gtheory_pilot._finite_cut(np.longdouble(0.5)) == 0.5
