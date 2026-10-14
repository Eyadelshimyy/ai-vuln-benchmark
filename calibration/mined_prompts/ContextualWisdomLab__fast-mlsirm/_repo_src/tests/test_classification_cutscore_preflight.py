"""Fail-first trust-boundary tests for classification cut-score controls."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

import fast_mlsirm.classification as classification
from fast_mlsirm import _classification_cutscore_safety as cutscore_safety


class _HostileFloat(float):
    """A float subclass whose coercion is observable and must never run."""

    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile __float__ callback executed")


def _unexpected_core_discovery(name: str) -> object:
    """Fail if an invalid cut-score reaches native capability discovery."""
    raise AssertionError(f"invalid cut-score reached Rust discovery: {name}")


def test_classification_uses_canonical_cutscore_normalizer() -> None:
    """The classification module must not retain a divergent fallback validator."""
    assert classification._normalize_cutscores is cutscore_safety.normalize_cutscores


def test_rudner_rejects_untrusted_cut_before_core_and_without_callback(
    monkeypatch,
) -> None:
    """Rudner cut controls reject scalar subclasses before native lookup."""
    monkeypatch.setattr(classification, "_core_or_raise", _unexpected_core_discovery)
    _HostileFloat.calls = 0

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification.rudner_classification(
            np.array([0.0]),
            np.array([1.0]),
            [_HostileFloat(0.0)],
        )

    assert _HostileFloat.calls == 0


def test_lee_rejects_untrusted_cut_before_core_and_without_callback(monkeypatch) -> None:
    """Lee cut controls reject scalar subclasses before native lookup."""
    monkeypatch.setattr(classification, "_core_or_raise", _unexpected_core_discovery)
    _HostileFloat.calls = 0

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification.lee_classification(
            np.array([[0.25, 0.75]], dtype=np.float64),
            [_HostileFloat(1.0)],
        )

    assert _HostileFloat.calls == 0


def test_rudner_rejects_non_iterable_cutscores_with_stable_value_error(
    monkeypatch,
) -> None:
    """Malformed cut-score containers fail at the public validation boundary."""
    monkeypatch.setattr(classification, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification.rudner_classification(
            np.array([0.0]),
            np.array([1.0]),
            None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "cut",
    [True, np.bool_(False), np.inf, np.float64(np.nan), 10**400],
)
def test_rudner_rejects_invalid_cut_identity_or_finiteness_before_core(
    monkeypatch,
    cut: object,
) -> None:
    """Boolean and non-finite cuts fail closed before native lookup."""
    monkeypatch.setattr(classification, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification.rudner_classification(
            np.array([0.0]),
            np.array([1.0]),
            [cut],
        )


def test_extended_precision_cut_outside_float_range_is_stable_value_error() -> None:
    """Trusted NumPy reals outside float64 range retain the benign error contract."""
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble does not extend float64 range")

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification._normalize_cutscores([np.longdouble(np.finfo(np.longdouble).max)])


def test_numpy_cut_conversion_overflow_is_stable_value_error(monkeypatch) -> None:
    """A platform conversion overflow is normalized at the package boundary."""
    original_float = builtins.float

    def overflow_numpy_float(value: object) -> float:
        if type(value) is np.float64:
            raise OverflowError("platform conversion overflow")
        return original_float(value)

    monkeypatch.setitem(
        classification._normalize_cutscores.__globals__,
        "float",
        overflow_numpy_float,
    )

    with pytest.raises(ValueError, match=r"cutscores entries must be finite real scalars"):
        classification._normalize_cutscores([np.float64(1.5)])


def test_trusted_builtin_and_numpy_cut_scores_normalize_to_builtin_floats() -> None:
    """Established built-in and concrete NumPy real scalar cuts remain accepted."""
    normalized = classification._normalize_cutscores(
        [0, 1.5, np.int64(2), np.uint8(3), np.float32(4.5), np.float64(5.5)]
    )

    assert normalized == [0.0, 1.5, 2.0, 3.0, 4.5, 5.5]
    assert all(type(value) is float for value in normalized)


def test_empty_cut_sequence_remains_rust_owned_domain_validation(monkeypatch) -> None:
    """An empty trusted sequence reaches Rust so domain validation stays native-owned."""
    calls: list[tuple[str, list[float]]] = []

    class StubCore:
        def rudner_classification(self, _theta, _sem, _weights, cuts):
            calls.append(("rudner_classification", cuts))
            raise RuntimeError("native domain validation sentinel")

    monkeypatch.setattr(classification, "_core_or_raise", lambda _name: StubCore())

    with pytest.raises(RuntimeError, match="native domain validation sentinel"):
        classification.rudner_classification(
            np.array([0.0]),
            np.array([1.0]),
            [],
        )

    assert calls == [("rudner_classification", [])]
