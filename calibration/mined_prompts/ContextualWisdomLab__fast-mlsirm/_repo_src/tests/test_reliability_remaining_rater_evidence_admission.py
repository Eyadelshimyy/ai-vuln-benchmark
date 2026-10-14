"""Regression tests for remaining reliability rater-evidence trust boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


class _HostileArrayProvider:
    """Array provider that must not execute during package admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller-controlled __array__ callback executed")


class _HostileText(str):
    """Text subclass whose comparison/string callbacks must never run."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def __eq__(self, other):
        self.calls += 1
        raise AssertionError("caller-controlled text equality callback executed")

    def __hash__(self):
        self.calls += 1
        raise AssertionError("caller-controlled text hash callback executed")

    def __str__(self):
        self.calls += 1
        raise AssertionError("caller-controlled text conversion callback executed")


def _native_discovery_must_not_run():
    raise AssertionError("compiled-core discovery ran before evidence admission")


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.kripp_alpha(value),
        lambda value: reliability.finn_coefficient(value, 5),
        lambda value: reliability.maxwell_re(value),
        lambda value: reliability.robinson_a(value),
    ],
)
def test_remaining_rater_apis_reject_array_provider_before_native(
    monkeypatch, invoke
):
    """Caller array protocols must not synthesize scientific ratings evidence."""
    hostile = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="ratings must be real numeric evidence"):
        invoke(hostile)

    assert hostile.calls == 0


@pytest.mark.parametrize(
    ("invoke", "message"),
    [
        (
            lambda ratings, control: reliability.kripp_alpha(
                ratings, method=control
            ),
            "method must be one of nominal, ordinal, interval, ratio",
        ),
        (
            lambda ratings, control: reliability.finn_coefficient(
                ratings, 5, model=control
            ),
            "model must be one of oneway, twoway",
        ),
    ],
)
def test_remaining_rater_text_controls_fail_before_data_or_native(
    monkeypatch, invoke, message
):
    """Semantic text controls must be exact built-ins before caller data work."""
    ratings = _HostileArrayProvider()
    control = _HostileText("nominal")
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match=message):
        invoke(ratings, control)

    assert ratings.calls == 0
    assert control.calls == 0


def test_finn_preserves_exact_object_array_diagnostic_before_native(monkeypatch):
    """Hardening must preserve the established exact-object input contract."""
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)
    ratings = np.array([[1, object()], [2, object()]], dtype=object)

    with pytest.raises(
        ValueError,
        match="object-dtype arrays are not supported; pass a numeric array",
    ):
        reliability.finn_coefficient(ratings, 3)


def test_remaining_rater_apis_preserve_trusted_sequence_compatibility(monkeypatch):
    """Trusted numeric sequences still reach the unchanged Rust-backed kernels."""
    seen: dict[str, object] = {}

    def _kripp(flat, nr, ns, method):
        seen["kripp"] = (flat, nr, ns, method)
        return {
            "value": 0.5,
            "subjects": ns,
            "raters": nr,
            "levels": 2,
            "nmatchval": 6.0,
        }

    def _finn(flat, ns, nr, s_levels, model):
        seen["finn"] = (flat, ns, nr, s_levels, model)
        return {
            "value": 0.4,
            "statistic": 1.5,
            "df2": 4.0,
            "p_value": 0.2,
            "subjects": ns,
            "raters": nr,
        }

    def _maxwell(flat, ns, nr):
        seen["maxwell"] = (flat, ns, nr)
        return {"value": 0.25, "subjects": ns, "raters": nr}

    def _robinson(flat, ns, nr):
        seen["robinson"] = (flat, ns, nr)
        return {"value": 0.75, "subjects": ns, "raters": nr}

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(
            kripp_alpha=_kripp,
            finn_coefficient=_finn,
            maxwell_re=_maxwell,
            robinson_a=_robinson,
        ),
    )

    kripp = reliability.kripp_alpha(
        [[np.int8(1), 2, 1], [1, np.float32(2.0), np.uint8(2)]],
        method="nominal",
    )
    finn = reliability.finn_coefficient(
        [[1, np.float32(2.0)], [2, np.int16(1)], [3, np.uint8(2)]],
        np.int16(5),
        model="oneway",
    )
    maxwell = reliability.maxwell_re(
        [[np.int8(0), 1], [1, np.uint8(0)]],
    )
    robinson = reliability.robinson_a(
        [[1, np.float32(2.0)], [2, np.int16(3)], [4, np.uint8(5)]],
    )

    assert kripp.value == pytest.approx(0.5)
    assert finn.value == pytest.approx(0.4)
    assert maxwell.value == pytest.approx(0.25)
    assert robinson.value == pytest.approx(0.75)
    for key in ("kripp", "finn", "maxwell", "robinson"):
        flat = seen[key][0]
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float64


@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: reliability.kripp_alpha(value),
        lambda value: reliability.finn_coefficient(value, 5),
        lambda value: reliability.maxwell_re(value),
        lambda value: reliability.robinson_a(value),
    ],
)
def test_remaining_rater_apis_preserve_boolean_sequence_diagnostic(
    monkeypatch, invoke
):
    """Pure Boolean sequences retain the historical Boolean-specific rejection."""
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="ratings must be numeric, not boolean"):
        invoke([[True, False], [False, True]])


def test_remaining_rater_apis_preserve_mixed_boolean_numeric_sequence_compatibility(
    monkeypatch,
):
    """Exact inert Boolean leaves retain historical NumPy numeric promotion."""
    seen: dict[str, np.ndarray] = {}

    def _kripp(flat, nr, ns, method):
        seen["kripp"] = flat
        return {"value": 0.5, "subjects": ns, "raters": nr, "levels": 2, "nmatchval": 6.0}

    def _finn(flat, ns, nr, s_levels, model):
        seen["finn"] = flat
        return {"value": 0.4, "statistic": 1.5, "df2": 4.0, "p_value": 0.2, "subjects": ns, "raters": nr}

    def _maxwell(flat, ns, nr):
        seen["maxwell"] = flat
        return {"value": 0.25, "subjects": ns, "raters": nr}

    def _robinson(flat, ns, nr):
        seen["robinson"] = flat
        return {"value": 0.75, "subjects": ns, "raters": nr}

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(
            kripp_alpha=_kripp,
            finn_coefficient=_finn,
            maxwell_re=_maxwell,
            robinson_a=_robinson,
        ),
    )

    mixed = [[True, 2], [1, False]]
    assert reliability.kripp_alpha(mixed).value == pytest.approx(0.5)
    assert reliability.finn_coefficient(mixed, 5).value == pytest.approx(0.4)
    assert reliability.maxwell_re(mixed).value == pytest.approx(0.25)
    assert reliability.robinson_a(mixed).value == pytest.approx(0.75)

    for flat in seen.values():
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float64
        assert flat.tolist() == [1.0, 2.0, 1.0, 0.0]


def test_kripp_alpha_rejects_oversized_view_before_native_discovery(monkeypatch):
    """Rater evidence shares the reliability logical-cell resource ceiling."""
    oversized = np.broadcast_to(
        np.array([[1.0]], dtype=np.float64),
        (2, 10_000_001),
    )
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="reliability evidence exceeds 20000000 cells"):
        reliability.kripp_alpha(oversized)


def test_remaining_rater_top_level_exports_use_hardened_adapters():
    """Historical package exports must use the same hardened callables."""
    assert fast_mlsirm.kripp_alpha is reliability.kripp_alpha
    assert fast_mlsirm.finn_coefficient is reliability.finn_coefficient
    assert fast_mlsirm.maxwell_re is reliability.maxwell_re
    assert fast_mlsirm.robinson_a is reliability.robinson_a
