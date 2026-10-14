"""Regression tests for ICC ratings admission before native discovery."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability
from fast_mlsirm import _icc_control_safety as icc_control_safety


class _HostileArrayProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller-controlled __array__ callback executed")


class _HostileClassSpoof:
    def __init__(self) -> None:
        self.class_reads = 0

    @property
    def __class__(self):
        self.class_reads += 1
        raise AssertionError("caller-controlled __class__ callback executed")


def _native_discovery_must_not_run():
    raise AssertionError("compiled-core discovery ran before ICC ratings admission")


def test_icc_rejects_array_provider_without_callback_or_native(monkeypatch):
    hostile = _HostileArrayProvider()
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="real numeric"):
        reliability.icc(hostile)

    assert hostile.calls == 0


@pytest.mark.parametrize(
    "api",
    (reliability.icc, reliability.mean_pairwise_cor, reliability.mean_pairwise_rho),
)
def test_reliability_mask_probe_rejects_class_spoof_without_callback_or_native(
    monkeypatch, api
):
    hostile = _HostileClassSpoof()
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match="real numeric"):
        api(hostile)

    assert hostile.class_reads == 0


@pytest.mark.parametrize(
    ("ratings", "message"),
    [
        (
            np.array([[1.0 + 0.5j, 2.0], [2.0, 1.0], [3.0, 4.0]], dtype=np.complex128),
            "ratings must be real-valued",
        ),
        (
            np.array([[True, False], [False, True]], dtype=np.bool_),
            "ratings must be numeric, not boolean",
        ),
        (
            np.array([[object(), object()], [object(), object()]], dtype=object),
            "object-dtype arrays are not supported; pass a numeric array",
        ),
        (
            np.array([["1", "2"], ["2", "1"]], dtype=np.str_),
            "ratings must be a numeric array",
        ),
    ],
)
def test_icc_preserves_trusted_ndarray_diagnostics_before_native(
    monkeypatch, ratings, message
):
    """Callback safety must not erase established diagnostics for inert ndarrays."""
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match=message):
        reliability.icc(ratings)


@pytest.mark.parametrize(
    "api",
    (reliability.mean_pairwise_cor, reliability.mean_pairwise_rho),
)
@pytest.mark.parametrize(
    ("ratings", "message"),
    [
        (
            np.array([[1.0 + 0.5j, 2.0], [2.0, 1.0], [3.0, 4.0]], dtype=np.complex128),
            "ratings must be real-valued",
        ),
        (
            np.array([[True, False], [False, True]], dtype=np.bool_),
            "ratings must be numeric, not boolean",
        ),
        (
            np.array([[object(), object()], [object(), object()]], dtype=object),
            "object-dtype arrays are not supported; pass a numeric array",
        ),
        (
            np.array([["1", "2"], ["2", "1"]], dtype=np.str_),
            "ratings must be a numeric array",
        ),
    ],
)
def test_pairwise_preserves_trusted_ndarray_diagnostics_before_native(
    monkeypatch, api, ratings, message
):
    """Pairwise wrappers preserve the same inert-ndarray error contract."""
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)

    with pytest.raises(ValueError, match=message):
        api(ratings)


def test_icc_preserves_masked_array_diagnostic_before_native(monkeypatch):
    monkeypatch.setattr(fitstats, "_core_module", _native_discovery_must_not_run)
    ratings = np.ma.array(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
        mask=[[False, False], [False, True], [False, False]],
    )

    with pytest.raises(
        ValueError,
        match="masked arrays are not supported; use NaN for missing",
    ):
        reliability.icc(ratings)


def test_icc_preserves_trusted_sequence_compatibility(monkeypatch):
    seen: dict[str, object] = {}

    def _icc(flat, n_subjects, n_raters, model, kind, unit, r0, conf_level):
        seen["flat"] = flat
        seen["shape"] = (n_subjects, n_raters)
        seen["controls"] = (model, kind, unit, r0, conf_level)
        return {
            "value": 0.5,
            "subjects": n_subjects,
            "raters": n_raters,
            "fvalue": 2.0,
            "df1": 2.0,
            "df2": 3.0,
            "p_value": 0.2,
            "lbound": 0.1,
            "ubound": 0.8,
        }

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(icc=_icc))

    r0 = np.float32(0.1)
    conf_level = np.float64(0.9)
    result = reliability.icc(
        [
            [np.float32(1.0), 2],
            [2.0, np.int16(1)],
            [3, np.float64(4.0)],
        ],
        model="twoway",
        type="agreement",
        unit="average",
        r0=r0,
        conf_level=conf_level,
    )

    assert result.value == pytest.approx(0.5)
    assert seen["shape"] == (3, 2)
    assert seen["controls"] == (
        "twoway",
        "agreement",
        "average",
        float(r0),
        float(conf_level),
    )
    assert isinstance(seen["flat"], np.ndarray)
    assert seen["flat"].dtype == np.float64


def test_icc_preserves_exact_ndarray_rows_inside_builtin_sequence(monkeypatch):
    seen: dict[str, object] = {}

    def _icc(flat, n_subjects, n_raters, model, kind, unit, r0, conf_level):
        seen["flat"] = flat
        seen["shape"] = (n_subjects, n_raters)
        return {
            "value": 0.5,
            "subjects": n_subjects,
            "raters": n_raters,
            "fvalue": 2.0,
            "df1": 2.0,
            "df2": 3.0,
            "p_value": 0.2,
            "lbound": 0.1,
            "ubound": 0.8,
        }

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(icc=_icc))

    result = reliability.icc(
        [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([2, 1], dtype=np.int16),
            np.array([3.0, 4.0], dtype=np.float64),
        ]
    )

    assert result.value == pytest.approx(0.5)
    assert seen["shape"] == (3, 2)
    assert isinstance(seen["flat"], np.ndarray)
    assert seen["flat"].dtype == np.float64
    np.testing.assert_array_equal(
        seen["flat"], np.array([1.0, 2.0, 2.0, 1.0, 3.0, 4.0], dtype=np.float64)
    )


def test_icc_safety_installer_is_idempotent():
    original_icc = reliability.icc
    try:
        icc_control_safety.install(reliability)
        assert reliability.icc is original_icc
    finally:
        reliability.icc = original_icc


def test_top_level_icc_export_uses_hardened_adapter():
    assert fast_mlsirm.icc is reliability.icc
