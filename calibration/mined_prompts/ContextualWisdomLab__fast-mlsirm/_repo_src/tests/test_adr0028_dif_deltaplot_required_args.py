"""ADR-0028 (#1962) RED coverage for ``fast_mlsirm.dif`` and ``fast_mlsirm.deltaplot``.

Required-argument omission must raise ``TypeError``; deprecated old names
must still work and emit ``DeprecationWarning``; the new name called with
the old default value must reproduce the old alias's behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fast_mlsirm import dif
from fast_mlsirm.deltaplot import delta_plot


def _responses(seed: int = 0, n_persons: int = 40, n_items: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(np.int64)


def _group(n_persons: int = 40) -> np.ndarray:
    return np.array([0] * (n_persons // 2) + [1] * (n_persons - n_persons // 2))


def _ordinal_responses(seed: int = 0, n_persons: int = 40, n_items: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(n_persons, n_items)).astype(np.int64)


# --- required-argument RED tests --------------------------------------------


@pytest.mark.parametrize(
    ("func", "kwargs", "missing"),
    (
        (dif.detect_dif_mantel_haenszel, {}, "fdr_q"),
        (
            dif.detect_dif_mantel_haenszel_purified,
            {},
            "fdr_q/max_rounds/min_anchor_items",
        ),
        (dif.detect_dif_logistic, {}, "fdr_q/max_iter"),
        (
            dif.detect_dif_logistic_purified,
            {},
            "fdr_q/max_iter/max_rounds/min_anchor_items",
        ),
        (dif.detect_dif_breslow_day, {}, "fdr_q"),
        (dif.sibtest, {}, "fdr_q/j_min"),
    ),
)
def test_dif_functions_require_their_thresholds(func, kwargs, missing):
    responses, group = _responses(), _group()
    with pytest.raises(TypeError):
        func(responses, group, **kwargs)


def test_raju_area_requires_alpha():
    z = np.zeros(2)
    ones = np.ones(2)
    with pytest.raises(TypeError):
        dif.raju_area(ones, z, z, z, z, ones, z, z, z, z)


def test_sibtest_requires_fdr_q_and_j_min_individually():
    responses, group = _responses(), _group()
    with pytest.raises(TypeError):
        dif.sibtest(responses, group, j_min=5)
    with pytest.raises(TypeError):
        dif.sibtest(responses, group, fdr_q=0.05)


def test_delta_plot_requires_alpha_and_max_iter():
    responses, group = _responses(n_items=4), _group()
    with pytest.raises(TypeError):
        delta_plot(responses, group, max_iter=10)
    with pytest.raises(TypeError):
        delta_plot(responses, group, alpha=0.05)


# --- deprecated aliases still work and warn ---------------------------------


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    (
        ("mantel_haenszel_dif", "detect_dif_mantel_haenszel"),
        ("mantel_haenszel_dif_purified", "detect_dif_mantel_haenszel_purified"),
        ("logistic_dif", "detect_dif_logistic"),
        ("logistic_dif_purified", "detect_dif_logistic_purified"),
        ("mantel_smd_dif", "detect_dif_mantel_smd"),
        ("gmh_dif", "detect_dif_gmh"),
        ("breslow_day_dif", "detect_dif_breslow_day"),
    ),
)
def test_old_dif_alias_warns_and_matches_new_name(old_name, new_name):
    old_fn = getattr(dif, old_name)
    new_fn = getattr(dif, new_name)
    if new_name in ("detect_dif_mantel_smd", "detect_dif_gmh"):
        responses, group = _ordinal_responses(), _group()
        with pytest.warns(DeprecationWarning, match=old_name):
            old_result = old_fn(responses, group)
        new_result = new_fn(responses, group)
    else:
        responses, group = _responses(), _group()
        with pytest.warns(DeprecationWarning, match=old_name):
            old_result = old_fn(responses, group)
        kwargs = {"fdr_q": 0.05}
        if new_name == "detect_dif_mantel_haenszel_purified":
            kwargs.update(max_rounds=3, min_anchor_items=4)
        elif new_name == "detect_dif_logistic":
            kwargs.update(max_iter=50)
        elif new_name == "detect_dif_logistic_purified":
            kwargs.update(max_iter=50, max_rounds=3, min_anchor_items=4)
        new_result = new_fn(responses, group, **kwargs)

    for key, old_value in old_result.items():
        new_value = new_result[key]
        if isinstance(old_value, np.ndarray) and old_value.dtype.kind == "f":
            assert np.allclose(old_value, new_value, equal_nan=True)
        else:
            assert np.array_equal(old_value, new_value)


def test_old_dif_alias_forwards_exception_types():
    """Deprecated aliases still validate through to the same errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError):
            dif.mantel_haenszel_dif(_responses(), _group(), fdr_q=1.5)
