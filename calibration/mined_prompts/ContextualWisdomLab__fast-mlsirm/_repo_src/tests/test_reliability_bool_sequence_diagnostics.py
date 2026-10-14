"""Regression tests for reliability compatibility at inert evidence boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


class _MaskedRatings(np.ma.MaskedArray):
    """Concrete masked-array subtype used to preserve actionable diagnostics."""


def _core_discovery_must_not_run() -> SimpleNamespace:
    """Fail if rejected ratings reach native capability discovery."""
    raise AssertionError("compiled-core discovery ran before ratings rejection")


@pytest.mark.parametrize(
    "invoke",
    [
        lambda ratings: reliability.icc(ratings),
        lambda ratings: reliability.mean_pairwise_cor(ratings),
        lambda ratings: reliability.mean_pairwise_rho(ratings),
    ],
)
@pytest.mark.parametrize(
    "ratings",
    [
        [[True, False], [False, True]],
        [[True, 0], [1, False]],
        [[np.bool_(True), np.bool_(False)], [np.bool_(False), np.bool_(True)]],
        [[np.bool_(True), np.int16(0)], [np.int16(1), np.bool_(False)]],
    ],
)
def test_boolean_rating_sequences_preserve_historical_diagnostic(
    monkeypatch, invoke, ratings
):
    """Any trusted Boolean sequence leaf should keep the ratings diagnostic."""
    monkeypatch.setattr(fitstats, "_core_module", _core_discovery_must_not_run)

    with pytest.raises(ValueError, match="ratings must be numeric, not boolean"):
        invoke(ratings)


@pytest.mark.parametrize(
    "invoke",
    [
        lambda ratings: reliability.icc(ratings),
        lambda ratings: reliability.mean_pairwise_cor(ratings),
        lambda ratings: reliability.mean_pairwise_rho(ratings),
    ],
)
def test_masked_array_subclasses_preserve_actionable_nan_guidance(monkeypatch, invoke):
    """MaskedArray subclasses should keep the same missingness guidance."""
    monkeypatch.setattr(fitstats, "_core_module", _core_discovery_must_not_run)
    ratings = np.ma.array(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
        mask=[[False, False], [False, True], [False, False]],
    ).view(_MaskedRatings)

    with pytest.raises(
        ValueError,
        match="masked arrays are not supported; use NaN for missing",
    ):
        invoke(ratings)
