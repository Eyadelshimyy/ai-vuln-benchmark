"""Regression coverage for historical reliability shape diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import reliability


def _core_must_not_run():
    raise AssertionError("compiled-core discovery reached invalid shape evidence")


@pytest.mark.parametrize(
    ("invoke", "message"),
    [
        (
            lambda: reliability.icc(np.array([1.0, 2.0])),
            "ratings must be a 2-D subjects x raters array",
        ),
        (
            lambda: reliability.guttman_lambdas(np.array([1.0, 2.0])),
            "data must be a 2-D persons x items array",
        ),
        (
            lambda: reliability.tenberge_mu(np.array([1.0, 2.0])),
            "data must be a 2-D persons x items array",
        ),
        (
            lambda: reliability.cronbach_alpha(np.array([1.0, 2.0])),
            "data must be a 2-D persons x items array",
        ),
        (
            lambda: reliability.mean_pairwise_cor(np.array([1.0, 2.0])),
            "ratings must be a 2-D subjects x raters array",
        ),
        (
            lambda: reliability.mean_pairwise_rho(np.array([1.0, 2.0])),
            "ratings must be a 2-D subjects x raters array",
        ),
    ],
)
def test_hardened_reliability_preserves_historical_shape_diagnostics(
    monkeypatch,
    invoke,
    message,
):
    monkeypatch.setattr(fitstats, "_core_module", _core_must_not_run)
    with pytest.raises(ValueError, match=message):
        invoke()


def test_shape_rejection_precedes_native_dispatch_even_with_core_present(monkeypatch):
    def _dispatch(*args, **kwargs):
        raise AssertionError("Rust dispatch reached invalid shape evidence")

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(
            icc=_dispatch,
            guttman_lambdas=_dispatch,
            tenberge_mu=_dispatch,
            cronbach_alpha=_dispatch,
            mean_pairwise_cor=_dispatch,
            mean_pairwise_rho=_dispatch,
        ),
    )
    with pytest.raises(
        ValueError,
        match="ratings must be a 2-D subjects x raters array",
    ):
        reliability.icc([[[1.0]]])
