"""Regression for raw polytomous prediction category-domain replay."""

from __future__ import annotations

import subprocess
import sys


def _run_isolated(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_raw_prediction_replays_category_ceiling_before_core_discovery() -> None:
    """The private raw helper must reject 65 categories before Rust discovery."""

    _run_isolated(
        r'''
import importlib
import numpy as np
import fast_mlsirm.polytomous as poly

poly = importlib.reload(poly)


def core_forbidden():
    raise AssertionError("compiled core discovery must not occur")


poly._core_module = core_forbidden
fit = poly.PolytomousFit(
    model="grm",
    slope=np.ones(1, dtype=np.float64),
    cat_params=np.zeros((1, 64), dtype=np.float64),
    loglik=0.0,
    n_iter=1,
)

try:
    poly._polytomous_predictions(fit, np.array([0.0], dtype=np.float64))
except ValueError as exc:
    assert str(exc) == "n_cat must be in 2..=64", str(exc)
else:
    raise AssertionError("expected category-domain ValueError")
'''
    )


def test_raw_prediction_preserves_supported_64_category_boundary() -> None:
    """The raw helper must still delegate the fitter-supported 64-category edge."""

    _run_isolated(
        r'''
import importlib
import numpy as np
import fast_mlsirm.polytomous as poly

poly = importlib.reload(poly)


class Core:
    def polytomous_predictions(
        self,
        theta,
        slope,
        cat_params,
        n_items,
        n_cat,
        model,
    ):
        assert n_items == 1
        assert n_cat == 64
        assert model == "gpcm"
        assert theta.shape == (1,)
        assert slope.shape == (1,)
        assert cat_params.shape == (63,)
        return {
            "probabilities": np.full(64, 1.0 / 64.0, dtype=np.float64),
            "expected": np.array([31.5], dtype=np.float64),
        }


poly._core_module = lambda: Core()
fit = poly.PolytomousFit(
    model="gpcm",
    slope=np.ones(1, dtype=np.float64),
    cat_params=np.zeros((1, 63), dtype=np.float64),
    loglik=0.0,
    n_iter=1,
)
probabilities, expected = poly._polytomous_predictions(
    fit,
    np.array([0.0], dtype=np.float64),
)
assert probabilities.shape == (1, 1, 64)
assert np.allclose(probabilities.sum(axis=2), 1.0)
assert expected.shape == (1, 1)
assert expected[0, 0] == 31.5
'''
    )
