"""Coverage for observed-score equating and linking (equating.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import fast_mlsirm.equating as E
from fast_mlsirm.equating import (
    CircleArcResult,
    EquateResult,
    circle_arc_equate,
    circle_arc_middle_anchor,
    composite_linking,
    equate_neat,
    equate_neat_linear,
    equate_observed_scores,
    equate_observed_scores_kernel,
    equating_standard_errors,
    loglinear_smooth,
    nominal_weights_mean_equate,
)

_NONE = "fast_mlsirm.fitstats._core_module"


def _eg(seed=0, n=500, k=20):
    rng = np.random.default_rng(seed)
    return rng.integers(0, k + 1, size=n), rng.integers(0, k + 1, size=n)


def _neat(seed=1, n=300):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, 21, n),
        rng.integers(0, 11, n),
        rng.integers(0, 21, n),
        rng.integers(0, 11, n),
    )


# ------------------------- equate_observed_scores + _infer_k -------------------------

def test_equate_observed_scores_happy_and_inference():
    xs, ys = _eg()
    res = equate_observed_scores(xs, ys, method="linear", k_x=20, k_y=20)
    assert isinstance(res, EquateResult)
    assert res.design == "EG"
    # k inference path (k_x/k_y omitted)
    assert isinstance(equate_observed_scores(xs, ys, method="mean"), EquateResult)


def test_infer_k_rejects_empty_and_non_finite():
    _, ys = _eg()
    with pytest.raises(ValueError):
        equate_observed_scores(np.array([]), ys)
    with pytest.raises(ValueError):
        equate_observed_scores(np.array([0.0, 1.0, np.inf]), ys)


def test_equate_observed_scores_requires_rust_core():
    xs, ys = _eg()
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            equate_observed_scores(xs, ys)


# ------------------------- equate_neat / equate_neat_linear -------------------------

def test_equate_neat_happy_and_core():
    xt, xa, yt, ya = _neat()
    assert equate_neat(xt, xa, yt, ya, method="chained", k_x=20, k_y=20, k_v=10).design == "NEAT"
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            equate_neat(xt, xa, yt, ya)


def test_equate_neat_linear_happy_and_core():
    xt, xa, yt, ya = _neat()
    res = equate_neat_linear(xt, xa, yt, ya, method="tucker", k_x=20, k_y=20)
    assert res.design == "NEAT"
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            equate_neat_linear(xt, xa, yt, ya)


# ------------------------- loglinear_smooth -------------------------

def test_loglinear_smooth_happy_and_short_clamp_and_core():
    xs, _ = _eg()
    counts = np.bincount(xs, minlength=21).astype(float)
    out = loglinear_smooth(counts, degree=6)
    assert "probs" in out
    # short-form degree clamp
    assert "probs" in loglinear_smooth(np.array([3.0, 4.0, 5.0]), degree=6)
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            loglinear_smooth(counts)


# ------------------------- equate_observed_scores_kernel -------------------------

def test_kernel_happy_and_guards():
    xs, ys = _eg()
    assert isinstance(
        equate_observed_scores_kernel(xs, ys, k_x=20, k_y=20), EquateResult
    )
    with pytest.raises(ValueError):
        equate_observed_scores_kernel(xs, ys, smooth_x=0, k_x=20, k_y=20)
    with pytest.raises(ValueError):
        equate_observed_scores_kernel(xs, ys, smooth_y=0, k_x=20, k_y=20)
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            equate_observed_scores_kernel(xs, ys)


# ------------------------- equating_standard_errors -------------------------

def test_see_bootstrap_and_analytic_routes():
    xs, ys = _eg()
    boot = equating_standard_errors(
        xs, ys, method="mean", route="bootstrap", k_x=20, k_y=20, n_boot=20
    )
    assert boot["n_boot"] == 20
    ana = equating_standard_errors(
        xs, ys, method="linear", route="analytic", k_x=20, k_y=20
    )
    assert ana["n_boot"] == 0


def test_see_invalid_route_and_missing_core():
    xs, ys = _eg()
    with pytest.raises(ValueError):
        equating_standard_errors(xs, ys, route="bogus", k_x=20, k_y=20)
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            equating_standard_errors(xs, ys, k_x=20, k_y=20)


def test_see_missing_backend_methods():
    xs, ys = _eg()
    # a core object lacking the specific SEE entry points triggers the guard
    with patch(_NONE, return_value=object()):
        with pytest.raises(RuntimeError):
            equating_standard_errors(xs, ys, route="bootstrap", k_x=20, k_y=20)
    with patch(_NONE, return_value=object()):
        with pytest.raises(RuntimeError):
            equating_standard_errors(xs, ys, route="analytic", k_x=20, k_y=20)


# ------------------------- circle_arc_equate + _ca_point -------------------------

def test_circle_arc_happy_and_core():
    res = circle_arc_equate(
        np.array([0.0, 10.0, 20.0]), (0.0, 0.0), (10.0, 11.0), (20.0, 20.0)
    )
    assert isinstance(res, CircleArcResult)
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            circle_arc_equate(np.array([0.0]), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0))


def test_circle_arc_rejects_bad_scores():
    with pytest.raises(ValueError):
        circle_arc_equate(np.array([1j]), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
    with pytest.raises(ValueError):
        circle_arc_equate(np.array(["a"]), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0))


def test_circle_arc_point_validation():
    scores = np.array([0.0, 10.0, 20.0])
    with pytest.raises(ValueError):
        circle_arc_equate(scores, (0.0, 0.0, 0.0), (10.0, 11.0), (20.0, 20.0))
    with pytest.raises(ValueError):
        circle_arc_equate(scores, (1j, 0.0), (10.0, 11.0), (20.0, 20.0))
    with pytest.raises(ValueError):
        circle_arc_equate(scores, ("a", "b"), (10.0, 11.0), (20.0, 20.0))


# ------------------------- circle_arc_middle_anchor -------------------------

def test_circle_arc_middle_anchor_happy_and_guards():
    assert circle_arc_middle_anchor(10.0, 5.0, 10.0, 4.0, 5.0, 4.0) == (10.0, 10.0)
    with pytest.raises(ValueError):
        circle_arc_middle_anchor(1j, 5.0, 10.0, 4.0, 5.0, 4.0)
    with pytest.raises(ValueError):
        circle_arc_middle_anchor("a", 5.0, 10.0, 4.0, 5.0, 4.0)
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            circle_arc_middle_anchor(10.0, 5.0, 10.0, 4.0, 5.0, 4.0)


# ------------------------- nominal_weights_mean_equate -------------------------

def test_nominal_weights_happy_and_guards():
    xt, xa, yt, ya = _neat()
    assert nominal_weights_mean_equate(xt, xa, yt, ya, 20, 20, 10).design == "NEAT"
    with pytest.raises(ValueError):
        nominal_weights_mean_equate(np.array([1j]), xa, yt, ya, 20, 20, 10)
    with pytest.raises(ValueError):
        nominal_weights_mean_equate(np.array(["a"]), xa, yt, ya, 20, 20, 10)
    with pytest.raises(ValueError):
        nominal_weights_mean_equate(xt, xa, yt, ya, 0, 20, 10)
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            nominal_weights_mean_equate(xt, xa, yt, ya, 20, 20, 10)


# ------------------------- composite_linking -------------------------

def test_composite_linking_happy_and_guards():
    tables = [np.arange(5.0), np.arange(5.0) + 1.0]
    assert "composite" in composite_linking(tables, [0.5, 0.5])
    assert composite_linking(tables, [0.5, 0.5], slopes=[1.0, 1.2])["symmetric"]
    with pytest.raises(ValueError):
        composite_linking([np.array([1j, 2j])], [1.0])
    with pytest.raises(ValueError):
        composite_linking([np.array(["a", "b"])], [1.0])
    with pytest.raises(ValueError):
        composite_linking(tables, np.array([1j, 2j]))
    with pytest.raises(ValueError):
        composite_linking(tables, np.array(["a", "b"]))
    with pytest.raises(ValueError):
        composite_linking(tables, [0.5, 0.5], slopes=np.array([1j, 2j]))
    with pytest.raises(ValueError):
        composite_linking(tables, [0.5, 0.5], slopes=np.array(["a", "b"]))
    with patch(_NONE, return_value=None):
        with pytest.raises(RuntimeError):
            composite_linking(tables, [0.5, 0.5])
