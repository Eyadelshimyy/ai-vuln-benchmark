"""Coverage for fixed-item linking and IRT scale linking (linking.py)."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.linking import IrtLinkResult, irt_link, link_fixed_item_parameters
from fast_mlsirm.types import MLSIRMParams


def _params(n_persons=10, n_items=6, n_dims=1, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return MLSIRMParams(
        theta=rng.standard_normal((n_persons, n_dims)),
        alpha=np.log(0.7 + 0.4 * rng.random(n_items)) + shift,
        b=np.linspace(-1.0, 1.0, n_items) + shift,
        xi=rng.standard_normal((n_persons, 2)),
        zeta=rng.standard_normal((n_items, 2)),
        tau=0.0,
    )


def test_link_fixed_happy_path():
    source = _params(seed=1, shift=0.2)
    target = _params(seed=2)
    linked, coeffs = link_fixed_item_parameters(source, target, np.array([0, 1, 2]))
    assert isinstance(linked, MLSIRMParams)
    assert coeffs["scale"].shape == (1,)
    assert coeffs["anchor_items"].tolist() == [0, 1, 2]


def test_link_fixed_multidim_skips_dim_without_anchor():
    source = _params(n_dims=2, seed=1)
    target = _params(n_dims=2, seed=2)
    factor_id = np.array([0, 0, 0, 1, 1, 1])
    # anchors only on dimension 0 -> dimension 1 has no anchors (continue branch)
    _linked, coeffs = link_fixed_item_parameters(
        source, target, np.array([0, 1]), factor_id=factor_id
    )
    assert coeffs["scale"][1] == 1.0


def test_link_fixed_rejects_bad_anchor_items():
    source = _params()
    target = _params(seed=2)
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.zeros((2, 2)))  # not 1-D
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([]))  # empty
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0.5, 1.0]))  # non-integer
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0, 99]))  # out of range
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0, 0]))  # not unique


def test_link_fixed_rejects_shape_mismatches():
    source = _params(n_items=6)
    target = _params(n_items=5)
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0, 1]))


def test_link_fixed_rejects_theta_dim_mismatch():
    source = _params(n_dims=1)
    target = _params(n_dims=2)
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0, 1]))


def test_link_fixed_rejects_non_2d_theta():
    source = _params(n_dims=1)
    source.theta = source.theta.ravel()  # collapse to 1-D
    target = _params(n_dims=1, seed=2)
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0, 1]))


def test_link_fixed_rejects_factor_id_over_n_dims():
    source = _params(n_dims=1)
    target = _params(n_dims=1, seed=2)
    factor_id = np.array([0, 2, 0, 0, 0, 0])  # 2 < n_items but >= n_dims
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0, 1]), factor_id=factor_id)


def test_link_fixed_rejects_non_finite_parameters():
    source = _params()
    source.b[0] = np.inf
    target = _params(seed=2)
    with pytest.raises(ValueError):
        link_fixed_item_parameters(source, target, np.array([0, 1]))


def test_link_fixed_rejects_bad_factor_id():
    source = _params(n_dims=1)
    target = _params(n_dims=1, seed=2)
    with pytest.raises(ValueError):
        link_fixed_item_parameters(
            source, target, np.array([0, 1]), factor_id=np.array([0.5] * 6)
        )
    with pytest.raises(ValueError):
        link_fixed_item_parameters(
            source, target, np.array([0, 1]), factor_id=np.zeros(3)
        )


def test_link_fixed_rejects_underflowed_target_slope():
    # A finite but very negative target alpha passes the finiteness check yet
    # underflows to a zero slope, reaching the "positive slopes" guard.
    source = _params(seed=1)
    target = _params(seed=2)
    target.alpha[0] = -800.0
    assert np.exp(target.alpha[0]) == 0.0
    with pytest.raises(ValueError, match="target anchor slopes must be positive"):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))


def test_link_fixed_rejects_underflowed_source_slope():
    # Every target slope is strictly positive, but a source anchor slope that
    # underflows to 0 sends log(source.a / target_a) to -inf, so the scale
    # underflows to 0.0 and the non-positive-coefficient guard fires.
    source = _params(seed=1)
    target = _params(seed=2)
    source.alpha[0] = -800.0
    assert np.exp(source.alpha[0]) == 0.0
    with np.errstate(divide="ignore"), pytest.raises(
        ValueError, match="non-finite or non-positive linking coefficients"
    ):
        link_fixed_item_parameters(source, target, np.array([0, 1, 2]))


def test_irt_link_happy_path():
    res = irt_link(
        np.array([1.0, 1.2, 0.8]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.1, 1.0, 0.9]),
        np.array([-0.9, 0.1, 1.1]),
        method="stocking_lord",
        q_theta=7,
    )
    assert isinstance(res, IrtLinkResult)
    assert np.isfinite(res.slope)


def test_irt_link_rejects_bad_arrays():
    ok = np.array([1.0, 1.2, 0.8])
    with pytest.raises(ValueError):
        irt_link(np.zeros((3, 1)), ok, ok, ok)  # not 1-D
    with pytest.raises(ValueError):
        irt_link(np.array([np.inf, 1.0, 1.0]), ok, ok, ok)  # non-finite
    with pytest.raises(ValueError):
        irt_link(ok, np.array([0.0, 1.0]), ok, ok)  # length mismatch


def test_irt_link_rejects_non_positive_slopes():
    b = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        irt_link(np.array([1.0, 0.0, 1.0]), b, np.array([1.0, 1.0, 1.0]), b)
    with pytest.raises(ValueError):
        irt_link(np.array([1.0, 1.0, 1.0]), b, np.array([1.0, -1.0, 1.0]), b)


def test_irt_link_rejects_bad_q_theta():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        irt_link(a, b, a, b, q_theta=True)
    with pytest.raises(ValueError):
        irt_link(a, b, a, b, q_theta=2.5)
