"""Python binding smoke for two_tier_oakes_se (#1992)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tiny_two_tier():
    rng = np.random.default_rng(20260918)
    n_persons, n_items, n_cat = 40, 4, 3
    # Ensure every category observed
    y = np.zeros((n_persons, n_items), dtype=np.int64)
    for i in range(n_items):
        y[:, i] = np.arange(n_persons) % n_cat
        y[:n_cat, i] = np.arange(n_cat)
    primary_map = np.array(
        [[True, False], [True, False], [False, True], [False, True]],
        dtype=bool,
    )
    specific_map = np.array([0, 0, 0, 0], dtype=np.int64)
    a_primary = np.array(
        [[1.1, 0.0], [0.9, 0.0], [0.0, 1.0], [0.0, 1.2]], dtype=np.float64
    )
    a_specific = np.array([0.7, 0.8, 0.6, 0.9], dtype=np.float64)
    threshold = np.array(
        [[0.5, -0.5], [0.4, -0.4], [0.6, -0.3], [0.3, -0.6]], dtype=np.float64
    )
    phi = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)
    return dict(
        a_primary=a_primary,
        a_specific=a_specific,
        threshold=threshold,
        phi=phi,
        responses=y,
        primary_map=primary_map,
        specific_map=specific_map,
        n_cat=n_cat,
        n_primary=2,
        n_specific=1,
        q_primary=7,
        q_specific=7,
        fd_step=1e-5,
        _rng=rng,
    )


def test_two_tier_oakes_se_smoke(tiny_two_tier):
    pytest.importorskip("fast_mlsirm._core")
    from fast_mlsirm import two_tier_oakes_se

    kwargs = {k: v for k, v in tiny_two_tier.items() if not k.startswith("_")}
    res = two_tier_oakes_se(**kwargs)
    assert len(res.labels) >= 1
    k = len(res.labels)
    assert res.information.shape == (k, k)
    assert np.isfinite(res.information).all()
    if res.positive_definite:
        assert res.se is not None
        assert np.isfinite(res.se).all()
        assert (res.se > 0).all()
    else:
        assert res.se is None
        assert res.non_pd_reason


def test_two_tier_oakes_requires_controls(tiny_two_tier):
    pytest.importorskip("fast_mlsirm._core")
    from fast_mlsirm import two_tier_oakes_se

    kwargs = {k: v for k, v in tiny_two_tier.items() if not k.startswith("_")}
    with pytest.raises(TypeError):
        two_tier_oakes_se(  # type: ignore[call-arg]
            kwargs["a_primary"],
            kwargs["a_specific"],
            kwargs["threshold"],
            kwargs["phi"],
            kwargs["responses"],
            kwargs["primary_map"],
            kwargs["specific_map"],
            kwargs["n_cat"],
            kwargs["n_primary"],
            kwargs["n_specific"],
            # missing q_*/fd_step
        )
