"""Regression evidence for the NumPy MMLE vectorized M-step fallback.

The pinned values were regenerated when the slope bound was made symmetric.
The previous oracle recorded item 2's slope as exactly ``0.001`` — not an
estimate but the old positivity floor, committed as an expectation. That item
is reverse-keyed in this fixture and its unconstrained estimate is ``-0.571``.

Regenerating an oracle from the code it guards is only legitimate when the old
values are known to be wrong for a stated reason, so the check is recorded here
rather than left implicit: the floored item's slope moved from the bound to a
real estimate, the other three moved because the floored item had been
distorting the joint fit, and the final log-likelihood improved from
``-13.879977`` to ``-13.768546``. A bound that costs fit was binding even on six
people and four items.
"""

import numpy as np

from fast_mlsirm.estimators.mmle import fit_mmle_2pl


def test_vectorized_mmle_matches_pre_vectorization_regression_oracle():
    """Preserve the deterministic scalar-M-step result within numeric tolerance."""
    responses = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    observed = np.array(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, False, True],
            [True, False, True, True],
        ]
    )

    result = fit_mmle_2pl(
        responses,
        observed,
        n_nodes=9,
        max_iter=4,
        tol=1e-12,
        seed=7,
    )

    np.testing.assert_allclose(
        result["a"],
        [1.8452420894875856, -0.5709225084629108, 0.3631825757247221, 0.9889385489229235],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["b"],
        [
            0.014455186910378878,
            -0.34874501615397036,
            1.4822006899271674,
            0.0021710039007449052,
        ],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["theta"],
        [
            0.8781620624591238,
            -0.436451505300687,
            0.10508546942605024,
            -0.6359380349756313,
            0.855356826386072,
            -0.745676944362071,
        ],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["loglik_trace"],
        [
            -14.873979688002832,
            -14.002066971964297,
            -13.86451982515555,
            -13.768545627818519,
        ],
        rtol=1e-10,
        atol=1e-10,
    )
    assert result["n_iter"] == 4
    assert result["status"] == "max_iter_reached"


def test_the_reverse_keyed_item_is_no_longer_pinned_to_the_removed_floor() -> None:
    """Guard the specific value the old oracle had frozen.

    ``0.001`` was the lower clamp bound, so an oracle asserting it was asserting
    that the optimizer had been stopped rather than that it had converged.
    """
    responses = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    observed = np.ones_like(responses, dtype=bool)
    observed[4, 2] = False
    observed[5, 1] = False

    result = fit_mmle_2pl(
        responses, observed, n_nodes=9, max_iter=4, tol=1e-12, seed=7
    )

    assert abs(result["a"][1] - 0.001) > 1e-6
    assert result["a"][1] < 0.0
