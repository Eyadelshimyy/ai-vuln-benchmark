"""Pin the fixed ``jg_class`` contract: "not applicable" (``"U"``), always.

Superseded contract, before #1880. ``logistic_dif`` classified items by comparing
``delta_r2`` -- a Nagelkerke pseudo-R-squared change across the two-degree-of-freedom
omnibus -- to the Jodoin-Gierl ``.035`` / ``.070`` boundaries, and reported
``delta_r2_uniform`` without a class. Jodoin & Gierl (2001), *Applied Measurement in
Education, 14*(4), 329-349, states the bands on the ONE-degree-of-freedom UNIFORM
increment of a Zumbo-Thomas weighted-least-squares partition (p. 335, p. 333), not the
Nagelkerke two-degree-of-freedom quantity this package computes -- so the applied
quantity was the one the bands were not built for.

Fixed contract (#1880). Correcting *which* quantity gets a letter is not available:
the replacement statistic is itself underdetermined by the source (eq. 4, p. 333, does
not say whether the correlation is against the observed or the working response, nor
on which scale the coefficient is standardized -- both choices change the number). So
``jg_class`` is retired to ``"U"`` ("not applicable") unconditionally, for every item,
regardless of fit success, omnibus significance, or which of ``delta_r2`` /
``delta_r2_uniform`` is inspected. ``delta_r2`` and ``delta_r2_uniform`` remain
reported as descriptive numbers with no letter attached to either.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import logistic_dif

N_PER_GROUP = 1_500
N_ITEMS = 8
CROSSING_ITEM = 3
INTERCEPT_SHIFT = 1.5
SLOPE_DIFFERENCE = 2.0
SEED = 20_260_915


@pytest.fixture(scope="module")
def sweep():
    """One CROSSING item: a group intercept shift AND a group slope difference.

    Crossing rather than purely uniform because that is the only place the two
    Nagelkerke quantities (``delta_r2`` vs. ``delta_r2_uniform``) separate at all --
    under pure uniform DIF the interaction term contributes almost nothing and the
    two increments nearly coincide. This keeps the fixture useful as a regression
    guard even though neither quantity is lettered any more: it still exercises
    that the descriptive numbers themselves stay distinct and finite.
    """
    rng = np.random.default_rng(SEED)
    theta = rng.standard_normal(2 * N_PER_GROUP)
    group = np.repeat([0, 1], N_PER_GROUP)
    focal = group == 1
    columns = []
    for item in range(N_ITEMS):
        slope = 1.2 + (SLOPE_DIFFERENCE * focal if item == CROSSING_ITEM else 0.0)
        intercept = INTERCEPT_SHIFT * focal if item == CROSSING_ITEM else 0.0
        p = 1.0 / (1.0 + np.exp(-(slope * theta + intercept)))
        columns.append((rng.random(theta.size) < p).astype(np.int64))
    return logistic_dif(np.column_stack(columns), group)


def test_jg_class_is_not_applicable_for_every_item(sweep) -> None:
    """The letter class is retired, not repointed at a different quantity."""
    assert list(sweep["jg_class"]) == ["U"] * N_ITEMS, (
        "jg_class must be 'U' (not applicable) unconditionally -- if this now "
        "reports a letter again, the classifier has been un-retired without "
        "resolving why it was retired (see this module's docstring)"
    )


def test_the_two_quantities_remain_distinct_descriptive_numbers(sweep) -> None:
    """The uniform increment is a different number, not a rounding of the same one."""
    finite = np.isfinite(sweep["delta_r2"]) & np.isfinite(sweep["delta_r2_uniform"])
    assert finite.any()
    assert not np.allclose(
        sweep["delta_r2"][finite], sweep["delta_r2_uniform"][finite], atol=1e-6
    )
    # The crossing item is exactly where they separate; sanity-check it stays large
    # enough to be a meaningful regression guard for the fixture itself.
    assert sweep["delta_r2"][CROSSING_ITEM] > sweep["delta_r2_uniform"][CROSSING_ITEM]


def test_the_uniform_increment_is_reported_without_a_class(sweep) -> None:
    """The contract: neither quantity carries a letter class."""
    assert "delta_r2_uniform" in sweep
    assert "jg_class_uniform" not in sweep
