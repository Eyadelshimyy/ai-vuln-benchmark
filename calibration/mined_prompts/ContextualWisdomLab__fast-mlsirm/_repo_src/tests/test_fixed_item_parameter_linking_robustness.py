"""Robustness of fixed-item-parameter (common-item) linking to partial anchors.

Fixed-item-parameter calibration links a new form onto an established scale by
holding a set of *anchor* (common) items at their reference values and deriving
a scale transform from them; the transform then places every *unique*
(non-anchor) item — and the ability metric — on the common scale (Kim, 2006,
*A comparative study of IRT fixed parameter calibration methods*, Journal of
Educational Measurement, 43(4), 355-381,
https://doi.org/10.1111/j.1745-3984.2006.00021.x; Kolen & Brennan, 2014,
*Test Equating, Scaling, and Linking*, 3rd ed., Springer). The defining
property is generalization: the transform is estimated from the common items
only, yet must correctly relocate items that were never part of the anchor set.

``tests/test_irt_stability.py`` exercises ``link_fixed_item_parameters`` only
with *every* item used as an anchor, so the transform is fit on the same items
it is checked against. These tests pin the realistic partial-anchor case: a
transform recovered from a strict subset of common items places the remaining
unique items on the metric, is invariant to which anchor subset is used, and is
identified from a single anchor when the forms differ by an exact linear map.
"""

import numpy as np

from fast_mlsirm.linking import link_fixed_item_parameters
from fast_mlsirm.types import MLSIRMParams


def _exact_transform_pair(scale: float, shift: float, seed: int = 0):
    """Return (source, target) that differ by an exact linear scale transform.

    ``target`` is ``source`` re-expressed on a metric obtained by
    ``theta -> scale*theta + shift`` (with the matching slope/intercept
    transform), so any non-empty anchor set recovers ``scale``/``shift`` exactly
    and the linked result should reproduce ``target`` for every item.
    """
    rng = np.random.default_rng(seed)
    n_items, n_persons = 6, 4
    source = MLSIRMParams(
        theta=rng.standard_normal((n_persons, 1)),
        alpha=np.log(0.7 + 1.0 * rng.random(n_items)),
        b=-1.0 + 2.0 * rng.random(n_items),
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
    )
    target = source.copy()
    target.theta = scale * source.theta + shift
    target.alpha = np.log(source.a / scale)
    target.b = source.b - target.a * shift
    return source, target


def test_partial_anchor_linking_places_unique_items_on_the_common_metric():
    """A transform fit on a subset of anchors relocates the non-anchor items.

    Three of six items serve as common anchors; the recovered scale/shift must
    match the true transform and, crucially, the three *unique* items that were
    never anchors must land on the target metric — the generalization property
    that makes common-item linking work.
    """
    scale, shift = 1.4, -0.3
    source, target = _exact_transform_pair(scale, shift)
    anchors = np.array([0, 2, 4])
    unique_items = np.array([1, 3, 5])

    linked, transform = link_fixed_item_parameters(source, target, anchor_items=anchors)

    assert np.isclose(transform["scale"][0], scale)
    assert np.isclose(transform["shift"][0], shift)
    # The non-anchor items are placed on the common metric though they never
    # entered the transform estimate.
    assert np.allclose(linked.alpha[unique_items], target.alpha[unique_items])
    assert np.allclose(linked.b[unique_items], target.b[unique_items])
    # The ability metric is transformed consistently for every person.
    assert np.allclose(linked.theta, target.theta)


def test_fixed_item_linking_is_invariant_to_anchor_subset_choice():
    """Any anchor subset yields the same transform under an exact linear map.

    When the forms differ by a single global scale/shift, the linking
    coefficients are over-identified: disjoint anchor subsets must agree, so the
    common metric does not depend on which common items an operator happens to
    pick.
    """
    scale, shift = 0.8, 0.6
    source, target = _exact_transform_pair(scale, shift, seed=3)

    _, transform_a = link_fixed_item_parameters(source, target, anchor_items=np.array([0, 2, 4]))
    _, transform_b = link_fixed_item_parameters(source, target, anchor_items=np.array([1, 3]))
    _, transform_c = link_fixed_item_parameters(source, target, anchor_items=np.array([5]))

    for other in (transform_b, transform_c):
        assert np.isclose(transform_a["scale"][0], other["scale"][0])
        assert np.isclose(transform_a["shift"][0], other["shift"][0])
    assert np.isclose(transform_a["scale"][0], scale)
    assert np.isclose(transform_a["shift"][0], shift)


def test_single_anchor_identifies_an_exact_linear_transform():
    """One common item is enough to recover an exact scale/shift and link all items.

    The mean-based coefficients reduce to that single anchor's ratio and
    difference, so a one-item anchor set recovers the transform and reproduces
    the full target form — the minimal-anchor boundary of fixed-item linking.
    """
    scale, shift = 1.25, -0.4
    source, target = _exact_transform_pair(scale, shift, seed=7)

    linked, transform = link_fixed_item_parameters(source, target, anchor_items=np.array([3]))

    assert np.isclose(transform["scale"][0], scale)
    assert np.isclose(transform["shift"][0], shift)
    assert np.allclose(linked.alpha, target.alpha)
    assert np.allclose(linked.b, target.b)
    assert np.allclose(linked.theta, target.theta)
