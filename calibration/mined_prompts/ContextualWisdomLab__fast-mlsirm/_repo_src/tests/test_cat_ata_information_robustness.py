"""Paper-grounded robustness of maximum-information CAT/ATA item selection.

Computerized adaptive testing selects, at the current ability estimate, the item
that maximizes Fisher information; the maximum-information (MFI) criterion is the
canonical rule (van der Linden & Pashley, 2010, *Item selection and ability
estimation in adaptive testing*, in *Elements of Adaptive Testing*, Springer,
pp. 3-30, https://doi.org/10.1007/978-0-387-85461-8_1). For a dichotomous item
Fisher information is ``I(theta) = a^2 * P * (1 - P)`` (Lord, 1980,
*Applications of Item Response Theory to Practical Testing Problems*), which is
maximized at ``P = 0.5`` — i.e. where the item's location matches the person's
ability. Automated test assembly then chooses a fixed-length set of items that
maximizes total information (van der Linden, 2005, *Linear Models for Optimal
Test Design*, Springer, https://doi.org/10.1007/0-387-29054-0).

``tests/test_irt_stability.py`` pins CAT selection only for items that share a
difficulty (``b = 0``, so only discrimination varies) plus ATA content
constraints. These tests pin the two properties that case leaves untested: MFI
selection *tracks ability* across the difficulty axis, and unconstrained ATA is
*information-optimal* over every same-length subset.
"""

from itertools import combinations

import numpy as np

from fast_mlsirm.test_design import assemble_test_form, item_information, select_cat_item
from fast_mlsirm.types import MLSIRMParams


def _equal_discrimination_bank(difficulties: np.ndarray) -> MLSIRMParams:
    """Return a simple-structure bank whose items differ only in difficulty.

    ``alpha = 0`` gives every item unit discrimination (``a = exp(alpha) = 1``),
    so Fisher information is driven purely by how close each item's location is
    to the ability being probed.
    """
    n_items = difficulties.size
    return MLSIRMParams(
        theta=np.zeros((1, 1)),
        alpha=np.zeros(n_items),
        b=np.asarray(difficulties, dtype=np.float64),
        xi=np.zeros((1, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=-2.0,
    )


def test_cat_selection_tracks_ability_via_maximum_fisher_information():
    """MFI selection picks the item whose location matches the ability estimate.

    With equal discrimination the linear predictor is ``a*theta + b``, so
    ``P = 0.5`` (peak Fisher information) occurs at ``b = -theta``. The selected
    item must therefore move across the difficulty axis in lock-step with the
    probed ability, and the peak information must equal ``a^2 * 0.25 = 0.25``.
    """
    difficulties = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    factors = np.zeros(difficulties.size, dtype=int)
    params = _equal_discrimination_bank(difficulties)

    for theta in difficulties:
        information = item_information(
            params, factors, theta=np.array([[theta]]), model="MIRT"
        )
        chosen = select_cat_item(
            params, factors, theta=np.array([[theta]]), model="MIRT"
        )
        # The maximum-information item is the one located at b = -theta.
        assert np.isclose(difficulties[chosen], -theta)
        assert chosen == int(np.argmax(information))
        # a = 1, so peak Fisher information is 0.25 (P = 0.5).
        assert np.isclose(information[chosen], 0.25, atol=1e-9)
        # Information is symmetric and unimodal about the matched location.
        ordered = information[np.argsort(difficulties)]
        peak = int(np.argmax(ordered))
        assert np.all(np.diff(ordered[: peak + 1]) >= -1e-12)
        assert np.all(np.diff(ordered[peak:]) <= 1e-12)


def test_cat_selection_excludes_administered_and_prefers_discrimination():
    """Administered items are never re-selected and higher discrimination wins ties.

    At ``theta = 0`` with all items located at ``b = 0`` every item sits at
    ``P = 0.5``, so Fisher information ``a^2 * 0.25`` is ordered purely by
    discrimination; excluding the most informative item must hand selection to
    the next-most-discriminating unused item.
    """
    alpha = np.log(np.array([0.5, 2.0, 1.0, 1.5]))  # a = 0.5, 2.0, 1.0, 1.5
    params = MLSIRMParams(
        theta=np.zeros((1, 1)),
        alpha=alpha,
        b=np.zeros(4),
        xi=np.zeros((1, 1)),
        zeta=np.zeros((4, 1)),
        tau=-2.0,
    )
    factors = np.zeros(4, dtype=int)
    theta = np.array([[0.0]])

    # Unrestricted: the a = 2.0 item (index 1) is most informative.
    assert select_cat_item(params, factors, theta=theta, model="MIRT") == 1
    # Exclude it: the a = 1.5 item (index 3) is next.
    assert (
        select_cat_item(
            params, factors, theta=theta, administered=np.array([1]), model="MIRT"
        )
        == 3
    )


def test_ata_greedy_form_is_maximum_information_optimal():
    """Unconstrained ATA returns the fixed-length set of maximum total information.

    With no content constraints, assembling a length-k form must select the k
    items whose summed Fisher information is at least that of every other
    k-subset — the optimal-test-design objective. Verified exhaustively against
    all C(n, k) subsets for a small bank.
    """
    information = np.array([0.31, 0.12, 0.44, 0.28, 0.09, 0.37, 0.19])
    n_items = information.size
    for length in range(1, n_items + 1):
        form = assemble_test_form(information, length=length)
        assert form.size == length
        assert len(set(form.tolist())) == length  # no item chosen twice
        chosen_total = float(information[form].sum())
        best_total = max(
            float(information[list(subset)].sum())
            for subset in combinations(range(n_items), length)
        )
        assert np.isclose(chosen_total, best_total)
