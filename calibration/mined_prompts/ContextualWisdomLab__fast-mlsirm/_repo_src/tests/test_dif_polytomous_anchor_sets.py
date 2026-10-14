"""Contract for :func:`dif_polytomous_anchor_sets`.

The reason this exists separately from the pooled sweep is that an item can be
invariant against one focal group and not another. The tests are built around
that case, and around the refusal to let a failed purification pass as a strict
result.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import dif_polytomous_anchor_sets

N_CAT = 4
SEED = 20_260_915
THRESHOLDS = np.array([1.2, 0.0, -1.2])
# Issue #1958: model/q_theta/max_iter/tol/fdr_q/max_rounds/min_anchor_items
# are all required caller arguments now (no unsourced defaults).
REQUIRED = dict(
    model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05,
    max_rounds=3, min_anchor_items=4,
)


def _graded_draw(shift, theta, rng):
    ge = np.ones((theta.size, N_CAT + 1))
    for k in range(1, N_CAT):
        ge[:, k] = 1.0 / (1.0 + np.exp(-(1.1 * theta + THRESHOLDS[k - 1] + shift)))
    ge[:, N_CAT] = 0.0
    cells = ge[:, :N_CAT] - ge[:, 1:]
    return (cells.cumsum(1) < rng.random((theta.size, 1))).sum(1)


def _three_group_data(n_per_group: int, shifts: dict[int, dict[int, float]], n_items: int):
    """``shifts[group][item]`` is that group's shift on that item."""
    rng = np.random.default_rng(SEED)
    total = 3 * n_per_group
    theta = rng.standard_normal(total)
    labels = np.repeat([10, 20, 30], n_per_group)
    columns = []
    for item in range(n_items):
        shift = np.array([shifts.get(int(g), {}).get(item, 0.0) for g in labels])
        columns.append(_graded_draw(shift, theta, rng))
    return np.column_stack(columns), labels


@pytest.fixture(scope="module")
def split_data():
    """Item 1 shifts only for group 20; item 4 only for group 30."""
    return _three_group_data(500, {20: {1: 1.8}, 30: {4: -1.8}}, 10)


def test_the_callers_own_group_labels_come_back(split_data) -> None:
    responses, labels = split_data

    result = dif_polytomous_anchor_sets(responses, labels, N_CAT, **REQUIRED)

    assert result["reference_group"] == 10
    assert result["focal_groups"] == [20, 30]
    assert set(result["per_group"]) == {20, 30}


def test_an_item_can_be_invariant_against_one_group_and_not_another(split_data) -> None:
    """The case the pooled sweep cannot express."""
    responses, labels = split_data

    result = dif_polytomous_anchor_sets(
        responses, labels, N_CAT, **dict(REQUIRED, min_anchor_items=3)
    )

    assert not result["per_group"][20]["anchor"][1]
    assert result["per_group"][30]["anchor"][1]
    assert not result["per_group"][30]["anchor"][4]
    assert result["per_group"][20]["anchor"][4]


def test_the_intersection_excludes_an_item_any_group_rejected(split_data) -> None:
    responses, labels = split_data

    result = dif_polytomous_anchor_sets(
        responses, labels, N_CAT, **dict(REQUIRED, min_anchor_items=3)
    )

    assert not result["intersection"][1]
    assert not result["intersection"][4]
    np.testing.assert_array_equal(
        result["intersection"], np.logical_and.reduce(result["anchor_by_group"], axis=0)
    )
    assert result["n_intersection"] == int(result["intersection"].sum())


def test_a_clean_bank_keeps_every_item_in_the_intersection() -> None:
    responses, labels = _three_group_data(400, {}, 5)

    result = dif_polytomous_anchor_sets(responses, labels, N_CAT, **REQUIRED)

    assert result["intersection"].all()
    assert result["intersection_trustworthy"]
    assert result["untrustworthy_groups"] == []


def test_a_failed_purification_marks_the_intersection_untrustworthy() -> None:
    """A smaller anchor must not look more conservative than it is."""
    responses, labels = _three_group_data(
        400, {20: {0: 1.7, 1: -1.7, 2: 1.7, 3: -1.7}}, 5
    )

    result = dif_polytomous_anchor_sets(
        # max_iter above the suite's usual 200: this fixture's simultaneous
        # four-item DIF makes the reference-vs-focal-20 two-group fit need
        # more EM rounds to converge than a well-behaved bank would.
        responses, labels, N_CAT,
        **dict(REQUIRED, min_anchor_items=5, max_iter=400),
    )

    assert not result["intersection_trustworthy"]
    assert 20 in result["untrustworthy_groups"]
    # Still computed, so a caller can inspect it.
    assert result["intersection"].dtype == bool


def test_the_reference_group_can_be_chosen(split_data) -> None:
    responses, labels = split_data

    result = dif_polytomous_anchor_sets(
        responses, labels, N_CAT, **dict(REQUIRED, reference_group=20)
    )

    assert result["reference_group"] == 20
    assert result["focal_groups"] == [10, 30]


def test_a_single_group_is_rejected(split_data) -> None:
    responses, labels = split_data
    with pytest.raises(ValueError, match="at least two distinct groups"):
        dif_polytomous_anchor_sets(
            responses, np.zeros(labels.size, dtype=int), N_CAT, **REQUIRED
        )


def test_a_mismatched_group_length_is_rejected_before_its_contents(split_data) -> None:
    """Shape is checked before content, so a short array reports the length."""
    responses, _ = split_data
    with pytest.raises(ValueError, match="one entry per person"):
        dif_polytomous_anchor_sets(
            responses, np.repeat([0, 1], 7), N_CAT, **REQUIRED
        )


def test_an_absent_reference_label_is_rejected(split_data) -> None:
    responses, labels = split_data
    with pytest.raises(ValueError, match="reference_group"):
        dif_polytomous_anchor_sets(
            responses, labels, N_CAT, **dict(REQUIRED, reference_group=15)
        )
