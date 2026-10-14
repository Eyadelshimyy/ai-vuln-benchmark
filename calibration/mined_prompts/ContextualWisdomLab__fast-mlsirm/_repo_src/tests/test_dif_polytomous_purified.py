"""Contract for :func:`dif_polytomous_purified`.

The value of purification is that an item with differential functioning stops
being part of the criterion every other item is judged against. The tests are
written around that, plus the two places this loop deliberately differs from
its dichotomous counterparts.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import dif_polytomous, dif_polytomous_purified

N_CAT = 4
SEED = 20_260_915
THRESHOLDS = np.array([1.2, 0.0, -1.2])
# Issue #1958: model/q_theta/max_iter/tol/fdr_q/max_rounds/min_anchor_items are
# all required caller arguments now (no unsourced defaults) -- these are the
# test suite's own arbitrary but explicit choices, not package defaults.
REQUIRED = dict(
    model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05,
    max_rounds=3, min_anchor_items=4,
)


def _graded_draw(slope, shift, theta, rng):
    ge = np.ones((theta.size, N_CAT + 1))
    for k in range(1, N_CAT):
        ge[:, k] = 1.0 / (1.0 + np.exp(-(slope * theta + THRESHOLDS[k - 1] + shift)))
    ge[:, N_CAT] = 0.0
    cells = ge[:, :N_CAT] - ge[:, 1:]
    return (cells.cumsum(1) < rng.random((theta.size, 1))).sum(1)


def _two_group_data(n_per_group: int, dif_items: dict[int, float], n_items: int):
    """Both groups share the trait distribution; listed items shift for group 1."""
    rng = np.random.default_rng(SEED)
    theta = rng.standard_normal(2 * n_per_group)
    group = np.repeat([0, 1], n_per_group)
    columns = []
    for item in range(n_items):
        shift = np.where(group == 1, dif_items.get(item, 0.0), 0.0)
        columns.append(_graded_draw(1.1, shift, theta, rng))
    return np.column_stack(columns), group


@pytest.fixture(scope="module")
def clean_data():
    return _two_group_data(600, {}, 8)


@pytest.fixture(scope="module")
def contaminated_data():
    return _two_group_data(600, {2: 1.4, 5: -1.3}, 8)


def test_a_clean_bank_keeps_every_item_in_the_anchor(clean_data) -> None:
    responses, group = clean_data

    report = dif_polytomous_purified(responses, group, N_CAT, **REQUIRED)

    assert report["n_anchor"] == responses.shape[1]
    assert report["anchor"].all()
    assert report["purify_converged"]
    assert report["purify_termination_reason"] == "stable_flag_set"
    assert report["rounds"] == 0


def test_items_with_differential_functioning_leave_the_anchor(contaminated_data) -> None:
    responses, group = contaminated_data

    report = dif_polytomous_purified(responses, group, N_CAT, **REQUIRED)

    assert not report["anchor"][2]
    assert not report["anchor"][5]
    assert report["n_anchor"] == responses.shape[1] - 2


def test_the_unpurified_sweep_returns_no_anchor_at_all(contaminated_data) -> None:
    """The gap this function exists to close."""
    responses, group = contaminated_data

    plain = dif_polytomous(
        responses, group, N_CAT,
        model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05,
    )

    assert "anchor" not in plain
    assert "purify_termination_reason" not in plain


def test_the_anchor_floor_stops_the_loop_and_says_so() -> None:
    """Termination must be reported, not inferred from a small anchor."""
    responses, group = _two_group_data(600, {0: 1.5, 1: -1.5, 2: 1.5, 3: -1.5}, 6)

    kwargs = dict(REQUIRED, min_anchor_items=5)
    report = dif_polytomous_purified(responses, group, N_CAT, **kwargs)

    assert report["purify_termination_reason"] == "insufficient_anchor_items"
    assert not report["purify_converged"]
    # The screened set is reported even though it is below the floor. Returning
    # the previous, larger anchor would make a failed purification look like a
    # clean bank -- the most permissive answer, delivered silently.
    assert report["n_anchor"] < 5
    assert not report["anchor"].all()


def test_the_plain_sweep_fields_are_all_still_returned(clean_data) -> None:
    responses, group = clean_data

    report = dif_polytomous_purified(responses, group, N_CAT, **REQUIRED)

    for key in ("item", "lr", "df", "p_value", "flagged_bh", "effect_size"):
        assert key in report
        assert len(report[key]) == responses.shape[1]


def test_zero_rounds_reduces_to_the_unpurified_sweep(contaminated_data) -> None:
    """``max_rounds=0`` must do the initial sweep and stop, not skip it."""
    responses, group = contaminated_data

    kwargs = dict(REQUIRED, max_rounds=0)
    report = dif_polytomous_purified(responses, group, N_CAT, **kwargs)

    assert report["rounds"] == 0
    assert report["anchor"].all()
    assert np.isfinite(report["p_value"]).any()


def test_missing_q_theta_or_model_raises(clean_data) -> None:
    """Issue #1958: no unsourced default node count or model selection."""
    responses, group = clean_data
    without_q_theta = {k: v for k, v in REQUIRED.items() if k != "q_theta"}
    with pytest.raises(TypeError):
        dif_polytomous_purified(responses, group, N_CAT, **without_q_theta)
    without_model = {k: v for k, v in REQUIRED.items() if k != "model"}
    with pytest.raises(TypeError):
        dif_polytomous_purified(responses, group, N_CAT, **without_model)
