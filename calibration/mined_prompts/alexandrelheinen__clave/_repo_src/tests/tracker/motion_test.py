"""Estimating where an object is going, rather than assuming it.

An alpha-beta filter is the steady-state Kalman filter for a
constant-velocity model, so the claims worth pinning are the ones that
distinguish it from the model it replaced: that it learns a velocity nobody
configured, that it can be turned back into the old behaviour exactly, and
that it does not pretend to know more than a constant-velocity model can.
"""

from __future__ import annotations

import pytest

from clave.tracker.motion import Estimate, Gains, begin, fold

BELT = 0.30
STEP = 0.5


def observed(
    start: tuple[float, float],
    velocity: tuple[float, float],
    count: int,
    along: Gains,
    across: Gains,
) -> Estimate:
    """Feed a filter observations of an object moving at a constant velocity.

    Args:
        start: Where it begins.
        velocity: How it truly moves.
        count: How many observations to feed.
        along: Gains for the travel axis.
        across: Gains for the lateral axis.

    Returns:
        The estimate after the last observation.
    """
    estimate = begin(start, BELT, 0.0)
    for index in range(1, count + 1):
        at = index * STEP
        estimate = fold(
            estimate,
            (start[0] + velocity[0] * at, start[1] + velocity[1] * at),
            at,
            along,
            across,
        )
    return estimate


def test_an_estimate_opens_on_the_belt_and_assumes_no_drift() -> None:
    """AC-TRACK-31: an estimate opens on the belt and assumes no drift.

    The belt drives travel, so that velocity is known before any observation
    arrives. Nothing drives the lateral axis, so zero is the honest prior.
    """
    estimate = begin((-1.0, 0.05), BELT, 2.0)
    assert estimate.velocity == (BELT, 0.0)
    assert estimate.at(4.0) == pytest.approx((-1.0 + BELT * 2.0, 0.05))


def test_the_filter_learns_a_lateral_drift_nobody_configured() -> None:
    """AC-TRACK-32: the filter learns a lateral drift nobody configured.

    The whole point. The belt model says lateral velocity is zero; an object
    rolling has one, and only successive observations can say so.
    """
    estimate = observed(
        (-1.0, 0.0),
        (BELT, 0.04),
        count=8,
        along=Gains(0.8, 0.05),
        across=Gains(0.7, 0.20),
    )
    assert estimate.velocity[1] == pytest.approx(0.04, abs=0.008)


def test_replacing_outright_is_this_filter_at_alpha_one_and_beta_zero() -> None:
    """AC-TRACK-30: replacing outright is this filter at alpha one, beta zero.

    Worth pinning because it is what the tracker did before this existed, and
    it makes the change a tuning of one model rather than a swap of two.
    """
    replace = Gains(1.0, 0.0)
    estimate = observed(
        (-1.0, 0.0), (BELT, 0.04), count=5, along=replace, across=replace
    )
    assert estimate.position == pytest.approx((-1.0 + BELT * 2.5, 0.04 * 2.5))
    assert estimate.velocity == (BELT, 0.0)


def test_a_prediction_carries_the_learned_velocity_rather_than_the_belt() -> None:
    """AC-TRACK-32: a prediction carries the learned velocity, not the belt.

    The non-vacuous half of the test above: learning a velocity is worth
    nothing unless the propagation uses it.
    """
    estimate = observed(
        (-1.0, 0.0),
        (BELT, 0.04),
        count=8,
        along=Gains(0.8, 0.05),
        across=Gains(0.7, 0.20),
    )
    ahead = estimate.at(estimate.at_seconds + 2.0)
    assert ahead[1] - estimate.position[1] == pytest.approx(0.08, abs=0.016)


def test_the_filter_rejects_a_single_outlier_rather_than_taking_it() -> None:
    """AC-TRACK-33: the filter rejects a single outlier rather than taking it.

    Replacing outright moves the estimate the whole way to any reading,
    however wrong. A gain below one is what buys the difference, and it is
    the reason to filter position at all rather than only velocity.
    """
    gains = Gains(0.7, 0.20)
    settled = observed((-1.0, 0.0), (BELT, 0.0), count=6, along=gains, across=gains)
    jumped = fold(
        settled, (settled.position[0], 0.30), settled.at_seconds + STEP, gains, gains
    )
    assert abs(jumped.position[1] - 0.30) > 0.05, "the outlier was taken whole"


def test_two_readings_at_one_instant_say_nothing_about_a_velocity() -> None:
    """AC-TRACK-34: two readings at one instant say nothing about a velocity.

    The velocity correction divides by the elapsed time, so a second reading
    at the same instant would divide by zero.
    """
    gains = Gains(0.7, 0.20)
    estimate = begin((-1.0, 0.0), BELT, 1.0)
    again = fold(estimate, (-0.9, 0.02), 1.0, gains, gains)
    assert again.velocity == estimate.velocity
    assert again.at_seconds == pytest.approx(1.0)


def test_the_shipped_gains_load_and_lie_inside_the_unit_range() -> None:
    """AC-TRACK-30: the shipped gains load and lie inside the unit range."""
    from pathlib import Path

    from clave.tracker.fusion import FusionSettings

    root = Path(__file__).resolve().parents[2]
    settings = FusionSettings.load(
        root / "configs" / "perception" / "fusion.yml",
        root / "configs" / "world" / "sorting_line.yml",
    )
    for gains in (settings.along, settings.across):
        assert 0.0 <= gains.alpha <= 1.0
        assert 0.0 <= gains.beta <= 1.0
    # The travel axis is measured to be harmed by learning its velocity, so
    # it is deliberately the quieter of the two.
    assert settings.along.beta < settings.across.beta


def test_a_gain_outside_the_unit_range_is_refused() -> None:
    """AC-TRACK-30: a gain outside the unit range is refused."""
    import copy
    from pathlib import Path

    import yaml

    from clave.tracker.fusion import FusionError, FusionSettings

    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "perception" / "fusion.yml"
    raw = copy.deepcopy(yaml.safe_load(path.read_text()))
    raw["motion"]["across"]["beta"] = 1.4
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as handle:
        yaml.safe_dump(raw, handle)
        broken = Path(handle.name)
    with pytest.raises(FusionError, match="corrects past the observation"):
        FusionSettings.load(broken, root / "configs" / "world" / "sorting_line.yml")
