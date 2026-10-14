"""The belt frame and the clock, which everything else in the tracker rests on.



Every test here works from integer literals. That is the point of banning
`time.monotonic_ns` inside the package: propagation is provable by advancing a
number rather than by stepping a simulation.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from clave.tracker.belt_frame import (
    Footprint,
    FrameError,
    NadirOptics,
    elapsed_seconds,
    propagate,
)
from clave.world.config import load, require
from clave.world.scene import field_of_view

ROOT = Path(__file__).resolve().parents[2]

NANOS_PER_SECOND = 1_000_000_000


def a_footprint(x: float = 0.0, y: float = 0.0) -> Footprint:
    """A footprint on the belt surface, oriented along travel."""
    return Footprint(
        center=np.asarray((x, y, 0.93), dtype=np.float64),
        major_extent=0.10,
        minor_extent=0.06,
        yaw=0.0,
    )


def test_elapsed_seconds_converts_two_monotonic_instants() -> None:
    """The clock is nanoseconds and the arithmetic is seconds."""
    assert elapsed_seconds(1_000, 1_000 + NANOS_PER_SECOND // 2) == pytest.approx(0.5)


def test_elapsed_seconds_is_negative_when_the_observation_is_older() -> None:
    """Out of order is reported rather than hidden.

    Fusion clamps this at zero; the clock does not, because a caller that
    cannot tell the difference cannot weight it.
    """
    assert elapsed_seconds(2 * NANOS_PER_SECOND, NANOS_PER_SECOND) == pytest.approx(
        -1.0
    )


def test_propagating_an_observation_moves_it_along_belt_travel() -> None:
    """Move the clock, never the object."""
    observed = a_footprint(x=-1.00)
    carried = propagate(
        observed,
        belt_speed=0.25,
        observed_at_nanos=0,
        to_nanos=4 * NANOS_PER_SECOND,
    )
    assert carried.center[0] == pytest.approx(0.00)


def test_propagation_changes_nothing_across_the_belt_or_above_it() -> None:
    """Travel is along x alone."""
    observed = a_footprint(x=-0.50, y=0.21)
    carried = propagate(
        observed, belt_speed=0.31, observed_at_nanos=0, to_nanos=NANOS_PER_SECOND
    )
    assert carried.center[1] == pytest.approx(observed.center[1])
    assert carried.center[2] == pytest.approx(observed.center[2])
    assert carried.major_extent == pytest.approx(observed.major_extent)
    assert carried.yaw == pytest.approx(observed.yaw)


def test_propagating_to_the_instant_it_was_observed_returns_it_unmoved() -> None:
    """Zero elapsed is not a special case, it is zero travel."""
    observed = a_footprint(x=0.4)
    assert propagate(
        observed, belt_speed=0.31, observed_at_nanos=77, to_nanos=77
    ).center == pytest.approx(observed.center)


def test_propagating_backwards_carries_the_observation_upstream() -> None:
    """The belt has a direction and the arithmetic respects it."""
    carried = propagate(
        a_footprint(x=0.0),
        belt_speed=0.25,
        observed_at_nanos=4 * NANOS_PER_SECOND,
        to_nanos=0,
    )
    assert carried.center[0] == pytest.approx(-1.00)


def test_a_negative_belt_speed_is_refused_naming_itself() -> None:
    """A belt that runs backwards is a configuration fault."""
    with pytest.raises(FrameError, match="belt speed"):
        propagate(a_footprint(), belt_speed=-0.25, observed_at_nanos=0, to_nanos=1)


def test_a_footprint_refuses_a_minor_extent_wider_than_its_major() -> None:
    """The axes are named, so they cannot be swapped silently."""
    with pytest.raises(FrameError, match="minor extent"):
        Footprint(
            center=np.asarray((0.0, 0.0, 0.93), dtype=np.float64),
            major_extent=0.05,
            minor_extent=0.09,
            yaw=0.0,
        )


def test_a_footprint_that_declines_to_state_a_yaw_says_so() -> None:
    """A footprint that declines to state a yaw says so.

    A can is circular in plan under a nadir camera. A confident random yaw on a
    circular footprint is worse than an absent one, because the safety layer
    acts on it.
    """
    circular = Footprint(
        center=np.asarray((0.0, 0.0, 0.93), dtype=np.float64),
        major_extent=0.07,
        minor_extent=0.07,
        yaw=0.0,
        oriented=False,
    )
    assert not circular.oriented


def test_the_nadir_scale_factor_matches_the_configured_detection_camera() -> None:
    """The nadir scale factor matches the configured detection camera.

    The optics are read from the shipped configuration rather than restated
    here, so this fails when the line is re-lensed and the figure in
    docs/measurements.md is not.

    An earlier version of this test hardcoded the camera height and the field
    of view and claimed in its own docstring to be derived from configuration.
    It passed unchanged through a change of sensor, lens, standoff and mounting
    angle, which is what a test asserting a number nobody reads does.
    """
    raw = load(ROOT / "configs" / "world" / "sorting_line.yml")
    surface = float(require(require(raw, "belt"), "surface_height_meters", "belt"))
    wide = next(c for c in require(raw, "cameras") if str(c["role"]) == "detection")
    pixels = max(require(require(raw, "sensors")[str(wide["sensor"])], "pixels"))

    optics = NadirOptics(
        camera_height=float(wide["position_meters"][2]),
        fovy_degrees=field_of_view(
            wide, require(raw, "sensors"), require(raw, "lenses")
        ),
    )
    measured = optics.meters_per_pixel(surface_height=surface, render_height=pixels)
    assert measured * 1000.0 == pytest.approx(0.449, abs=5e-4)


def test_the_scale_factor_is_taken_at_the_object_and_not_at_the_belt() -> None:
    """The scale factor is taken at the object and not at the belt.

    This is the whole reason the criterion exists. Using the belt plane for a
    0.10 m object inflates its footprint by 13.3 percent, and that error reaches
    grasp_width and the mass band.
    """
    # Literal optics on purpose: this is the arithmetic, not the shipped line.
    optics = NadirOptics(camera_height=1.750, fovy_degrees=45.0)
    at_belt = optics.meters_per_pixel(surface_height=0.90, render_height=1080)
    at_object = optics.meters_per_pixel(surface_height=1.00, render_height=1080)
    assert at_object < at_belt
    assert at_belt / at_object - 1.0 == pytest.approx(0.133, abs=0.002)


def test_the_scale_factor_follows_the_render_size_not_the_declared_resolution() -> None:
    """The scale factor follows the render size not the declared resolution.

    configs/data/recording.yml renders 320 by 240 against cameras declaring
    1920 by 1080. A factor fitted to one and applied to the other is wrong by
    the ratio, so the render height is an argument rather than read from the
    sensor.
    """
    optics = NadirOptics(camera_height=1.750, fovy_degrees=45.0)
    coarse = optics.meters_per_pixel(surface_height=0.90, render_height=240)
    fine = optics.meters_per_pixel(surface_height=0.90, render_height=1080)
    assert coarse / fine == pytest.approx(1080 / 240)


def test_a_camera_below_the_surface_it_looks_at_is_refused() -> None:
    """A negative standoff yields a negative scale factor."""
    optics = NadirOptics(camera_height=0.80, fovy_degrees=45.0)
    with pytest.raises(FrameError, match="standoff"):
        optics.meters_per_pixel(surface_height=0.90, render_height=1080)


def test_the_tracker_package_reads_no_clock_of_its_own() -> None:
    """The tracker package reads no clock of its own.

    Every instant is an argument. A module that reaches for the clock makes
    propagation untestable from literals and makes a replay irreproducible.

    The check is for a call rather than for a word, because these modules
    discuss monotonic instants constantly in their documentation and an
    assertion that fires on prose is an assertion nobody keeps.
    """
    package = Path(__file__).resolve().parents[2] / "src" / "clave" / "tracker"
    reaches_for_a_clock = re.compile(
        r"^\s*(?:import\s+time\b|from\s+time\s+import)|"
        r"\btime\.(?:monotonic|monotonic_ns|time|perf_counter)\s*\(",
        re.MULTILINE,
    )
    offenders = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if path.name != "debug_run.py" and reaches_for_a_clock.search(path.read_text())
    )
    assert offenders == []


def test_the_footprint_carries_a_belt_frame_point_not_a_pixel() -> None:
    """The footprint carries a belt frame point not a pixel.

    The belt frame has its origin at the belt centre with z from the floor, so
    a footprint resting on the belt sits near 0.90 and never near zero.
    """
    resting = a_footprint()
    assert resting.center[2] > 0.5
    assert math.isfinite(resting.center[0])
