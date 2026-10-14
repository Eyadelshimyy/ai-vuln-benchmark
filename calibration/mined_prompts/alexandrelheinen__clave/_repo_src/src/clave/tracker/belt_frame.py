"""The belt frame and the clock, which every other part of the tracker rests on.

Two invariants make the perception contract work, and both live here.

**One frame.** Every observation is expressed in the belt frame: `x` along belt
travel, `y` across it, `z` measured from the floor, origin at the centre of the
belt on the centreline, under the arm. A 3.00 m belt runs from -1.50 m to
+1.50 m and its surface sits at 0.90 m, so a footprint resting on the belt
carries a `z` near 0.90 rather than near zero.

**One clock.** Every instant arrives as an argument. Nothing in this package
reads a clock of its own, which is what lets propagation be proved by advancing
an integer rather than by stepping a simulation, and what makes a replay of the
same observations produce the same answer twice.

The projection from pixels to metres is a scale factor rather than a homography
because every camera looks straight down. It is not a constant: the scale is set
by the height of the surface being imaged, so a tall object resolves finer than
the belt it stands on, and using the belt plane for both inflates the object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError

NANOS_PER_SECOND = 1_000_000_000
"""Nanoseconds in a second, for the monotonic instants an observation carries."""


class FrameError(ClaveError):
    """A frame quantity describes no geometry the belt could hold."""


@dataclass(frozen=True)
class Footprint:
    """An oriented box on the belt, in belt frame meters.

    Attributes:
        center_belt: Where the box sits, in belt frame meters.
        major_extent: The longer horizontal side, in meters.
        minor_extent: The shorter horizontal side, in meters. This is what a
            jaw would have to open to, which is why the two are named rather
            than ordered by axis.
        yaw_belt: Rotation of the major axis about the belt normal, in radians.
        oriented: Whether the yaw means anything. An object that is circular in
            plan has no major axis to find, and a confident random yaw on a
            circular footprint is worse than an absent one because the safety
            layer acts on it.
    """

    center_belt: NDArray[np.float64]
    major_extent: float
    minor_extent: float
    yaw_belt: float
    oriented: bool = True

    def __init__(
        self,
        center_belt: NDArray[np.float64] | None = None,
        major_extent: float = 0.0,
        minor_extent: float = 0.0,
        yaw_belt: float | None = None,
        oriented: bool = True,
        *,
        center: NDArray[np.float64] | None = None,
        yaw: float | None = None,
    ) -> None:
        c = center_belt if center_belt is not None else center
        if c is None:
            raise TypeError("Footprint requires center_belt or center")
        y = yaw_belt if yaw_belt is not None else (0.0 if yaw is None else yaw)
        object.__setattr__(self, "center_belt", np.asarray(c, dtype=np.float64))
        object.__setattr__(self, "major_extent", major_extent)
        object.__setattr__(self, "minor_extent", minor_extent)
        object.__setattr__(self, "yaw_belt", y)
        object.__setattr__(self, "oriented", oriented)
        self.__post_init__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Footprint):
            return NotImplemented
        return (
            bool(np.allclose(self.center_belt, other.center_belt, rtol=0.0, atol=1e-12))
            and self.major_extent == other.major_extent
            and self.minor_extent == other.minor_extent
            and self.yaw_belt == other.yaw_belt
            and self.oriented == other.oriented
        )

    @property
    def center(self) -> NDArray[np.float64]:
        """Backwards compatibility alias for center_belt."""
        return self.center_belt

    @property
    def yaw(self) -> float:
        """Backwards compatibility alias for yaw_belt."""
        return self.yaw_belt

    def __post_init__(self) -> None:
        """Refuse a box that names its axes the wrong way round.

        Raises:
            FrameError: If either extent is not positive and finite, or if the
                minor extent is the wider of the two.
        """
        for name, extent in (
            ("major extent", self.major_extent),
            ("minor extent", self.minor_extent),
        ):
            if not math.isfinite(extent) or extent <= 0.0:
                raise FrameError(f"{name} is {extent!r}, which describes no box")
        if self.minor_extent > self.major_extent:
            raise FrameError(
                f"minor extent {self.minor_extent} exceeds major extent "
                f"{self.major_extent}, so the axes are the wrong way round"
            )


def elapsed_seconds(earlier_nanos: int, later_nanos: int) -> float:
    """Return the seconds between two monotonic instants.

    The result is negative when the second instant precedes the first. That is
    reported rather than hidden, because a caller that cannot tell an
    out-of-order observation from a fresh one cannot weight it correctly.
    Fusion clamps the value; the clock does not.

    Args:
        earlier_nanos: The instant measured from.
        later_nanos: The instant measured to.

    Returns:
        The interval in seconds.
    """
    return (later_nanos - earlier_nanos) / NANOS_PER_SECOND


def carry(
    position_belt: NDArray[np.float64] | None = None,
    belt_speed: float = 0.0,
    observed_at_nanos: int = 0,
    to_nanos: int = 0,
    *,
    point: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Carry one point along the belt to a later instant.

    The belt drives `x` and leaves everything else to physics, and this is
    the one place that says so. [propagate] carries a whole footprint and
    reads its travel from here, so a consumer that only has a position does
    not grow a second definition of which way the belt runs.

    Args:
        position_belt: Where it was, in belt frame meters.
        belt_speed: Belt speed in meters per second.
        observed_at_nanos: When it was there.
        to_nanos: The instant to carry it to. May precede the observation, in
            which case the point is carried upstream.
        point: Legacy keyword alias for position_belt.

    Returns:
        Where it is at `to_nanos`.

    Raises:
        FrameError: If the belt speed is negative or not finite.
    """
    p = position_belt if position_belt is not None else point
    if p is None:
        raise TypeError("carry requires position_belt or point")
    if not math.isfinite(belt_speed) or belt_speed < 0.0:
        raise FrameError(
            f"belt speed is {belt_speed!r}; a belt that runs backwards or "
            f"nowhere describes no line this can propagate along"
        )
    travel = belt_speed * elapsed_seconds(observed_at_nanos, to_nanos)
    result = np.asarray(p, dtype=np.float64).copy()
    result[0] += travel
    return result


def propagate(
    footprint: Footprint,
    belt_speed: float,
    observed_at_nanos: int,
    to_nanos: int,
) -> Footprint:
    """Carry an observation along the belt to a later instant.

    This is what lets a barcode read at the gate attach to an object picked a
    metre downstream, and what lets two cameras firing 8 ms apart describe the
    same object. Only the coordinate along travel changes: the belt drives `x`
    and leaves everything else to physics.

    Args:
        footprint: The observation, as it was seen.
        belt_speed: Belt speed in meters per second.
        observed_at_nanos: When the observation was taken.
        to_nanos: The instant to carry it to. May precede the observation, in
            which case the footprint is carried upstream.

    Returns:
        The footprint at `to_nanos`, with its extents and yaw unchanged.

    Raises:
        FrameError: If the belt speed is negative or not finite.
    """
    return Footprint(
        center_belt=carry(
            footprint.center_belt, belt_speed, observed_at_nanos, to_nanos
        ),
        major_extent=footprint.major_extent,
        minor_extent=footprint.minor_extent,
        yaw_belt=footprint.yaw_belt,
        oriented=footprint.oriented,
    )


@dataclass(frozen=True)
class NadirOptics:
    """What one downward-looking camera's pixels are worth on the belt.

    A nadir view keeps the image plane parallel to the belt, so pixel to world
    is a scale factor rather than a homography that varies across the frame.

    Attributes:
        camera_height: Where the camera sits, in belt frame meters, which is
            measured from the floor like every other `z` in this frame.
        fovy_degrees: The vertical field of view, which is what MuJoCo's `fovy`
            means. It sets the image height axis, and for these cameras that
            axis lies across the belt rather than along it.
    """

    camera_height: float
    fovy_degrees: float

    def meters_per_pixel(self, surface_height: float, render_height: int) -> float:
        """Return the belt-frame meters one pixel spans at a given height.

        The height is the surface being imaged, not the belt. An object 0.10 m
        tall is 0.10 m closer to the camera than the belt it rests on, so it
        resolves about 13 percent finer, and using the belt-plane factor for it
        inflates its footprint by the same amount. That error reaches the grasp
        width and the mass band, which is why this takes a height rather than
        assuming one.

        The render height is an argument rather than the sensor's declared
        resolution, because the two differ: the cameras declare an IMX264 at
        2448 by 2048 and `configs/data/recording.yml` renders 320 by 240. A
        factor fitted to one and applied to the other is wrong by the ratio,
        and that gap is not academic. Measured at 1080 pixels the shipped code
        cameras decode no barcode at all; measured at their own 2448 they do.

        Args:
            surface_height: Height of the imaged surface, in belt frame meters.
            render_height: Height of the rendered frame, in pixels.

        Returns:
            Meters per pixel at that surface.

        Raises:
            FrameError: If the camera does not stand above the surface, or the
                render has no height.
        """
        if render_height <= 0:
            raise FrameError(f"render height is {render_height!r}, so it has no pixels")
        standoff = self.camera_height - surface_height
        if not math.isfinite(standoff) or standoff <= 0.0:
            raise FrameError(
                f"standoff is {standoff!r}: a camera at {self.camera_height} "
                f"does not look down on a surface at {surface_height}"
            )
        span = 2.0 * standoff * math.tan(math.radians(self.fovy_degrees) / 2.0)
        return span / render_height
