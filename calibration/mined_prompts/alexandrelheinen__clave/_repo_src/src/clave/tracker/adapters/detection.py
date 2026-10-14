"""Turning an instance mask into a footprint on the belt.

The conversion is a scale factor rather than a homography, because every camera
looks straight down, but it is not a constant. The scale is set by the height of
the surface being imaged, so a tall object resolves finer than the belt it
stands on and scaling both at the belt plane inflates the object.

Two axis facts were measured rather than reasoned about, by placing an object at
a known position and reading back its bounds from a segmentation render:

- The column axis runs along belt travel, so `x = camera_x + (column - width/2)`
  scaled. A 0.20 m move downstream shifted the bounds 96 columns at 640 by 480.
- The row axis runs across the belt and is inverted, because row zero is the top
  of the image and that is the far side. Getting this backwards mirrors every
  footprint about the centreline, and nothing downstream would say so.

**The key the render is indexed by is thrown away here.** A MuJoCo segmentation
render is keyed by geometry name, that name is `object_<slot>`, and
`clave.world.belt` makes the slot the same integer the world hands out as
`ObjectLabel.object_id`. An adapter that passed it through would hand the
tracker the answer while appearing to perceive it, which is exactly the trap
this adapter exists to close.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError
from clave.tracker.belt_frame import Footprint, FrameError, NadirOptics
from clave.tracker.evidence import Detection, Evidence, PixelMask, Role

ROUNDNESS = 1.15
"""How elongated a mask has to be before it states a yaw.

Below this ratio between the two principal extents the shape is round enough
that its major axis is noise, and a confident random yaw on a circular footprint
is worse than an absent one because the safety layer acts on it.
"""


class DetectionError(ClaveError):
    """A mask cannot be placed on the belt as it stands."""


def to_belt(
    column: float,
    row: float,
    camera: NDArray[np.float64],
    optics: NadirOptics,
    surface_height: float,
    render: tuple[int, int],
) -> tuple[float, float]:
    """Return where one pixel falls on the belt, in belt frame meters.

    Args:
        column: Pixel column, from the left of the frame.
        row: Pixel row, from the top of the frame.
        camera: Where the camera stands, in belt frame meters.
        optics: What its pixels are worth.
        surface_height: Height of the surface being imaged.
        render: The rendered frame size as `(width, height)` in pixels.

    Returns:
        The point as `(x, y)` in belt frame meters.

    Raises:
        DetectionError: If the camera does not stand above the surface.
    """
    width, height = render
    try:
        metres = optics.meters_per_pixel(surface_height, height)
    except FrameError as error:
        raise DetectionError(str(error)) from error
    return (
        float(camera[0]) + (column - width / 2.0) * metres,
        float(camera[1]) - (row - height / 2.0) * metres,
    )


def oriented_extent(mask: PixelMask) -> tuple[float, float, float, bool]:
    """Return the mask's principal extents and orientation, in pixels.

    Second moments rather than a bounding box, so a package lying at an angle
    reports the box it actually occupies rather than the axis-aligned box that
    contains it.

    Args:
        mask: The pixels the instance covers.

    Returns:
        The major extent, the minor extent, the yaw in radians, and whether the
        yaw means anything. A shape round enough that its major axis is noise
        declines to state one.
    """
    pixels = [
        (column + offset, row)
        for row, column, length in mask.runs
        for offset in range(length)
    ]
    count = len(pixels)
    mean_x = math.fsum(x for x, _ in pixels) / count
    mean_y = math.fsum(y for _, y in pixels) / count

    xx = math.fsum((x - mean_x) ** 2 for x, _ in pixels) / count
    yy = math.fsum((y - mean_y) ** 2 for _, y in pixels) / count
    xy = math.fsum((x - mean_x) * (y - mean_y) for x, y in pixels) / count

    common = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
    larger = (xx + yy + common) / 2.0
    smaller = (xx + yy - common) / 2.0

    # Twice the standard deviation each way spans a uniform box of that extent
    # to within a constant, and the constant cancels in every ratio below.
    major = 2.0 * math.sqrt(max(larger, 0.0)) * math.sqrt(3.0)
    minor = 2.0 * math.sqrt(max(smaller, 0.0)) * math.sqrt(3.0)
    oriented = minor > 0.0 and major / minor >= ROUNDNESS
    yaw = 0.5 * math.atan2(2.0 * xy, xx - yy) if oriented else 0.0
    return major, max(minor, 1.0), yaw, oriented


def detections_from_masks(
    masks: Mapping[str, PixelMask],
    source_id: str,
    observed_at_nanos: int,
    camera: NDArray[np.float64],
    optics: NadirOptics,
    surface_height: float,
    render: tuple[int, int],
    belt_surface: float | None = None,
) -> tuple[Evidence, ...]:
    """Turn every instance mask in one frame into a reading.

    The mapping's keys are read to iterate and are never carried into a result.
    The render's key is the simulator's object id, and a reading carrying it
    would be a reading carrying the answer.

    Args:
        masks: Instance masks by whatever the renderer keyed them under.
        source_id: Which sensor produced the frame.
        observed_at_nanos: When the frame was taken.
        camera: Where the camera stands, in belt frame meters.
        optics: What its pixels are worth.
        surface_height: Height of the surface being imaged, which is the
            object's upper surface rather than the belt. Scaling both at the
            belt plane inflates a 0.10 m object's footprint by 13.3 percent,
            and that error reaches the grasp width and the mass band.
        render: The rendered frame size as `(width, height)` in pixels.
        belt_surface: Where the belt surface sits, in belt frame meters, so the
            object's height above it is a subtraction rather than a guess.
            Defaults to `surface_height`, meaning an object of no known height.

    Returns:
        One reading per mask, ordered by where each sits along belt travel so
        the result does not depend on the mapping's iteration order.

    Raises:
        DetectionError: If the camera does not stand above the surface.
    """
    metres = _metres_per_pixel(optics, surface_height, render[1])
    readings = []
    for mask in masks.values():
        major, minor, yaw, oriented = oriented_extent(mask)
        if major * metres <= 0.0 or minor * metres <= 0.0:
            # A sliver at the very edge of the frame segments to a mask one
            # pixel across, whose projected extent rounds to nothing. A
            # footprint refuses to be built from that, and refusing is
            # right: a detection of no size is not a detection. Dropping it
            # here is what keeps one pixel from ending a run, which it did.
            continue
        left, top, right, bottom = mask.bounds
        x, y = to_belt(
            (left + right) / 2.0,
            (top + bottom) / 2.0,
            camera,
            optics,
            surface_height,
            render,
        )
        readings.append(
            Evidence(
                source_id=source_id,
                role=Role.DETECTION,
                observed_at_nanos=observed_at_nanos,
                confidence=_confidence_of(mask),
                payload=Detection(
                    footprint=Footprint(
                        center=np.asarray((x, y, surface_height), dtype=np.float64),
                        major_extent=major * metres,
                        minor_extent=minor * metres,
                        # The row axis is inverted, so a yaw measured in image
                        # space turns the other way on the belt.
                        yaw=-yaw,
                        oriented=oriented,
                    ),
                    mask=mask,
                    height=surface_height
                    - (surface_height if belt_surface is None else belt_surface),
                ),
            )
        )
    # Ordered along belt travel so the result does not inherit the mapping's
    # iteration order, which would make two identical frames differ.
    return tuple(sorted(readings, key=_along_travel))


def _along_travel(reading: Evidence) -> float:
    """Return how far along the belt one detection sits."""
    payload = reading.payload
    assert isinstance(payload, Detection)
    return float(payload.footprint.center[0])


def _metres_per_pixel(
    optics: NadirOptics, surface_height: float, render_height: int
) -> float:
    """Return the scale factor, reporting a camera below its subject."""
    try:
        return optics.meters_per_pixel(surface_height, render_height)
    except FrameError as error:
        raise DetectionError(str(error)) from error


def _confidence_of(mask: PixelMask) -> float:
    """Return how sure the adapter is that this mask is one object.

    A larger instance is a more certain one: a mask of a handful of pixels is as
    likely to be a speck of belt as a package. The curve saturates quickly, so
    anything over a few hundred pixels is reported at full confidence. The
    reasoning is written down here rather than hidden behind a returned
    constant.
    """
    return min(1.0, mask.pixel_count / 400.0)
