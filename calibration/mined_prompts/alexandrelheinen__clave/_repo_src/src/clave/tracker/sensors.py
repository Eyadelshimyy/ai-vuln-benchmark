"""The line's sensors, read from configuration and found only by role.

The perception contract promises that adding, moving or removing a camera edits
`configs/world/sorting_line.yml` and nothing else. That promise holds exactly as
long as nothing reaches for a camera by its id, so this module offers no lookup
that takes one. `source_id` survives on a `SensorSpec` for provenance and for
debugging, and no consumer branches on it.

A camera names the sensor and the lens it is built from, and the optics follow
from those. Nothing here restates an angle, because an angle written twice is an
angle that disagrees with itself the first time a lens changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError
from clave.tracker.belt_frame import NadirOptics
from clave.tracker.evidence import Role
from clave.world.config import require
from clave.world.scene import field_of_view


class SensorError(ClaveError):
    """The configured sensors do not describe a line this can observe."""


@dataclass(frozen=True)
class SensorSpec:
    """One installed sensor, as the tracker needs to see it.

    Attributes:
        source_id: What the line calls it. Provenance only: no fusion rule
            accepts it and no lookup here takes it.
        role: What kind of thing it reports, which is the only key a consumer
            may select on.
        position: Where it stands, in belt frame meters.
        optics: What its pixels are worth on the belt.
        pixels: The sensor's native resolution, which is not the size anything
            necessarily renders at. The two differ, and the difference is why
            the projection takes a render height rather than reading this.
    """

    source_id: str
    role: Role
    position: NDArray[np.float64]
    optics: NadirOptics
    pixels: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SensorSpec):
            return NotImplemented
        return (
            self.source_id == other.source_id
            and self.role == other.role
            and bool(np.allclose(self.position, other.position, rtol=0.0, atol=1e-12))
            and self.optics == other.optics
            and self.pixels == other.pixels
        )


def load_sensors(raw: dict[str, object]) -> tuple[SensorSpec, ...]:
    """Read every declared camera into a sensor.

    Args:
        raw: The loaded world configuration.

    Returns:
        One entry per camera, in the order the configuration declares them.

    Raises:
        SensorError: If a camera names a role the evidence union does not
            define, or two cameras share one `source_id`. Ambiguous provenance
            is worse than none, because a reader who cannot tell two sensors
            apart will assume they are one.
    """
    catalogs = (require(raw, "sensors"), require(raw, "lenses"))
    roles = {role.value: role for role in Role}

    sensors: list[SensorSpec] = []
    seen: set[str] = set()
    for entry in require(raw, "cameras"):
        source_id = str(require(entry, "id", "cameras"))
        if source_id in seen:
            raise SensorError(
                f"two cameras are declared as {source_id!r}, so nothing can "
                f"tell their evidence apart"
            )
        seen.add(source_id)

        named = str(require(entry, "role", "cameras"))
        if named not in roles:
            raise SensorError(
                f"camera {source_id!r} reports role {named!r}, which is not "
                f"one of {', '.join(sorted(roles))}"
            )
        position = [
            float(value) for value in require(entry, "position_meters", "cameras")
        ]
        pixels = require(
            catalogs[0][str(require(entry, "sensor", "cameras"))], "pixels"
        )
        sensors.append(
            SensorSpec(
                source_id=source_id,
                role=roles[named],
                position=np.asarray(
                    (position[0], position[1], position[2]), dtype=np.float64
                ),
                optics=NadirOptics(
                    camera_height=position[2],
                    fovy_degrees=field_of_view(entry, *catalogs),
                ),
                pixels=(int(pixels[0]), int(pixels[1])),
            )
        )
    return tuple(sensors)


def of_role(sensors: tuple[SensorSpec, ...], role: Role) -> tuple[SensorSpec, ...]:
    """Return every sensor reporting one role, which may be none.

    Args:
        sensors: The installed sensors.
        role: What kind of reading is wanted.

    Returns:
        The matching sensors, in declaration order. An empty result is an
        ordinary answer: a line without a depth sensor has no `DEPTH` role, and
        a caller that needs one says so by calling `require_role`.
    """
    return tuple(sensor for sensor in sensors if sensor.role is role)


def require_role(sensors: tuple[SensorSpec, ...], role: Role) -> SensorSpec:
    """Return the sensor reporting one role, failing when none does.

    Args:
        sensors: The installed sensors.
        role: The role that has to be produced.

    Returns:
        The first sensor reporting it.

    Raises:
        SensorError: If nothing produces the role. A tracker configured to fuse
            a role no camera produces would emit records silently missing that
            evidence, which is worse than refusing to start.
    """
    found = of_role(sensors, role)
    if not found:
        raise SensorError(
            f"no camera produces the {role.value!r} role, so nothing would ever fuse it"
        )
    return found[0]
