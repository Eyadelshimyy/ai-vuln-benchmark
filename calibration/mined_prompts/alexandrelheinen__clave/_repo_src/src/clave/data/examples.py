"""Labeled examples, as the world produces them.

Labels come from the simulator rather than from a labeling pass. The world
already knows each object's material class, so a synthetic example is labeled by
construction and the usual source of label noise does not arise.

What that does not give is appearance. These frames show parametric primitives,
so a classifier trained only on them learns shape rather than material.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError
from clave.taxonomy import BY_ID


class LabelError(ClaveError):
    """A label names a material class the taxonomy does not define."""


class Origin(Enum):
    """Where an example came from.

    Kept on every example so a held-out set of real imagery can be excluded from
    training without re-deriving which examples are which.
    """

    SIMULATED = "simulated"
    REAL = "real"


@dataclass(frozen=True)
class ObjectLabel:
    """One object as the world knows it at capture time.

    Attributes:
        object_id: Identity, stable across the frames of one rollout.
        material_class: Taxonomy identifier of the form `M-NN`.
        channel: Default channel for that class.
        position: Object position in meters, in world coordinates.
        in_reachable_window: Whether the arm could have reached it. An object
            outside the window cannot be picked, so training on it as a pick
            target teaches a false association.
        bbox: Pixel bounds as (x_min, y_min, x_max, y_max), or None when the
            object is not visible in the frame. The camera sees roughly half the
            belt, so an object on the belt is frequently absent from the image,
            and a detector trained on a label with no pixels would be taught to
            hallucinate. None is therefore the visibility flag as well as the
            absence of a box.
        orientation: Unit quaternion (w, x, y, z) in world coordinates, or None
            on a record written before the corpus stored pose.
        linear_velocity: Meters per second, world frame, or None when unstored.
        angular_velocity: Radians per second, world frame, or None when unstored.
        object_name: Catalog name of the spawned mesh. Empty when unstored.
    """

    object_id: int
    material_class: str
    channel: str
    position: NDArray[np.float64]
    in_reachable_window: bool
    bbox: tuple[int, int, int, int] | None = None
    orientation: tuple[float, float, float, float] | None = None
    linear_velocity: tuple[float, float, float] | None = None
    angular_velocity: tuple[float, float, float] | None = None
    object_name: str = ""

    def __post_init__(self) -> None:
        """Refuse a class the taxonomy does not define, and coerce position.

        Raises:
            LabelError: If the material class is unknown, naming it.
        """
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64)
        )
        if self.material_class not in BY_ID:
            raise LabelError(
                f"object {self.object_id} carries material class "
                f"{self.material_class!r}, which is not in the taxonomy"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectLabel):
            return NotImplemented
        return (
            self.object_id == other.object_id
            and self.material_class == other.material_class
            and self.channel == other.channel
            and bool(np.allclose(self.position, other.position, rtol=0.0, atol=1e-12))
            and self.in_reachable_window == other.in_reachable_window
            and self.bbox == other.bbox
            and self.orientation == other.orientation
            and self.linear_velocity == other.linear_velocity
            and self.angular_velocity == other.angular_velocity
            and self.object_name == other.object_name
        )

    def as_dict(self) -> dict[str, Any]:
        """Render as plain data for the dataset index."""
        return {
            "object_id": self.object_id,
            "material_class": self.material_class,
            "channel": self.channel,
            "position": list(self.position),
            "in_reachable_window": self.in_reachable_window,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "orientation": list(self.orientation) if self.orientation else None,
            "linear_velocity": (
                list(self.linear_velocity) if self.linear_velocity else None
            ),
            "angular_velocity": (
                list(self.angular_velocity) if self.angular_velocity else None
            ),
            "object_name": self.object_name,
        }


@dataclass(frozen=True)
class CameraCapture:
    """One camera's image and the truth measured in that image.

    Attributes:
        camera_id: Sensor id from the world configuration. Provenance for the
            pixels, not a key a fusion rule selects on.
        frame: Height by width by three, unsigned bytes.
        labels: Objects at this instant. Boxes are in this camera's pixels.
        instance_ids: Height by width, the spawn serial of the object owning
            each pixel, or zero for background. None on a record that stored
            boxes only.
    """

    camera_id: str
    frame: NDArray[np.uint8]
    labels: tuple[ObjectLabel, ...]
    instance_ids: NDArray[np.uint16] | None = None


@dataclass(frozen=True)
class Example:
    """One captured frame and everything known about it.

    Attributes:
        frame: The rendered frame, height by width by three, unsigned bytes.
        labels: The objects visible in it.
        simulated_time: Seconds of simulated time at capture.
        seed: Seed the rollout ran under.
        config_digest: Digest of the world configuration that produced it.
        origin: Simulated or real.
        arm_joints: Manipulator joint angles at capture, in radians. Empty for
            an example recorded before v0.6.2, and for any real image, since a
            photograph carries no proprioception. A policy that cannot see where
            its own arm is cannot account for it, which is why v0.7.0's policies
            were vision only.
        captures: One entry per detection camera. Empty on a format-1 record,
            which stores only `frame`. When present, `frame` and `labels` repeat
            the first camera so a reader that knows one image still works.
    """

    frame: NDArray[np.uint8]
    labels: tuple[ObjectLabel, ...]
    simulated_time: float
    seed: int
    config_digest: str
    origin: Origin = Origin.SIMULATED
    arm_joints: tuple[float, ...] = ()
    captures: tuple[CameraCapture, ...] = ()

    @property
    def material_classes(self) -> tuple[str, ...]:
        """The material classes of every labeled object, visible or not."""
        return tuple(label.material_class for label in self.labels)

    @property
    def visible_labels(self) -> tuple[ObjectLabel, ...]:
        """Only the objects actually present in the frame.

        Detection training must use these. Training on a label whose object is
        outside the camera's field of view teaches the model to predict a box
        where there are no pixels.
        """
        return tuple(label for label in self.labels if label.bbox is not None)


@dataclass(frozen=True)
class Rollout:
    """One continuous run of the world.

    Splits partition by rollout rather than by frame, because two frames of one
    object are almost perfectly correlated and a frame-level split would put
    near duplicates on both sides of the boundary.

    Attributes:
        rollout_id: Identity, unique within a dataset.
        seed: Seed this rollout ran under.
        examples: Captured examples, in capture order.
        belt_speed: Meters per second the belt was driven at, when the recording
            fixed it. None when the world drew a speed and the record did not
            keep it.
        spacing_meters: Metres of belt between releases, when the recording
            fixed the gap. None when each gap was drawn from a range.
    """

    rollout_id: str
    seed: int
    examples: tuple[Example, ...]
    belt_speed: float | None = None
    spacing_meters: float | None = None
