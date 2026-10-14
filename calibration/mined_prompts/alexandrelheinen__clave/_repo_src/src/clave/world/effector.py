"""The jaw the pick geometry is measured against.

The dimensions are the shipped Robotiq 2F-85, read from configuration so a
marker and the compiled gripper share one description.

The jaw opening is not a field of its own. It is `arm.max_grasp_width_meters`,
which already decides which objects the world may spawn, and reading it from
there is what keeps a marker from calling an object unreachable that the world
was told to produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clave.world.config import WorldConfigError, require

ARRIVAL_LEAD_METERS = 0.001
"""How far above the open jaw's geometry the descent stops, in meters.

A rest-to-rest quintic over the 0.20 s the hold spends rising lags the hang
the linkage adds, by 0.16 mm at the sample that lags most. A millimetre
covers that and stays well inside the clearance the jaw keeps.
"""


@dataclass(frozen=True)
class Effector:
    """What the shipped jaw measures.

    Attributes:
        finger_length: How far the pads sit below the face the tool bolts to,
            in meters. This is the offset between where an object is and where
            the flange has to be.
        pad_thickness: One pad's extent along the closing direction, in
            meters, which is the way it travels to close.
        pad_depth: One pad's extent across the closing direction and
            horizontal, in meters.
        pad_height: One pad's vertical extent, in meters.
        grasp_height: Where the pads close, measured up from the belt surface,
            in meters. A configured plane rather than a reading, because no
            sensor on this line estimates an object's height.
        lowest_below_flange: How far the jaw's lowest collision geometry reaches
            below the flange, in meters, at its lowest over the jaw's own
            travel. Measured with the jaw shut, and larger than `finger_length`
            on the shipped jaw: the pads hang below the point the jaw closes at,
            so a clearance quoted at that point is not a clearance the jaw has.
        open_lowest_below_flange: The same reach with the jaw open, in meters.
            The linkage hangs less far open than shut, and the descent arrives
            against this one. The hold then rises by the difference while the
            jaw closes, which is what keeps the shut reach at the same clearance.
        jaw_clearance: The clearance the jaw's lowest geometry keeps above the
            belt surface, in meters.
        opening: The widest the jaw goes, in meters, read from the arm.
    """

    finger_length: float
    pad_thickness: float
    pad_depth: float
    pad_height: float
    grasp_height: float
    lowest_below_flange: float
    open_lowest_below_flange: float
    jaw_clearance: float
    opening: float

    @property
    def pinch_floor(self) -> float:
        """The lowest pinch plane that keeps the shut jaw clear of the belt.

        Measured from the belt surface, so a caller adds its own surface to it.
        The pads hang below the pinch point, so the plane the pinch point may
        stand at once the jaw has shut is the clearance plus that overhang.
        The marker clamps to `open_pinch_floor` instead: the descent arrives
        with the jaw open, and the hold climbs the difference.
        """
        return self.jaw_clearance + self.lowest_below_flange - self.finger_length

    @property
    def flange_floor(self) -> float:
        """The lowest flange height that keeps the shut jaw clear of the belt.

        Measured from the belt surface. This is where the flange has to be
        once the jaw has finished closing. The arrival is lower: see
        `open_flange_floor`.
        """
        return self.jaw_clearance + self.lowest_below_flange

    @property
    def closing_drop(self) -> float:
        """How much further the pads hang once the jaw has shut, in meters.

        The hold raises the flange by this while the jaw closes, so the
        clearance the open jaw arrived with is the clearance the shut jaw
        keeps.
        """
        return self.lowest_below_flange - self.open_lowest_below_flange

    @property
    def open_pinch_floor(self) -> float:
        """The lowest pinch plane that keeps the open jaw clear of the belt.

        Measured from the belt surface. A millimetre above the geometry, so
        the rise during the close, which a rest-to-rest quintic starts late,
        still leads the hang the linkage adds. The marker clamps its grasp
        plane to this. The shut plane is `pinch_floor`, and the hold climbs
        the difference.
        """
        return (
            self.jaw_clearance
            + self.open_lowest_below_flange
            - self.finger_length
            + ARRIVAL_LEAD_METERS
        )

    @property
    def open_flange_floor(self) -> float:
        """The lowest flange height that keeps the open jaw clear of the belt.

        Measured from the belt surface. The controller admits a descent to
        this, and the hold is above it.
        """
        return self.jaw_clearance + self.open_lowest_below_flange + ARRIVAL_LEAD_METERS

    @classmethod
    def load(cls, raw: dict[str, Any]) -> Effector:
        """Read the effector from a world configuration.

        Args:
            raw: The parsed world configuration.

        Returns:
            The effector it describes.

        Raises:
            WorldConfigError: If the block, or any key in it, is absent, or if
                a dimension is not positive. A pad of no thickness
                describes nothing, and a pad plane at or below the belt
                describes a jaw closing through it. A grasp plane inside the
                clearance the jaw's geometry keeps is refused for the same
                reason: the numbers would describe a configuration that closes
                the pads on the belt.
        """
        block = require(raw, "effector")
        arm = require(raw, "arm")
        values = {
            "finger_length": _positive(block, "finger_length_meters"),
            "pad_thickness": _positive(block, "pad_thickness_meters"),
            "pad_depth": _positive(block, "pad_depth_meters"),
            "pad_height": _positive(block, "pad_height_meters"),
            "grasp_height": _positive(block, "grasp_height_meters"),
            "lowest_below_flange": _positive(block, "lowest_below_flange_meters"),
            "open_lowest_below_flange": _positive(
                block, "open_lowest_below_flange_meters"
            ),
            "jaw_clearance": _positive(block, "jaw_clearance_meters"),
        }
        opening = float(require(arm, "max_grasp_width_meters", "arm"))
        effector = cls(opening=opening, **values)
        if effector.open_lowest_below_flange >= effector.lowest_below_flange:
            raise WorldConfigError(
                f"effector.open_lowest_below_flange_meters is "
                f"{effector.open_lowest_below_flange} and the shut jaw reaches "
                f"{effector.lowest_below_flange}, so closing would not drop the "
                f"pads and the hold would have nothing to rise by"
            )
        if effector.grasp_height < effector.pinch_floor:
            raise WorldConfigError(
                f"effector.grasp_height_meters is {effector.grasp_height} and the "
                f"jaw's lowest geometry reaches {effector.lowest_below_flange} "
                f"below the flange while keeping {effector.jaw_clearance} above "
                f"the belt, so the pads would close at {effector.pinch_floor} or "
                f"below"
            )
        return effector


def _positive(block: dict[str, Any], key: str) -> float:
    """Read one dimension, refusing anything that describes no solid.

    Args:
        block: The `effector` block.
        key: The key to read.

    Returns:
        The value.

    Raises:
        WorldConfigError: If the key is absent or the value is not above zero.
    """
    value = float(require(block, key, "effector"))
    if not value > 0.0:
        raise WorldConfigError(
            f"effector.{key} is {value}, and a dimension at or below zero "
            f"describes no effector"
        )
    return value
