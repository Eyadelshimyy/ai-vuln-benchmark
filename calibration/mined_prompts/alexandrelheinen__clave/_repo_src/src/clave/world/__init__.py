"""The simulated sorting line.

Importing this package imports no physics engine. MuJoCo is loaded when a scene
is built, so the configuration and object set are readable without it.
"""

from clave.world.arm import (
    ArmIndices,
    ReachBounds,
    ReachError,
    end_effector_position,
    end_effector_position_world,
    joint_positions,
    locate,
    pinch_position,
    pinch_position_world,
    reachable,
    solve,
    step_toward,
    tool_yaw,
    tool_yaw_world,
)
from clave.world.belt import Conveyor, ReachReport, SpawnedObject, reach_report
from clave.world.config import Range, WorldConfigError, load, require, require_range
from clave.world.effector import Effector
from clave.world.objects import ObjectSpec, channels, material_classes, parse
from clave.world.scene import BeltGeometry, SceneLayout, TakeawayDrive, build, layout

__all__ = [
    "ArmIndices",
    "BeltGeometry",
    "Conveyor",
    "Effector",
    "ObjectSpec",
    "Range",
    "ReachBounds",
    "ReachError",
    "ReachReport",
    "SceneLayout",
    "SpawnedObject",
    "TakeawayDrive",
    "WorldConfigError",
    "build",
    "channels",
    "end_effector_position",
    "end_effector_position_world",
    "joint_positions",
    "layout",
    "load",
    "locate",
    "material_classes",
    "parse",
    "pinch_position",
    "pinch_position_world",
    "reach_report",
    "reachable",
    "require",
    "require_range",
    "solve",
    "step_toward",
    "tool_yaw",
    "tool_yaw_world",
]
