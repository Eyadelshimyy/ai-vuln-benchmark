"""Where the reachable window ends, and why half its length is not that.

`clave.runtime.loop` and `clave.data.recorder` both computed the exit as
`window_length / 2.0`. That is only the exit coordinate because the arm happens
to stand at the origin, so the two agree today and would stop agreeing the
moment anybody moved it. The exit is a measured edge, and it is read as one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clave.world import belt, config
from clave.world.scene import BeltGeometry, SceneLayout, TakeawayDrive

ROOT = Path(__file__).resolve().parents[2]
_ARM = config.load(ROOT / "configs" / "world" / "sorting_line.yml")["arm"]
_REACH_MIN = float(_ARM["reach_min_meters"])
_REACH_MAX = float(_ARM["reach_max_meters"])
_TOOL_ABOVE = (
    float(_ARM["tool_above_base_meters"][0]),
    float(_ARM["tool_above_base_meters"][1]),
)


def layout(arm_x: float) -> SceneLayout:
    """A layout for reachability arithmetic, needing no physics.

    Reach bounds come from the shipped world configuration.
    """
    return SceneLayout(
        belt=BeltGeometry(
            length=3.0,
            width=1.0,
            surface_height=0.90,
            speed=0.31,
            height_tolerance=0.05,
        ),
        arm_base=np.asarray((arm_x, -0.70, 0.90), dtype=np.float64),
        reach_min=_REACH_MIN,
        reach_max=_REACH_MAX,
        tool_above_base=_TOOL_ABOVE,
        pedestal=(0.30, 0.30),
        channels=("CH-PET",),
        objects=(),
        pool_size=0,
        timestep=0.002,
        takeaway=TakeawayDrive(
            speed=0.20,
            height_min=0.25,
            height_max=0.45,
            mouth_half_width=0.15,
            past_mouth=1.50,
            toward_mouth=0.10,
        ),
    )


def test_the_window_exit_is_the_measured_downstream_edge() -> None:
    """The exit is the downstream edge of the annulus, not half the window."""
    plan = layout(0.0)
    exit_coordinate = belt.window_exit(plan)
    assert exit_coordinate is not None
    assert exit_coordinate == pytest.approx(belt.reach_report(plan).window_edges[1])


def test_moving_the_arm_moves_the_exit() -> None:
    """An exit computed as half the window length would not move with the arm."""
    at_origin = belt.window_exit(layout(0.0))
    shifted = belt.window_exit(layout(0.20))
    assert at_origin is not None and shifted is not None
    assert shifted != pytest.approx(at_origin)


def test_a_belt_that_never_enters_reach_reports_no_exit() -> None:
    """No window means no exit."""
    plan = layout(arm_x=0.0)
    # Push the belt far past the outer radius.
    plan = SceneLayout(
        belt=plan.belt,
        arm_base=np.asarray((0.0, -(_REACH_MAX + 0.5), 0.90), dtype=np.float64),
        reach_min=_REACH_MIN,
        reach_max=_REACH_MAX,
        tool_above_base=_TOOL_ABOVE,
        pedestal=plan.pedestal,
        channels=plan.channels,
        takeaway=plan.takeaway,
        objects=plan.objects,
        pool_size=plan.pool_size,
        timestep=plan.timestep,
    )
    assert belt.window_exit(plan) is None
    assert belt.reach_report(plan).window_length == 0.0
