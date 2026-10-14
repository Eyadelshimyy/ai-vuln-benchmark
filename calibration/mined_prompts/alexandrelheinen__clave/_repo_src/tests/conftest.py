"""Fixtures shared by the tests that start the real runtime, and the one
environment variable every rendering test depends on.

The runtime is a built binary and a JSON configuration, and three test modules
need both. A second copy of either would drift from the first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from clave.runtime.bridge import BridgeError, locate_binary
from clave.runtime.proposal import Proposal

# Pin the rendering backend before any test module is imported.
#
# MuJoCo picks a backend the first time a renderer is built, and on a machine
# with no display the default aborts the interpreter rather than raising. Two
# test modules used to pin it inside a test function, which worked only
# because they happened to sort before every other module that renders:
# `tests/data` runs ahead of `tests/demo` and `tests/tracker`, so those were
# free-riding on a variable somebody else had set. Adding a rendering test
# under `tests/control`, which sorts earlier still, moved the first render
# ahead of the pin and the suite began aborting on CI while passing on a
# developer machine that resolves a working backend on its own.
#
# conftest is imported before any test module, so pinning it here is the one
# placement collection order cannot defeat.
os.environ.setdefault("MUJOCO_GL", "osmesa")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def runtime_binary() -> Path:
    """The built runtime, built once when it is absent.

    An operator building the workspace gets it for free; a clone that has not
    run cargo gets one build, and a machine with no cargo skips.
    """
    try:
        return locate_binary(ROOT)
    except BridgeError:
        pass
    if shutil.which("cargo") is None:
        pytest.skip("cargo is absent, so the runtime cannot be built here")
    subprocess.run(
        ["cargo", "build", "-p", "clave-sitl"], cwd=ROOT, check=True, timeout=900
    )
    return locate_binary(ROOT)


@pytest.fixture
def runtime_config() -> dict[str, Any]:
    """An envelope and routing policy matching the shipped world's geometry."""
    return {
        "base_meters": [0.0, -0.70, 0.90],
        "reach_meters": [0.25, 1.25],
        "tool_above_base_meters": [-0.05, 0.45],
        "belt_surface_z_meters": 0.90,
        "belt_x_meters": [-1.5, 1.5],
        "belt_y_meters": [-0.5, 0.5],
        "channels": {"M-01": 1},
        "reject_channel": 0,
        "confidence_floor": 0.6,
    }


@pytest.fixture
def make_proposal() -> Callable[[float, float, float, float], Proposal]:
    """Build a proposal naming a point at a confidence."""

    def build(x: float, y: float, z: float, confidence: float) -> Proposal:
        return Proposal(
            object_id=7,
            material_class="M-01",
            confidence=confidence,
            point=np.asarray((x, y, z), dtype=np.float64),
            yaw_radians=0.0,
            reference_time_nanos=1_000,
            window_start_nanos=1_000,
            window_end_nanos=2**62,
        )

    return build
