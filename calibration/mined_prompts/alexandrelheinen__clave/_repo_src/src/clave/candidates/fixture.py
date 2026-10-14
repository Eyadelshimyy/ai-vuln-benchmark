"""Deterministic inputs for benchmarking.

No binary is committed. Frames and states are generated from a seed through the
platform's single seeding entry point, so two benchmark runs see identical
inputs and a third machine can reproduce them without downloading anything.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from clave.experiment.seeding import seed_everything

FRAME_HEIGHT = 96
FRAME_WIDTH = 96
STATE_DIMENSION = 6
ACTION_DIMENSION = 6


def frame(seed: int = 0) -> NDArray[np.float32]:
    """Generate one synthetic conveyor frame.

    Args:
        seed: Seed applied before generation.

    Returns:
        A float32 array shaped (3, FRAME_HEIGHT, FRAME_WIDTH) with values in
        [0, 1], matching the channel-first convention the candidates expect.
    """
    seed_everything(seed)
    return np.random.rand(3, FRAME_HEIGHT, FRAME_WIDTH).astype(np.float32)


def state(seed: int = 0) -> NDArray[np.float32]:
    """Generate one synthetic effector state.

    Args:
        seed: Seed applied before generation.

    Returns:
        A float32 array shaped (STATE_DIMENSION,).
    """
    seed_everything(seed)
    return np.random.rand(STATE_DIMENSION).astype(np.float32)
