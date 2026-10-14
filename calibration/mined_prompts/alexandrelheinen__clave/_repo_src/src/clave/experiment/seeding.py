"""Seeding and the definition of reproducible.

Reproducible is a measured property here, not a claim. :data:`TOLERANCE` states
in numbers what two runs at the same seed are allowed to differ by, and a test
proves both halves: that the same seed reproduces within it, and that a
different seed does not, because a determinism test that also passes for a
constant function proves nothing.

PyTorch is seeded only when it is importable. The core carries no hard
dependency on a deep learning framework, so this module works on a machine that
has never installed one.
"""

from __future__ import annotations

import os
import random

import numpy as np

TOLERANCE: float = 1e-9
"""Maximum absolute difference between metrics of two runs at the same seed.

Set for deterministic CPU arithmetic in NumPy, which is what the platform
controls today. Accelerated or framework-backed training introduces its own
nondeterminism, so v0.7.0 has to revisit this number rather than inherit it.
"""


def seed_everything(seed: int) -> int:
    """Seed every source of randomness this platform controls.

    Seeds the standard library, NumPy, and the interpreter hash seed. PyTorch is
    seeded too when it is installed, which keeps this the single entry point
    without making the framework a hard dependency.

    Args:
        seed: The seed to apply.

    Returns:
        The seed that was applied, so a caller can record it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        pass
    else:  # pragma: no cover - exercised only where torch is installed
        torch.manual_seed(seed)
    return seed


def reproduces(left: float, right: float, tolerance: float = TOLERANCE) -> bool:
    """Whether two metrics count as the same result.

    Args:
        left: A metric from one run.
        right: The same metric from another run.
        tolerance: Maximum absolute difference permitted.

    Returns:
        True when the two differ by no more than the tolerance.
    """
    return abs(left - right) <= tolerance
