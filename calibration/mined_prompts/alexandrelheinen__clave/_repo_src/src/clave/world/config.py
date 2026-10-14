"""Configuration for the simulated sorting line.

Every tunable lives in YAML. Nothing in this package carries a numeric default,
because a world whose belt speed is buried in Python cannot be randomized, and
randomization is the reason the world exists.

A randomizable quantity is a two-element range rather than a scalar. Resolving
one draws through the platform's single seeding entry point, so one seed and one
configuration give one world.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from clave.errors import ClaveError


class WorldConfigError(ClaveError):
    """The world configuration is missing a key or carries an invalid range."""


@dataclass(frozen=True)
class Range:
    """An inclusive interval a value is drawn from.

    Attributes:
        low: Lower bound.
        high: Upper bound, not less than `low`.
    """

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one value from the interval.

        Args:
            rng: The generator to draw from.

        Returns:
            A value in [low, high].
        """
        return float(rng.uniform(self.low, self.high))

    @property
    def midpoint(self) -> float:
        """The centre of the interval, used where a nominal value is wanted."""
        return (self.low + self.high) / 2.0


def require(mapping: dict[str, Any], key: str, path: str = "") -> Any:
    """Read a key, failing with its location when it is absent.

    Args:
        mapping: The mapping to read from.
        key: The key required.
        path: Dotted path of the parent, used in the error message.

    Returns:
        The value.

    Raises:
        WorldConfigError: If the key is absent. Nothing is substituted, because
            a silently defaulted belt speed is a world nobody can reproduce.
    """
    if key not in mapping:
        where = f"{path}.{key}" if path else key
        raise WorldConfigError(f"required configuration key {where!r} is missing")
    return mapping[key]


def require_range(mapping: dict[str, Any], key: str, path: str = "") -> Range:
    """Read a two-element range, failing on absence or inversion.

    Args:
        mapping: The mapping to read from.
        key: The key required.
        path: Dotted path of the parent, used in the error message.

    Returns:
        The range.

    Raises:
        WorldConfigError: If the key is absent, is not a pair, or is inverted.
    """
    where = f"{path}.{key}" if path else key
    value = require(mapping, key, path)
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise WorldConfigError(f"{where!r} must be a two-element range, got {value!r}")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise WorldConfigError(f"{where!r} is inverted: {low} is greater than {high}")
    return Range(low, high)


def load(path: Path) -> dict[str, Any]:
    """Read a world configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.

    Raises:
        WorldConfigError: If the file is not a mapping.
    """
    parsed = yaml.safe_load(path.read_text())
    if not isinstance(parsed, dict):
        raise WorldConfigError(f"{path} does not contain a mapping")
    return parsed
