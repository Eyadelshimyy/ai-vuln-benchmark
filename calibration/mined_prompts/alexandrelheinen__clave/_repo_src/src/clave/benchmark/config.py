"""What the benchmark compares, and over which seeds.

Every tunable lives in YAML for the same reason the world's does: a seed list
buried in Python is a comparison nobody can rerun differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clave.errors import ClaveError


class BenchmarkConfigError(ClaveError):
    """The benchmark configuration is missing a key or names something unknown."""


@dataclass(frozen=True)
class Configuration:
    """One row of the comparison.

    Attributes:
        name: How the row is labeled.
        predictor: `scripted` for the control, `checkpoints` for trained models.
        perception: Registry name of the classifier, when there is one.
        policy: Registry name of the pick policy, when there is one.
        note: Why this row is in the table, carried into the report.
    """

    name: str
    predictor: str
    perception: str | None
    policy: str | None
    note: str


@dataclass(frozen=True)
class BenchmarkConfig:
    """The whole comparison.

    Attributes:
        seeds: Every configuration runs each of these, so a difference between
            rows is a difference between configurations.
        seconds_per_run: Simulated seconds per run.
        runtime: The runtime configuration every row loads.
        gates: The gate configuration every row is scored against.
        configurations: The rows.
        digest: Digest of the file this came from, carried into the evidence
            pack so a number can be traced to the comparison that produced it.
    """

    seeds: tuple[int, ...]
    seconds_per_run: float
    runtime: Path
    gates: Path
    configurations: tuple[Configuration, ...]
    digest: str

    @classmethod
    def load(cls, path: Path) -> BenchmarkConfig:
        """Read the configuration from YAML.

        Args:
            path: The configuration file.

        Returns:
            The configuration.

        Raises:
            BenchmarkConfigError: If a key is absent or a row names no
                predictor this benchmark can build.
        """
        from clave.corpus.artifacts import digest_of

        raw = yaml.safe_load(path.read_text())
        benchmark = _require(raw, "benchmark")
        rows = _require(raw, "configurations")
        if not isinstance(rows, list) or not rows:
            raise BenchmarkConfigError("configurations names no row to compare")
        return cls(
            seeds=tuple(
                int(seed) for seed in _require(benchmark, "seeds", "benchmark")
            ),
            seconds_per_run=float(_require(benchmark, "seconds_per_run", "benchmark")),
            runtime=Path(str(_require(benchmark, "runtime", "benchmark"))),
            gates=Path(str(_require(benchmark, "gates", "benchmark"))),
            configurations=tuple(_configuration(row) for row in rows),
            digest=digest_of(path),
        )


def _configuration(row: Any) -> Configuration:
    """Read one row, refusing a predictor the benchmark cannot build."""
    name = str(_require(row, "name", "configurations"))
    predictor = str(_require(row, "predictor", name))
    if predictor not in {"scripted", "checkpoints"}:
        raise BenchmarkConfigError(
            f"{name}: predictor {predictor!r} is neither 'scripted' nor 'checkpoints'"
        )
    if predictor == "checkpoints":
        perception: str | None = str(_require(row, "perception", name))
        policy: str | None = str(_require(row, "policy", name))
    else:
        perception, policy = None, None
    return Configuration(
        name=name,
        predictor=predictor,
        perception=perception,
        policy=policy,
        note=str(row.get("note", "")).strip(),
    )


def _require(mapping: Any, key: str, path: str = "") -> Any:
    """Read a key, failing with its location when it is absent.

    Args:
        mapping: The mapping to read from.
        key: The key required.
        path: Name of the parent, used in the error message.

    Returns:
        The value.

    Raises:
        BenchmarkConfigError: If the key is absent.
    """
    if not isinstance(mapping, dict) or key not in mapping:
        where = f"{path}.{key}" if path else key
        raise BenchmarkConfigError(f"required configuration key {where!r} is missing")
    return mapping[key]
