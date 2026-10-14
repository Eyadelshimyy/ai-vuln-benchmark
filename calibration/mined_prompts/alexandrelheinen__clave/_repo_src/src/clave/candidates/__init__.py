"""Shortlisted architectures behind one interface.

Importing this package imports no deep learning library. Every adapter defers
its imports until asked to load, so the registry is readable on a machine with
none of them installed.
"""

from clave.candidates.base import (
    MATERIAL_CLASS_COUNT,
    Candidate,
    CandidateSpec,
    LoadResult,
    Stage,
)
from clave.candidates.bench import BenchResult, benchmark, count_parameters
from clave.candidates.registry import REGISTRY, by_stage, specs, sweep

__all__ = [
    "MATERIAL_CLASS_COUNT",
    "REGISTRY",
    "BenchResult",
    "Candidate",
    "CandidateSpec",
    "LoadResult",
    "Stage",
    "benchmark",
    "by_stage",
    "count_parameters",
    "specs",
    "sweep",
]
