"""Turning the simulated world into datasets.

Importing this package imports no physics engine. The recorder loads MuJoCo when
asked to run, so the example types, splits, composition and ingestion are usable
without it.
"""

from clave.data.composition import PartComposition, compose, compose_parts
from clave.data.dataset import DatasetDescription, DatasetError, read, write
from clave.data.examples import Example, LabelError, ObjectLabel, Origin, Rollout
from clave.data.expert import PickDecision, decide
from clave.data.ingest import CORPUS_MAPPINGS, MappedLabel, coverage, map_label
from clave.data.splits import PART_NAMES, SplitError, SplitPlan, split, verify

__all__ = [
    "CORPUS_MAPPINGS",
    "PART_NAMES",
    "DatasetDescription",
    "DatasetError",
    "Example",
    "LabelError",
    "MappedLabel",
    "ObjectLabel",
    "Origin",
    "PartComposition",
    "PickDecision",
    "Rollout",
    "SplitError",
    "SplitPlan",
    "compose",
    "compose_parts",
    "coverage",
    "decide",
    "map_label",
    "read",
    "split",
    "verify",
    "write",
]
