"""Ground-truth CLI subcommand: ``ground-truth`` over an extraction shard dir.

Decision 5 of ``docs/ground-truth-review.md`` (womblex-benchmark): reconstruction
is a *renderer*, not a stage — a library function
(:func:`womblex.process.ground_truth.build_ground_truth`) plus this CLI command
over an existing shard directory, regenerating baselines and their metadata
sidecars on demand, with determinism guaranteed by the recorded inputs rather
than by persisting the artefact. The extraction half is an ordinary
``womblex run --config configs/ground-truth.yaml``; this command is the thin
helper that runs after it.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_ground_truth(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Extraction shard directory (*.elements.parquet + *._manifest.parquet)",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output directory for the *.gt.md baselines and *.meta.json sidecars",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="Preset config (e.g. configs/ground-truth.yaml): sources the "
             "segmentation budget and processing.text_source. Its digest is each "
             "unit's preset_digest and its dataset.name the preset, both read from "
             "the shard footer rather than here.",
    )


def cmd_ground_truth(args: argparse.Namespace) -> int:
    """Segment, render and stamp every document in a shard directory."""
    from womblex.config import load_config
    from womblex.process.ground_truth import build_ground_truth

    shard_dir = Path(args.shards)
    if not shard_dir.is_dir():
        logger.error("shards dir not found: %s", shard_dir)
        return 1
    if not Path(args.config).is_file():
        logger.error("config file not found: %s", args.config)
        return 1

    config = load_config(args.config)
    try:
        result = build_ground_truth(
            shard_dir, Path(args.out), config.segmentation,
            text_source=config.processing.text_source,
        )
    except FileNotFoundError as exc:
        # A declared text_source overlay is missing beside the shard: the render
        # path fails loudly rather than baselining verbatim (U7). Report it as an
        # operational error, not a traceback.
        logger.error("ground-truth: %s", exc)
        return 1
    logger.info(
        "ground-truth: %d documents, %d units (%d oversize) -> %s",
        result.documents, result.units_written, result.oversize_units, args.out,
    )
    return 0


COMMANDS = [
    Command(
        "ground-truth",
        "Segment + render an extraction shard dir into *.gt.md baselines + *.meta.json sidecars",
        _register_ground_truth,
        cmd_ground_truth,
    ),
]
