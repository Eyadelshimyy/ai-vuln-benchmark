"""CLI: ``normalise`` — per-stage text cleaning over an extraction shard dir.

Per-stage (``--shards``) command mirroring ``womblex chunk``/``pii``: reads
``*.elements.parquet`` and writes a ``*.normalised_text.parquet`` sibling in
place — a cleaned text layer over the narrative elements (whitespace collapse,
footer-glyph despacing, configurable letterhead/font-map substitutions).
Verbatim-policy cleanup lives here, never in extraction. Offline (no API).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_normalise(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) with `*.elements.parquet`; "
             "writes `*.normalised_text.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source the `normalise:` section. "
             "Defaults to NormaliseConfig() if omitted.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.normalise-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="normalise",
                   help="Checkpoint dataset name. Default: 'normalise'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear normalise-stage checkpoint before running (re-normalise every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.normalised_text.parquet` integrity scan.")


def cmd_normalise(args: argparse.Namespace) -> int:
    """Normalise narrative text over a shard directory (per-stage, offline)."""
    from womblex.config import NormaliseConfig, load_config
    from womblex.process.normalise_stage import normalise_shards
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.normalise_output import NORMALISED_TEXT_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.elements.parquet")):
        logger.error(
            "--shards directory has no `*.elements.parquet` — run extraction "
            "first: %s", shard_dir,
        )
        return 1

    config = load_config(args.config).normalise if args.config else NormaliseConfig()

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".normalise-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_normalise")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=NORMALISED_TEXT_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "normalised_text shards; they will be re-normalised.", len(dropped))

    logger.info(
        "normalise --shards: dir=%s collapse_ws=%s despace_page=%s subs=%d",
        shard_dir, config.collapse_whitespace, config.despace_page_marker,
        len(config.substitutions),
    )
    result = normalise_shards(shard_dir, config, checkpoint_mgr=ckpt)
    logger.info(
        "Done: %d batches written, %d elements normalised (%d changed)",
        result.batches_written, result.elements_normalised, result.elements_changed,
    )
    return 0


COMMANDS = [
    Command("normalise", "Clean narrative text over a shard dir (per-stage, offline)",
            _register_normalise, cmd_normalise),
]
