"""CLI: ``spellfix`` — dictionary-gated OCR character-confusion repair.

Per-stage (``--shards``) command mirroring ``womblex normalise``: reads
``*.elements.parquet`` and writes a repaired ``*.spellfix_text.parquet`` overlay
plus a ``*.spellfix_corrections.parquet`` audit sibling in place. Only
out-of-dictionary tokens with a single unambiguous in-dictionary candidate are
rewritten (Tier A homoglyph by default; Tier B general edits behind ``--general``).
Raw elements are never modified; downstream stages opt in via
``text_source='spellfix'``. Offline (no API).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command, add_models_check_argument, check_stage_models
from womblex.utils.model_check import SCOPE_SPELLFIX

logger = logging.getLogger("womblex")


def _register_spellfix(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) with `*.elements.parquet`; writes "
             "`*.spellfix_text.parquet` + `*.spellfix_corrections.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source the `spellfix:` section. "
             "Defaults to SpellfixConfig() if omitted.",
    )
    p.add_argument(
        "--general", action="store_true",
        help="Enable Tier B general edit-distance-1 candidates (higher recall, "
             "proper-noun risk). Overrides the config `general_edits` flag.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.spellfix-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="spellfix",
                   help="Checkpoint dataset name. Default: 'spellfix'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear spellfix-stage checkpoint before running (re-repair every batch).")
    add_models_check_argument(p)


def cmd_spellfix(args: argparse.Namespace) -> int:
    """Repair OCR character-confusions over a shard directory (per-stage, offline)."""
    from womblex.config import SpellfixConfig, load_config
    from womblex.process.spellfix_stage import spellfix_shards
    from womblex.store.checkpoint import CheckpointManager

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.elements.parquet")):
        logger.error(
            "--shards directory has no `*.elements.parquet` — run extraction first: %s",
            shard_dir,
        )
        return 1

    full = load_config(args.config) if args.config else None
    config = full.spellfix if full else SpellfixConfig()
    if args.general:
        config = config.model_copy(update={"general_edits": True})
    if not check_stage_models(args, (SCOPE_SPELLFIX,), config=full, spellfix=config):
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".spellfix-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_spellfix")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()

    logger.info(
        "spellfix --shards: dir=%s general_edits=%s dict=%s",
        shard_dir, config.general_edits, config.dict_name,
    )
    result = spellfix_shards(shard_dir, config, checkpoint_mgr=ckpt)
    logger.info(
        "Done: %d batches written, %d elements repaired (%d corrections applied)",
        result.batches_written, result.elements_repaired, result.corrections_applied,
    )
    return 0


COMMANDS = [
    Command("spellfix", "Repair OCR character-confusions over a shard dir (per-stage, offline)",
            _register_spellfix, cmd_spellfix),
]
