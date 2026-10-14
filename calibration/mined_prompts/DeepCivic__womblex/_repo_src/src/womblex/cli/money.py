"""CLI: ``money`` — monetary amount annotation over an extraction shard dir.

Per-stage (``--shards``) command mirroring ``womblex normalise``/``quality``:
reads ``*.elements.parquet`` + ``*.table_cells.parquet`` and writes
``*.money_spans.parquet`` (one row per amount, across the narrative / table_cell
/ sheet_cell loci) plus ``*.money_columns.parquet`` (the column-classification
audit) per batch. Annotation only — element and chunk text are untouched.
Offline (no API). See ``docs/money-extraction.md``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_money(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) with `*.elements.parquet`; "
             "writes `*.money_spans.parquet` + `*.money_columns.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source the `money:` section (and processing.text_source). "
             "Defaults to MoneyConfig() if omitted.",
    )
    p.add_argument(
        "--text-source", type=str, default=None,
        choices=["elements", "normalised", "spellfix"],
        help="Override the element-text layer narrative offsets index.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.money-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="money",
                   help="Checkpoint dataset name. Default: 'money'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear money-stage checkpoint before running (re-annotate every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.money_spans.parquet` integrity scan.")


def cmd_money(args: argparse.Namespace) -> int:
    """Annotate monetary amounts over a shard directory (per-stage, offline)."""
    from womblex.config import MoneyConfig, load_config
    from womblex.process.money_stage import money_shards
    from womblex.process.text_overlay import MissingOverlayError
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.money_output import MONEY_SPANS_SUFFIX
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

    text_source = "elements"
    if args.config:
        cfg = load_config(args.config)
        config, text_source = cfg.money, cfg.processing.text_source
    else:
        config = MoneyConfig()
    if args.text_source:
        # An explicit flag outranks the config: `money.text_source` would
        # otherwise win inside the stage and silently ignore the override.
        text_source = args.text_source
        config = config.model_copy(update={"text_source": args.text_source})
    if not config.enabled:
        logger.info("money stage disabled in config; nothing to do.")
        return 0

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".money-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_money")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=MONEY_SPANS_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "money_spans shards; they will be re-annotated.", len(dropped))

    logger.info(
        "money --shards: dir=%s text_source=%s narrative=%s columns=%s "
        "implicit_context=%s intl_numbers=%s",
        shard_dir, config.text_source or text_source, config.narrative,
        config.columns.enabled, config.implicit_context, config.international_numbers,
    )
    try:
        result = money_shards(shard_dir, config, text_source=text_source, checkpoint_mgr=ckpt)
    except MissingOverlayError as e:
        logger.error("money --shards: refused: %s", e)
        return 1
    logger.info(
        "Done: %d batches written, %d amounts annotated "
        "(%d columns classified, %d money columns)",
        result.batches_written, result.spans_written,
        result.columns_classified, result.money_columns,
    )
    return 0


COMMANDS = [
    Command("money", "Annotate monetary amounts over a shard dir (per-stage, offline)",
            _register_money, cmd_money),
]
