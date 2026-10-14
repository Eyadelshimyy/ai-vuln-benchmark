"""CLI: ``pii`` — per-stage PII span detection over an extraction shard dir.

Per-stage (``--shards``) command mirroring ``womblex chunk``/``enrich``/``link``:
reads ``*.chunks.parquet`` + ``*.enrichment_entities.parquet`` and writes a
``*.pii_spans.parquet`` sibling in place (sidecar-only — no text rewrite). The
Kanon-2 enrichment graph is the primary entity source; a local regex+context
backstop covers table chunks and graph misses (toggle via
``pii.use_regex_backstop`` in config). Offline (no API) — the enrichment graph
was produced upstream by ``womblex enrich``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command, add_models_check_argument, check_stage_models
from womblex.utils.model_check import SCOPE_PII

logger = logging.getLogger("womblex")


def _register_pii(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) with `*.chunks.parquet` + "
             "`*.enrichment_entities.parquet`; writes `*.pii_spans.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source the `pii:` section (entities, person_types, "
             "use_regex_backstop, threshold, model). Defaults to PIIConfig() if omitted.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.pii-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="pii",
                   help="Checkpoint dataset name. Default: 'pii'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear pii-stage checkpoint before running (re-detect every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.pii_spans.parquet` integrity scan.")
    add_models_check_argument(p)


def cmd_pii(args: argparse.Namespace) -> int:
    """Detect PII spans over a shard directory (per-stage, offline)."""
    from womblex.config import PIIConfig, load_config
    from womblex.pii.pii_stage import pii_shards
    from womblex.process.text_overlay import MissingOverlayError
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.pii_output import PII_SPANS_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.chunks.parquet")):
        logger.error(
            "--shards directory has no `*.chunks.parquet` — run `womblex chunk "
            "--shards` first: %s", shard_dir,
        )
        return 1
    if not any(shard_dir.glob("*.enrichment_entities.parquet")):
        logger.warning(
            "--shards directory has no `*.enrichment_entities.parquet` — PII will "
            "rely on the regex backstop only (run `womblex enrich --shards` for "
            "graph-driven detection): %s", shard_dir,
        )

    full = load_config(args.config) if args.config else None
    pii_config = full.pii if full else PIIConfig()
    if not check_stage_models(args, (SCOPE_PII,), config=full, pii=pii_config):
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".pii-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_pii")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=PII_SPANS_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "pii_spans shards; they will be re-detected.", len(dropped))

    logger.info(
        "pii --shards: dir=%s entities=%s regex_backstop=%s clean_text=%s",
        shard_dir, pii_config.entities, pii_config.use_regex_backstop,
        pii_config.write_clean_text,
    )
    try:
        result = pii_shards(
            shard_dir, pii_config,
            text_source=full.processing.text_source if full else "elements",
            checkpoint_mgr=ckpt,
        )
    except MissingOverlayError as e:
        logger.error("pii --shards: refused: %s", e)
        return 1
    logger.info(
        "Done: %d batches written, %d PII spans, %d chunks masked",
        result.batches_written, result.spans_written, result.chunks_masked,
    )
    return 0


COMMANDS = [
    Command("pii", "Detect PII spans over a shard dir (per-stage, offline)", _register_pii, cmd_pii),
]
