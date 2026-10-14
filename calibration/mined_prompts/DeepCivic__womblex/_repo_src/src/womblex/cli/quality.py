"""CLI: ``quality`` — chunk-quality annotation over an extraction shard dir.

Per-stage (``--shards``) command mirroring ``womblex normalise``/``chunk``:
reads ``*.chunks.parquet`` and writes a ``*.chunk_quality.parquet`` sibling per
batch with ML-readiness flags (char_len, alpha_frac, is_short, boilerplate) and
cross-batch duplicate cluster ids. Annotation only — chunk text is untouched.
Single global pass (dedup is corpus-wide); offline (no API).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _register_quality(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) with `*.chunks.parquet`; "
             "writes `*.chunk_quality.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source the `quality:` section. "
             "Defaults to QualityConfig() if omitted.",
    )


def cmd_quality(args: argparse.Namespace) -> int:
    """Annotate chunk quality over a shard directory (per-stage, offline)."""
    from womblex.config import QualityConfig, load_config
    from womblex.process.quality_stage import quality_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.chunks.parquet")):
        logger.error(
            "--shards directory has no `*.chunks.parquet` — run chunking "
            "first: %s", shard_dir,
        )
        return 1

    config = load_config(args.config).quality if args.config else QualityConfig()
    if not config.enabled:
        logger.info("quality stage disabled in config; nothing to do.")
        return 0

    logger.info(
        "quality --shards: dir=%s dedup=%s short_chars=%d boilerplate_patterns=%d",
        shard_dir, config.dedup, config.short_chars, len(config.boilerplate_patterns),
    )
    result = quality_shards(shard_dir, config)
    logger.info(
        "Done: %d batches written, %d chunks annotated "
        "(%d exact-dup clusters, %d near-dup clusters)",
        result.batches_written, result.chunks_annotated,
        result.exact_dup_clusters, result.near_dup_clusters,
    )
    return 0


COMMANDS = [
    Command("quality", "Annotate chunk quality over a shard dir (per-stage, offline)",
            _register_quality, cmd_quality),
]
