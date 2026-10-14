"""CLI: `embed` — vectorise chunks via Isaacus (kanon-2-embedder).

Per-stage (`--shards`) command over an existing shard directory that already
has `*.chunks.parquet`, mirroring `womblex enrich --shards`. The Isaacus SDK
is a core dependency; the deployment is picked from the environment
(`ISAACUS_API_KEY` or `ISAACUS_SAGEMAKER_ENDPOINTS`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command
from womblex.utils.isaacus_client import make_isaacus_client

logger = logging.getLogger("womblex")


def _register_embed(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir with `*.chunks.parquet`; writes `*.embeddings.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source embedding settings (model, task, dimensions).",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.embed-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="embed",
                   help="Checkpoint dataset name. Default: 'embed'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear embed-stage checkpoint before running (re-embed every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.embeddings.parquet` integrity scan.")


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed a shard directory's chunks via the Isaacus embedding API."""
    from womblex.analyse.embed_stage import embed_shards
    from womblex.config import EmbeddingConfig, load_config
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.embed_output import EMBEDDINGS_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.chunks.parquet")):
        logger.error(
            "--shards directory has no `*.chunks.parquet` — run "
            "`womblex chunk --shards` first: %s", shard_dir,
        )
        return 1

    embedding_config = load_config(args.config).embedding if args.config else EmbeddingConfig()

    try:
        # Hosted API (ISAACUS_API_KEY) or SageMaker (ISAACUS_SAGEMAKER_ENDPOINTS),
        # whichever the environment declares; the model is checked against the
        # deployed endpoints up front.
        client = make_isaacus_client(models=[embedding_config.model])
    except ImportError as e:
        logger.error("Isaacus SDK not usable (reinstall womblex): %s", e)
        return 1
    except Exception as e:
        # Logs the exception, not the key — the rule trips on "API_KEY" in the literal.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        logger.error(
            "Could not construct Isaacus client (check ISAACUS_API_KEY, or "
            "ISAACUS_SAGEMAKER_ENDPOINTS for a private deployment): %s", e,
        )
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".embed-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_embed")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=EMBEDDINGS_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "embeddings shards; they will be re-embedded.", len(dropped))

    logger.info("embed --shards: dir=%s model=%s task=%s",
                shard_dir, embedding_config.model, embedding_config.task)
    result = embed_shards(shard_dir, embedding_config, client=client, checkpoint_mgr=ckpt)
    logger.info("Done: %d batches written, %d chunks embedded",
                result.batches_written, result.chunks_embedded)
    return 0


COMMANDS = [
    Command("embed", "Vectorise chunks via Isaacus (per-stage)", _register_embed, cmd_embed),
]
