"""CLI: graph-building stages — ``enrich`` (Kanon-2), ``link`` (register match)
and ``graph-refresh`` (offline mention→chunk edge rebuild).

All are per-stage (``--shards``) commands over an existing extraction shard
directory, mirroring ``womblex chunk --shards``: each has an independent
``CheckpointManager`` and writes sibling parquets in place. ``enrich`` uses
the Isaacus SDK (a core dependency) + ``ISAACUS_API_KEY``; ``link`` and
``graph-refresh`` are offline. Run ``graph-refresh`` *after* ``chunk`` (the
AI-chunking pipeline order is enrich → chunk → graph-refresh).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from womblex.cli._shared import Command
from womblex.utils.isaacus_client import make_isaacus_client

logger = logging.getLogger("womblex")


# --- enrich ------------------------------------------------------------------


def _register_enrich(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) to enrich; writes "
             "`*.enrichment_entities.parquet` + `*.enrichment_meta.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML to source enrichment settings (model, retries, "
             "skip_short_documents). Defaults to EnrichmentConfig() if omitted.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.enrich-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="enrich",
                   help="Checkpoint dataset name. Default: 'enrich'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear enrich-stage checkpoint before running (re-enrich every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.enrichment_entities.parquet` integrity scan.")


def cmd_enrich(args: argparse.Namespace) -> int:
    """Enrich a shard directory via the Kanon-2 enrichment API (per-stage)."""
    from womblex.analyse.enrich_stage import enrich_shards
    from womblex.config import EnrichmentConfig, load_config
    from womblex.process.text_overlay import MissingOverlayError
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*._manifest.parquet")):
        logger.error("--shards directory has no `*._manifest.parquet`: %s", shard_dir)
        return 1

    if args.config:
        _cfg = load_config(args.config)
        enrichment_config = _cfg.enrichment
        text_source = _cfg.processing.text_source
        # Persist the raw Document for chunk-stage reuse whenever this config
        # also enables AI chunking — even if enrichment.enabled is false, since
        # the user is running `enrich` explicitly here. WomblexConfig already
        # auto-enables persist_document for the both-on case.
        persist_document = (
            enrichment_config.persist_document
            or bool(_cfg.chunking.chunking_model)
        )
    else:
        enrichment_config = EnrichmentConfig()
        text_source = "elements"
        persist_document = enrichment_config.persist_document

    try:
        # Hosted API (ISAACUS_API_KEY) or SageMaker (ISAACUS_SAGEMAKER_ENDPOINTS),
        # whichever the environment declares; the model is checked against the
        # deployed endpoints up front.
        client = make_isaacus_client(models=[enrichment_config.model])
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

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".enrich-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_enrich")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=ENRICHMENT_ENTITIES_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "enrichment shards; they will be re-enriched.", len(dropped))

    logger.info("enrich --shards: dir=%s model=%s text_source=%s overflow=%s persist_doc=%s",
                shard_dir, enrichment_config.model, text_source,
                enrichment_config.overflow_strategy, persist_document)
    try:
        result = enrich_shards(shard_dir, enrichment_config, client=client,
                               text_source=text_source, persist_document=persist_document,
                               checkpoint_mgr=ckpt)
    except MissingOverlayError as e:
        logger.error("enrich --shards: refused: %s", e)
        return 1
    logger.info(
        "Done: %d batches written, %d docs enriched, %d entities",
        result.batches_written, result.docs_enriched, result.total_entities,
    )
    return 0


# --- link --------------------------------------------------------------------


def _register_link(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir with `*.enrichment_entities.parquet`; writes "
             "`*.entity_links.parquet` in place.",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="YAML carrying the `linking:` section (reference register mapping).",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.link-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="link",
                   help="Checkpoint dataset name. Default: 'link'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear link-stage checkpoint before running (re-link every batch).")
    p.add_argument("--no-verify-resume", action="store_true",
                   help="Skip the resume-time `*.entity_links.parquet` integrity scan.")


def cmd_link(args: argparse.Namespace) -> int:
    """Match enrichment candidates to a reference register (per-stage, offline)."""
    from womblex.config import load_config
    from womblex.link.stage import link_shards
    from womblex.process.text_overlay import MissingOverlayError
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.entity_links_output import ENTITY_LINKS_SUFFIX
    from womblex.store.shard_audit import reconcile_stage_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.enrichment_entities.parquet")):
        logger.error(
            "--shards directory has no `*.enrichment_entities.parquet` — run "
            "`womblex enrich --shards` first: %s", shard_dir,
        )
        return 1

    full_config = load_config(args.config)
    linking_config = full_config.linking
    if linking_config.reference is None:
        logger.error("config `linking.reference` is required to run the link stage")
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".link-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_link")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_stage_checkpoint_with_shards(
                ckpt, shard_dir, suffix=ENTITY_LINKS_SUFFIX)
            if dropped:
                logger.warning("Resume integrity scan: dropped %d doc(s) with corrupted "
                               "entity_links shards; they will be re-linked.", len(dropped))

    logger.info("link --shards: dir=%s reference=%s", shard_dir, linking_config.reference.path)
    try:
        result = link_shards(
            shard_dir, linking_config,
            text_source=full_config.processing.text_source, checkpoint_mgr=ckpt,
        )
    except MissingOverlayError as e:
        logger.error("link --shards: refused: %s", e)
        return 1
    logger.info(
        "Done: %d batches written, %d docs linked, %d/%d link rows matched",
        result.batches_written, result.docs_linked, result.matched_links, result.total_links,
    )
    return 0


# --- graph-refresh -----------------------------------------------------------


def _register_graph_refresh(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir with `*.enrichment_entities.parquet` + `*.chunks.parquet`; "
             "rewrites those + `*.graph_edges.parquet` in place with mention→chunk links.",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Per-stage checkpoint root (default: `<shard_dir>/../.graph-refresh-checkpoint/`).")
    p.add_argument("--dataset", type=str, default="graph_refresh",
                   help="Checkpoint dataset name. Default: 'graph_refresh'.")
    p.add_argument("--no-resume", action="store_true",
                   help="Clear the checkpoint before running (re-refresh every batch).")


def cmd_graph_refresh(args: argparse.Namespace) -> int:
    """Rebuild mention→chunk edges from entities + chunks (per-stage, offline)."""
    from womblex.analyse.graph_refresh import refresh_graph_edges
    from womblex.store.checkpoint import CheckpointManager

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*.enrichment_entities.parquet")):
        logger.error(
            "--shards directory has no `*.enrichment_entities.parquet` — run "
            "`womblex enrich --shards` first: %s", shard_dir,
        )
        return 1
    if not any(shard_dir.glob("*.chunks.parquet")):
        logger.error(
            "--shards directory has no `*.chunks.parquet` — run "
            "`womblex chunk --shards` first: %s", shard_dir,
        )
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".graph-refresh-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_graph_refresh")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()

    logger.info("graph-refresh --shards: dir=%s", shard_dir)
    result = refresh_graph_edges(shard_dir, checkpoint_mgr=ckpt)
    logger.info(
        "Done: %d batches written, %d docs refreshed, %d mention edges",
        result.batches_written, result.docs_refreshed, result.edges_added,
    )
    return 0


COMMANDS = [
    Command("enrich", "Enrich a shard directory via Kanon-2 (per-stage)", _register_enrich, cmd_enrich),
    Command("link", "Match enrichment candidates to a reference register", _register_link, cmd_link),
    Command(
        "graph-refresh", "Rebuild mention→chunk graph edges after chunking (offline)",
        _register_graph_refresh, cmd_graph_refresh,
    ),
]
