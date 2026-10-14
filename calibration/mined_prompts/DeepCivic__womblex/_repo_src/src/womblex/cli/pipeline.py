"""Pipeline CLI subcommands: ``run`` (extraction + optional redaction detection),
``extract`` (single file), ``chunk`` (extract + chunk a directory)."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from womblex.cli._shared import (
    Command,
    NestedCorpusError,
    add_models_check_argument,
    apply_models_check,
    check_stage_models,
    discover_files,
    format_eta,
)
from womblex.utils.model_check import SCOPE_CHUNK, SCOPE_EXTRACT

logger = logging.getLogger("womblex")


# --- run ---------------------------------------------------------------------


def _register_run(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument(
        "--no-verify-resume", action="store_true",
        help="Skip the resume-time shard integrity scan (advanced — only use "
             "if you've already verified shards yourself).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max documents to process")
    p.add_argument("--skip", type=int, default=0, help="Skip first N documents")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    p.add_argument(
        "--run-id", type=str, default=None,
        help=(
            "Identifier for this run instance (overrides dataset.run_id in config). "
            "Outputs land under <output_root>/<run_id>/documents/. "
            "If omitted, config value or an auto-generated timestamp is used."
        ),
    )
    add_models_check_argument(p)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the extraction pipeline from a config file.

    ``run`` is extraction only (extract → optional redaction detection). The
    downstream stages (chunk, enrich, embed, money, pii, …) are their own
    ``run-stage`` contracts and are dispatched separately — their
    ``config.<stage>.enabled`` flags declare pipeline membership, they do not
    make the stage run here.
    """
    from womblex.batch import process_batch
    from womblex.config import load_config
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.output import ShardVerificationError, verify_shard_persistence
    from womblex.store.retention import apply_retention, generate_run_id, most_recent_run
    from womblex.store.run_stamp import RunStamp
    from womblex.store.shard_audit import reconcile_checkpoint_with_shards
    from womblex.store.source_provenance import IngestProvenance
    from womblex.utils.log_format import log_context
    from womblex.utils.run_log import capture_batch_log

    config = load_config(args.config)
    apply_models_check(config, args)
    logger.info("Loaded config: %s", config.dataset.name)

    input_root = config.paths.input_root
    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)
        return 1

    try:
        all_files = discover_files(input_root, args.limit, args.skip)
    except NestedCorpusError as e:
        logger.error(str(e))
        return 1
    logger.info("Found %d documents to process", len(all_files))
    if not all_files:
        logger.error("No supported files found in %s", input_root)
        return 1

    # Before the output tree, retention or any shard exists, and for every
    # enabled stage, not just extraction: a run whose later stages cannot load
    # their models would spend its extraction on output nothing can finish.
    from womblex.utils.model_check import check_models

    model_check = check_models(config)
    if bad := model_check.failures:
        logger.error("Model check failed; no document was processed: %s", model_check.message(bad))
        return 1

    output_root = config.paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve run_id: CLI > config > (resume: most-recent existing) > auto-generated
    explicit_run_id = args.run_id or config.dataset.run_id
    if explicit_run_id:
        run_id = explicit_run_id
    elif args.resume:
        prev = most_recent_run(output_root)
        if prev is None:
            logger.error("--resume given but no existing run dir found under %s", output_root)
            return 1
        run_id = prev.name
        logger.info("--resume: picked most-recent run %s", run_id)
    else:
        run_id = generate_run_id()
    logger.info("run_id: %s", run_id)

    run_root = output_root / run_id
    shard_dir = run_root / "documents"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Batch logs land in a `logs/` sibling of `documents/`, the same layout the
    # cloud worker publishes to the store, so `ui/readers.py` reads both by one
    # `runs/<run_id>/logs/batch-NNNN.log` convention.
    logs_dir = run_root / "logs"
    cumulative_shard_size = sum(s.stat().st_size for s in shard_dir.glob("*.parquet"))

    # Apply retention before processing (only on fresh runs — never on resume,
    # which would risk purging newer runs the user is intentionally bypassing).
    if not args.resume:
        purged = apply_retention(
            output_root,
            config.paths.checkpoint_dir,
            current_run_id=run_id,
            policy=config.processing.retention.policy,
            keep=config.processing.retention.keep,
        )
        if purged:
            logger.info(
                "retention(%s, keep=%d): purged %d old run(s)",
                config.processing.retention.policy,
                config.processing.retention.keep,
                len(purged),
            )

    checkpoint_root = config.paths.checkpoint_dir / run_id
    checkpoint_mgr = CheckpointManager(checkpoint_root, config.dataset.name)
    # Offset the batch counter when resuming so the shard path
    # (batch-NNNN.parquet) continues from where the prior invocation left
    # off — otherwise resumed batches overwrite earlier shards.
    batch_num_offset = 0
    if args.resume:
        checkpoint_mgr.load()
        if not args.no_verify_resume:
            dropped = reconcile_checkpoint_with_shards(checkpoint_mgr, shard_dir)
            if dropped:
                logger.warning(
                    "Resume integrity scan: dropped %d doc(s) with corrupted shards; "
                    "they will be re-extracted.", len(dropped),
                )
                # Recompute cumulative size after archive renames so the
                # post-write shrink check doesn't trip on the archived files.
                cumulative_shard_size = sum(
                    s.stat().st_size for s in shard_dir.glob("*.parquet")
                )
        batch_num_offset = checkpoint_mgr.state.last_batch
        all_files = checkpoint_mgr.filter_unprocessed(all_files)
        logger.info("Resuming: %d documents remaining (next batch=%d)", len(all_files), batch_num_offset + 1)
    else:
        checkpoint_mgr.clear()

    if not all_files:
        logger.info("All documents already processed")
        return 0

    # Declared once for the run: every batch stamps the same root, so a shard
    # copied out of the run still names the corpus it was extracted from.
    provenance = IngestProvenance.from_config(config)
    logger.info("ingest_root: %s (collection %s)", provenance.ingest_root, provenance.collection_id)

    # Likewise declared once: the run id is otherwise only a directory name, so
    # a shard moved out of the run loses which run produced it.
    stamp = RunStamp.declare(run_id, config, stage="extract")
    logger.info("run stamp: version %s, config %s", stamp.version, stamp.config_digest)

    batch_size = args.batch_size or config.processing.batch_size
    total_files = len(all_files)
    total_succeeded = 0
    total_failed = 0
    start_time = time.time()

    # `run` is extraction only; redaction detection is the one true-to-source
    # annotation it also does. Chunk / PII / enrich / embed / money are
    # downstream `run-stage` contracts, dispatched separately — the config
    # flags declare pipeline membership, they don't run those stages here.
    stages = ["extraction"]
    if config.redaction.enabled:
        stages.append(f"redaction({config.redaction.mode})")
    logger.info(
        "Starting extraction: %d documents, batch_size=%d, stages=[%s]",
        total_files, batch_size, ", ".join(stages),
    )
    downstream = [
        name for name, enabled in (
            ("chunk", config.chunking.enabled),
            ("pii", config.pii.enabled),
        ) if enabled
    ]
    if downstream:
        logger.info(
            "Config also enables downstream stage(s) [%s] — `run` does not execute "
            "them; dispatch them via `womblex <stage> --shards` (or `run-stage`) "
            "after extraction, in order (enrich → chunk → graph-refresh → embed → "
            "money → pii).",
            ", ".join(downstream),
        )

    for batch_idx, i in enumerate(range(0, total_files, batch_size), start=1):
        batch_num = batch_idx + batch_num_offset
        batch_files = all_files[i : i + batch_size]
        batch_start = time.time()
        logger.info(
            "[Batch %d] Processing %d documents (%d-%d of %d)...",
            batch_num, len(batch_files), i + 1, min(i + batch_size, total_files), total_files,
        )

        with log_context(run_id=run_id, stage="extract"), \
                capture_batch_log(logs_dir / f"batch-{batch_num:04d}.log"):
            outcome = process_batch(
                batch_files, config, batch_num=batch_num, shard_dir=shard_dir,
                provenance=provenance, stamp=stamp,
            )
        batch = outcome.batch
        total_succeeded += batch.succeeded
        total_failed += batch.failed

        if outcome.rows_written > 0:
            try:
                cumulative_shard_size = verify_shard_persistence(
                    outcome.shard_path, outcome.rows_written, cumulative_shard_size,
                )
            except ShardVerificationError as e:
                logger.error("[Batch %d] integrity check failed: %s", batch_num, e)
                raise

        doc_ids = [r.doc_id for r in batch.results]
        checkpoint_mgr.update(doc_ids, batch.succeeded, batch.failed, batch_num)

        batch_elapsed = time.time() - batch_start
        docs_done = i + len(batch_files)
        docs_remaining = total_files - docs_done
        total_elapsed = time.time() - start_time
        avg_per_doc = total_elapsed / docs_done if docs_done > 0 else 0
        logger.info(
            "[Batch %d] Complete: %d ok, %d failed (%.1fs, ETA: %s)",
            batch_num, batch.succeeded, batch.failed, batch_elapsed,
            format_eta(avg_per_doc * docs_remaining),
        )

    from womblex.store.run_manifest import write_run_manifest

    manifest_path = write_run_manifest(shard_dir)

    total_elapsed = time.time() - start_time
    logger.info(
        "Done in %s: %d succeeded, %d failed. Output: %s (manifest: %s)",
        format_eta(total_elapsed), total_succeeded, total_failed, output_root, manifest_path,
    )
    return 0


# --- extract -----------------------------------------------------------------


def _register_extract(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="Path to document")
    p.add_argument("-o", "--output", default="output/", help="Output directory")
    p.add_argument(
        "--format",
        choices=["txt", "parquet"],
        default="txt",
        help="Output format: txt (one .txt per unit) or parquet (single .parquet file)",
    )


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract text from a single document (extraction stage only)."""
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
        WomblexConfig,
    )
    from womblex.operations import run_extraction

    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s", path)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = WomblexConfig(
        dataset=DatasetConfig(name="single"),
        paths=PathsConfig(
            input_root=path.parent,
            output_root=output_dir,
            checkpoint_dir=output_dir / ".checkpoints",
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )

    results = run_extraction([path], config)

    if args.format == "txt":
        completed = [r for r in results if r.status == "completed" and r.extraction]
        if len(completed) != 1:
            logger.error(
                "--format txt requires exactly 1 extraction unit, got %d. "
                "Use --format parquet for multi-unit files (e.g. spreadsheets).",
                len(completed),
            )
            return 1
        r = completed[0]
        assert r.extraction is not None  # a completed unit always carries extraction
        out_path = output_dir / f"{r.doc_id}.txt"
        out_path.write_text(r.extraction.full_text, encoding="utf-8")
        logger.info("  %s -> %s (%d chars)", r.doc_id, out_path, len(r.extraction.full_text))
    else:
        from womblex.store.output import write_results
        from womblex.store.source_provenance import IngestProvenance

        rows = [
            (r.doc_id, str(r.path), r.extraction)
            for r in results
            if r.status == "completed" and r.extraction
        ]
        if rows:
            out_path = output_dir / f"{path.stem}.parquet"
            write_results(
                rows, out_path, provenance=IngestProvenance.from_config(config),
            )
            logger.info("Wrote %d unit(s) to %s", len(rows), out_path)

    for r in results:
        if r.error:
            logger.error("  %s: %s", r.doc_id, r.error)

    ok = sum(1 for r in results if r.status == "completed")
    err = sum(1 for r in results if r.status == "error")
    logger.info("Extracted %d unit(s) (%d ok, %d errors)", len(results), ok, err)
    return 0 if err == 0 else 1


# --- chunk -------------------------------------------------------------------


def _register_chunk(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path,
        help=(
            "Per-stage mode: chunk an existing shard directory "
            "(``<run_id>/documents/`` containing ``*.elements.parquet``). "
            "Writes ``*.chunks.parquet`` siblings in place."
        ),
    )
    p.add_argument(
        "--config", type=Path,
        help=(
            "Config YAML. With --shards: sources chunking settings "
            "(tokenizer, chunk_size, chunking_model for AI chunking) and "
            "processing.text_source. Without --shards: E2E composition mode "
            "(extraction + chunking)."
        ),
    )
    p.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help=(
            "Per-stage checkpoint root (default: ``<shard_dir>/../.chunk-checkpoint/``). "
            "Only used with --shards."
        ),
    )
    p.add_argument(
        "--dataset", type=str, default="chunk",
        help="Checkpoint dataset name (only used with --shards). Default: 'chunk'.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Clear chunk-stage checkpoint before running (re-chunk every batch).",
    )
    p.add_argument(
        "--no-verify-resume", action="store_true",
        help=(
            "Skip the resume-time *.chunks.parquet integrity scan (advanced — "
            "only use if you've already verified the chunks files yourself)."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Max documents to process (--config only)")
    add_models_check_argument(p)


def cmd_chunk(args: argparse.Namespace) -> int:
    """Chunk a shard directory (per-stage) or a config-described corpus (E2E)."""
    if args.shards is not None:
        return _cmd_chunk_shards(args)
    if args.config is not None:
        return _cmd_chunk_config(args)
    logger.error("chunk requires --shards (per-stage) and/or --config (E2E)")
    return 1


def _cmd_chunk_shards(args: argparse.Namespace) -> int:
    from womblex.config import ChunkingConfig, load_config
    from womblex.process.chunk_stage import chunk_shards
    from womblex.process.text_overlay import MissingOverlayError
    from womblex.store.checkpoint import CheckpointManager
    from womblex.store.shard_audit import reconcile_chunk_checkpoint_with_shards

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*._manifest.parquet")):
        logger.error("--shards directory has no `*._manifest.parquet`: %s", shard_dir)
        return 1

    full = load_config(args.config) if args.config is not None else None
    if full is not None:
        chunking_config = full.chunking
        text_source = full.processing.text_source
    else:
        chunking_config = ChunkingConfig()
        text_source = "elements"
    if not check_stage_models(args, (SCOPE_CHUNK,), config=full, chunking=chunking_config):
        return 1

    checkpoint_root = args.checkpoint_dir or shard_dir.parent / ".chunk-checkpoint"
    ckpt = CheckpointManager(checkpoint_root, f"{args.dataset}_chunk")
    if args.no_resume:
        ckpt.clear()
    else:
        ckpt.load()
        if not args.no_verify_resume:
            dropped = reconcile_chunk_checkpoint_with_shards(ckpt, shard_dir)
            if dropped:
                logger.warning(
                    "Resume integrity scan: dropped %d doc(s) with corrupted "
                    "chunks shards; they will be re-chunked.", len(dropped),
                )

    # Logs chunking config only — the rule trips on "token" in "tokenizer".
    # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
    logger.info(
        "chunk --shards: dir=%s tokenizer=%s chunk_size=%d processes=%d text_source=%s",
        shard_dir, chunking_config.tokenizer, chunking_config.chunk_size,
        chunking_config.processes, text_source,
    )
    try:
        result = chunk_shards(
            shard_dir, chunking_config, text_source=text_source, checkpoint_mgr=ckpt)
    except MissingOverlayError as e:
        logger.error("chunk --shards: refused: %s", e)
        return 1
    logger.info(
        "Done: %d batches written, %d docs chunked, %d total chunks",
        result.batches_written, result.docs_chunked, result.total_chunks,
    )
    return 0


def _cmd_chunk_config(args: argparse.Namespace) -> int:
    """E2E composition: extract + chunk in one pass (back-compat path)."""
    from womblex.config import load_config
    from womblex.operations import BatchResult, run_chunking, run_extraction, write_batch_parquet
    from womblex.store.source_provenance import IngestProvenance

    config = load_config(args.config)
    if not config.chunking.enabled:
        logger.warning("Chunking is disabled in config. Set chunking.enabled: true to enable.")
        return 1
    if not check_stage_models(args, (SCOPE_EXTRACT, SCOPE_CHUNK), config=config):
        return 1

    input_root = config.paths.input_root
    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)
        return 1

    try:
        all_files = discover_files(input_root, args.limit)
    except NestedCorpusError as e:
        logger.error(str(e))
        return 1
    if not all_files:
        logger.error("No supported files found in %s", input_root)
        return 1

    output_root = config.paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Extraction stage: %d documents", len(all_files))
    results = run_extraction(all_files, config)

    logger.info("Chunking stage")
    results = run_chunking(results, config)

    batch = BatchResult(results=results)
    write_batch_parquet(
        batch,
        output_root / "documents.parquet",
        provenance=IngestProvenance.from_config(config),
    )

    total_chunks = sum(len(r.chunks) for r in results if r.status == "completed")
    logger.info(
        "Done: %d ok, %d failed, %d total chunks",
        batch.succeeded, batch.failed, total_chunks,
    )
    return 0


# --- manifest ----------------------------------------------------------------


def _register_manifest(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shards", type=Path, required=True,
        help="Shard dir (`<run_id>/documents/`) whose `*._manifest.parquet` "
             "sidecars are consolidated.",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output parquet path (default: `<run_root>/manifest.parquet`).",
    )


def cmd_manifest(args: argparse.Namespace) -> int:
    """Consolidate per-batch manifests into one run-level documents table."""
    from womblex.store.run_manifest import write_run_manifest

    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if not any(shard_dir.glob("*._manifest.parquet")):
        logger.error("--shards directory has no `*._manifest.parquet`: %s", shard_dir)
        return 1
    write_run_manifest(shard_dir, args.output)
    return 0


COMMANDS = [
    Command("run", "Run all pipeline stages from a config file", _register_run, cmd_run),
    Command("extract", "Extract text from a single document", _register_extract, cmd_extract),
    Command("chunk", "Extract and chunk documents", _register_chunk, cmd_chunk),
    Command(
        "manifest", "Consolidate per-batch manifests into a run-level documents table",
        _register_manifest, cmd_manifest,
    ),
]
