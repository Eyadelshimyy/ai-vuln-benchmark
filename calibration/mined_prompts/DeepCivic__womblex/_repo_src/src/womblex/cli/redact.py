"""Redaction CLI subcommands: ``redact`` (per-stage ``--shards`` over an
existing shard dir, or E2E ``--config`` extract+redact),
``annotate-redactions`` (back-compat alias for the per-stage path),
``validate-redactions`` (detector sanity check vs a labels packet)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from womblex.cli._shared import (
    Command,
    NestedCorpusError,
    add_models_check_argument,
    check_stage_models,
    discover_files,
)
from womblex.utils.model_check import SCOPE_EXTRACT, SCOPE_REDACT

logger = logging.getLogger("womblex")


# --- redact (dual-mode: --shards per-stage | --config E2E) ------------------


def _register_redact(p: argparse.ArgumentParser) -> None:
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--shards", type=Path,
        help=(
            "Per-stage mode: detect redactions over an existing shard directory "
            "(``<run_id>/documents/`` containing ``*.elements.parquet`` + "
            "``*._manifest.parquet``). Writes ``*.redactions.parquet`` siblings. "
            "Requires --pdfs (detection rasterises the source pages)."
        ),
    )
    mode.add_argument(
        "--config", type=Path,
        help="E2E composition mode: run extraction + redaction from config YAML.",
    )
    p.add_argument(
        "--pdfs", type=Path, default=None,
        help="Source PDF directory, resolved via manifest filename. Required with --shards.",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory for *.redactions.parquet sidecars (default: same as --shards). --shards only.",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="JSON checkpoint path for resumable runs (skips already-processed batches). --shards only.",
    )
    p.add_argument("--dpi", type=int, default=150, help="Page render DPI for raster fallback. --shards only.")
    p.add_argument(
        "--max-area-ratio", type=float, default=0.05,
        help="Reject candidate regions larger than this fraction of the page. --shards only.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max documents to process. --config only.")
    add_models_check_argument(p)


def cmd_redact(args: argparse.Namespace) -> int:
    """Detect redactions over a shard directory (per-stage) or a config-described corpus (E2E)."""
    if args.shards is not None:
        return _cmd_redact_shards(args)
    return _cmd_redact_config(args)


def _cmd_redact_shards(args: argparse.Namespace) -> int:
    shard_dir: Path = args.shards
    if not shard_dir.is_dir():
        logger.error("--shards path is not a directory: %s", shard_dir)
        return 1
    if args.pdfs is None:
        logger.error("--pdfs is required with --shards (detection rasterises source pages)")
        return 1
    if not args.pdfs.is_dir():
        logger.error("--pdfs path is not a directory: %s", args.pdfs)
        return 1
    if not any(shard_dir.glob("*._manifest.parquet")):
        logger.error("--shards directory has no `*._manifest.parquet`: %s", shard_dir)
        return 1
    return _run_redact_shards(
        shard_dir=shard_dir,
        pdf_dir=args.pdfs,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        dpi=args.dpi,
        max_area_ratio=args.max_area_ratio,
        args=args,
    )


def _run_redact_shards(
    shard_dir: Path,
    pdf_dir: Path,
    output_dir: Path | None,
    checkpoint_path: Path | None,
    dpi: int,
    max_area_ratio: float,
    args: argparse.Namespace,
) -> int:
    """Shared per-stage path used by ``redact --shards`` and the
    ``annotate-redactions`` back-compat alias."""
    from womblex.config import RedactionConfig
    from womblex.redact.batch import annotate_redactions_for_shards

    config = RedactionConfig(dpi=dpi, max_area_ratio=max_area_ratio)
    if not check_stage_models(args, (SCOPE_REDACT,), redaction=config):
        return 1
    summary = annotate_redactions_for_shards(
        shard_dir=shard_dir,
        pdf_dir=pdf_dir,
        config=config,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )
    n_with = sum(1 for v in summary.values() if v > 0)
    total = sum(summary.values())
    logger.info(
        "annotated %d docs, %d with redactions, %d total regions",
        len(summary), n_with, total,
    )
    return 0


def _cmd_redact_config(args: argparse.Namespace) -> int:
    """E2E composition: extract + redact in-memory from a config YAML."""
    from womblex.config import load_config
    from womblex.operations import BatchResult, run_extraction, run_redaction, write_batch_parquet
    from womblex.store.source_provenance import IngestProvenance

    config = load_config(args.config)
    if not config.redaction.enabled:
        logger.warning("Redaction is disabled in config. Set redaction.enabled: true to enable.")
        return 1
    if not check_stage_models(args, (SCOPE_EXTRACT,), config=config):
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

    logger.info("Redaction stage: mode=%s", config.redaction.mode)
    results = run_redaction(results, config)

    batch = BatchResult(results=results)
    write_batch_parquet(
        batch,
        output_root / "documents.parquet",
        provenance=IngestProvenance.from_config(config),
    )

    redacted_count = sum(
        bool(
            r.status == "completed"
            and r.extraction
            and r.extraction.redaction_report
            and r.extraction.redaction_report.total > 0
        )
        for r in results
    )
    logger.info(
        "Done: %d ok, %d failed, %d documents with redactions",
        batch.succeeded, batch.failed, redacted_count,
    )
    return 0


# --- annotate-redactions (back-compat alias for `redact --shards`) ----------


def _register_annotate_redactions(p: argparse.ArgumentParser) -> None:
    p.add_argument("shards", type=Path, help="Directory containing extracted *.elements.parquet + *._manifest.parquet")
    p.add_argument("pdfs", type=Path, help="Directory containing the source PDFs (resolved via manifest filename)")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory for *.redactions.parquet sidecars (default: same as shards)",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="JSON checkpoint path for resumable runs (skips already-processed batches)",
    )
    p.add_argument("--dpi", type=int, default=150, help="Page render DPI for raster fallback")
    p.add_argument(
        "--max-area-ratio", type=float, default=0.05,
        help="Reject candidate regions larger than this fraction of the page",
    )
    add_models_check_argument(p)


def cmd_annotate_redactions(args: argparse.Namespace) -> int:
    """Deprecated alias for ``womblex redact --shards <dir> --pdfs <dir>``.

    Retained so existing scripts keep working; new callers should prefer the
    per-stage ``redact --shards`` surface. Both route through the same engine.
    """
    if not args.shards.is_dir():
        logger.error("shards dir not found: %s", args.shards)
        return 1
    if not args.pdfs.is_dir():
        logger.error("pdfs dir not found: %s", args.pdfs)
        return 1
    return _run_redact_shards(
        shard_dir=args.shards,
        pdf_dir=args.pdfs,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        dpi=args.dpi,
        max_area_ratio=args.max_area_ratio,
        args=args,
    )


# --- validate-redactions (detector sanity check vs labels packet) -----------


def _register_validate_redactions(p: argparse.ArgumentParser) -> None:
    p.add_argument("--labels", type=Path, required=True, help="Labels packet directory (*.meta.json files)")
    p.add_argument("--pdfs", type=Path, required=True, help="Source PDF directory")
    p.add_argument(
        "--report", type=Path, default=None,
        help="JSON output path (else markdown table printed to stdout)",
    )
    p.add_argument("--dpi", type=int, default=150, help="Page render DPI for raster fallback")
    p.add_argument(
        "--max-area-ratio", type=float, default=0.05,
        help="Reject candidate regions larger than this fraction of the page",
    )


def cmd_validate_redactions(args: argparse.Namespace) -> int:
    """Run detection over PDFs in a labels packet; print or save per-doc summaries."""
    from womblex.config import RedactionConfig
    from womblex.redact.batch import validate_redactions_against_labels

    if not args.labels.is_dir():
        logger.error("labels dir not found: %s", args.labels)
        return 1
    if not args.pdfs.is_dir():
        logger.error("pdfs dir not found: %s", args.pdfs)
        return 1

    config = RedactionConfig(dpi=args.dpi, max_area_ratio=args.max_area_ratio)
    summaries = validate_redactions_against_labels(args.labels, args.pdfs, config)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            [
                {
                    "source_pdf": s.source_pdf,
                    "n_pages": s.n_pages,
                    "total_regions": s.total_regions,
                    "affected_pages": s.affected_pages,
                    "labelled_pages": s.labelled_pages,
                    "per_page_bboxes": s.per_page_bboxes,
                }
                for s in summaries
            ],
            indent=2,
            default=str,
        ))
        logger.info("wrote: %s", args.report)
    else:
        print("| source pdf | pages | regions | affected pages | labelled pages |")
        print("|---|---:|---:|---|---|")
        for s in summaries:
            short = s.source_pdf[:60]
            print(
                f"| {short} | {s.n_pages} | {s.total_regions} | "
                f"{s.affected_pages} | {s.labelled_pages} |"
            )

    logger.info("validated %d docs", len(summaries))
    return 0


COMMANDS = [
    Command(
        "redact",
        "Detect redactions over a shard dir (--shards) or extract+redact a corpus (--config)",
        _register_redact,
        cmd_redact,
    ),
    Command(
        "annotate-redactions",
        "Deprecated alias for `redact --shards <dir> --pdfs <dir>`",
        _register_annotate_redactions,
        cmd_annotate_redactions,
    ),
    Command(
        "validate-redactions",
        "Validate the redaction detector against a labels packet",
        _register_validate_redactions,
        cmd_validate_redactions,
    ),
]
