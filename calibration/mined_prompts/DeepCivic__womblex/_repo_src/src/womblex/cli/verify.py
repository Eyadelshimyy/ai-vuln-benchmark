"""Verify CLI subcommands: ``verify-shards``, ``verify-evidence`` and ``resolve-source``.

All inspect a finished run without changing it — one audits the shards'
internal integrity, one re-checks each annotated span against the source text
it points at, the other resolves the rows back out to the corpus they were
extracted from.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from womblex.cli._shared import Command

logger = logging.getLogger("womblex")


def _resolve_shard_dir(arg: Path) -> Path | None:
    """Accept either a run root (`output/run-…/`) or a shard dir.

    Returns the path to use, or None if neither shape matches.
    """
    if not arg.is_dir():
        return None
    documents = arg / "documents"
    if documents.is_dir():
        return documents
    # Already a shard dir if any manifest sibling exists
    if any(arg.glob("*._manifest.parquet")):
        return arg
    return None


def _register_verify_shards(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "run_dir", type=Path,
        help="Run root (containing 'documents/') or a shard directory.",
    )
    p.add_argument(
        "--compare-to", type=Path, default=None, action="append",
        help="Additional run/shard dir to diff against. May be repeated.",
    )
    p.add_argument(
        "--input-dir", type=Path, default=None,
        help="Source-file directory; enables source-count comparison.",
    )
    p.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (json only valid for single-run audit).",
    )


def cmd_verify_shards(args: argparse.Namespace) -> int:
    """Audit shard directory integrity; optionally diff against other runs."""
    from womblex.store.shard_audit import (
        audit_shard_directory,
        format_audit_diff,
        format_audit_json,
        format_audit_text,
    )

    primary = _resolve_shard_dir(args.run_dir)
    if primary is None:
        logger.error("not a run or shard dir: %s", args.run_dir)
        return 1

    primary_report = audit_shard_directory(primary, input_dir=args.input_dir)

    if args.compare_to:
        if args.format == "json":
            logger.error("--format json is not supported with --compare-to")
            return 1
        reports = {args.run_dir.name: primary_report}
        for other in args.compare_to:
            other_shards = _resolve_shard_dir(other)
            if other_shards is None:
                logger.error("not a run or shard dir: %s", other)
                return 1
            reports[other.name] = audit_shard_directory(
                other_shards, input_dir=args.input_dir,
            )
        print(format_audit_diff(reports))
    elif args.format == "json":
        print(format_audit_json(primary_report))
    else:
        print(format_audit_text(primary_report))

    corrupted = primary_report.scan.corrupted_batches
    if corrupted:
        logger.warning(
            "verify-shards: %d corrupted batch(es) in %s",
            len(corrupted), primary,
        )
        return 2
    return 0


def _register_resolve_source(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "run_dir", type=Path,
        help="Run root (containing 'documents/' or manifest.parquet) or a shard directory.",
    )
    p.add_argument(
        "--root", type=Path, default=None,
        help="Corpus root to resolve against. Defaults to the manifest's ingest_root; "
             "pass it when the corpus has moved.",
    )
    p.add_argument(
        "--hash", dest="source_hash", default=None,
        help="Resolve one source_hash. Without it, every source_hash in the manifest is resolved.",
    )
    p.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format.",
    )


def cmd_resolve_source(args: argparse.Namespace) -> int:
    """Resolve manifest rows back to their source documents.

    Exit 0 when everything asked for resolved, 2 when any row did not — a
    declined hash basis counts as resolved-as-far-as-it-goes, since the row is
    accounted for rather than missing.
    """
    from womblex.store.source_resolver import RESOLVED, UNSUPPORTED_BASIS, SourceResolver

    try:
        resolver = SourceResolver.for_run(args.run_dir, root=args.root)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    if args.source_hash:
        results = [resolver.resolve(args.source_hash)]
    else:
        results = resolver.resolve_all()

    if args.format == "json":
        print(json.dumps([
            {
                "source_hash": r.source_hash, "status": r.status,
                "hash_basis": r.hash_basis, "detail": r.detail,
                "path": str(r.path) if r.path else None,
                "doc_id": r.doc_id, "source_relpath": r.source_relpath,
            }
            for r in results
        ], indent=2))
    else:
        counts: dict[str, int] = {}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
            if not r.ok or args.source_hash:
                print(f"{r.status:<18} {r.source_hash[:12]}  {r.doc_id or '-'}  "
                      f"{r.path or '-'}  ({r.hash_basis}: {r.detail})")
        if counts:
            print(f"{len(results)} document(s): " + ", ".join(
                f"{n} {status}" for status, n in sorted(counts.items())
            ))
        else:
            print(f"no rows to resolve: the manifest under {args.run_dir} is empty")

    unresolved = [r for r in results if r.status not in (RESOLVED, UNSUPPORTED_BASIS)]
    if unresolved:
        logger.warning("resolve-source: %d row(s) unresolved", len(unresolved))
        return 2
    return 0


def _register_verify_evidence(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "run_dir", type=Path,
        help="Run root (containing 'documents/') or a shard directory.",
    )


def cmd_verify_evidence(args: argparse.Namespace) -> int:
    """Re-check every money, entity-link and PII receipt against the source it points at."""
    from womblex.process.evidence import verify_shards

    shard_dir = _resolve_shard_dir(args.run_dir)
    if shard_dir is None:
        logger.error("not a run root or shard directory: %s", args.run_dir)
        return 1
    report = verify_shards(shard_dir)
    print(
        f"evidence: {report.checked} receipt(s) checked, {report.n_mismatched} fail, "
        f"{report.unchecked} unchecked"
    )
    for line in report.mismatches:
        print(f"  fails: {line}")
    return 0 if report.ok else 2


COMMANDS = [
    Command(
        "verify-shards",
        "Audit shard directory integrity; optionally diff across runs",
        _register_verify_shards,
        cmd_verify_shards,
    ),
    Command(
        "verify-evidence",
        "Re-check annotated spans against the source text they point at",
        _register_verify_evidence,
        cmd_verify_evidence,
    ),
    Command(
        "resolve-source",
        "Resolve a run's manifest rows back to their source documents",
        _register_resolve_source,
        cmd_resolve_source,
    ),
]
