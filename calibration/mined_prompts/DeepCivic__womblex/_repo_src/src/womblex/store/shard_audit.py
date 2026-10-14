"""Directory-level audit of per-batch parquet shards.

Where ``output.verify_shard_persistence`` is a post-write check on a
single batch (with a known ``expected_docs`` count), this module walks an
existing shard directory and reasons across all batches:

- ``scan_shard_directory(shard_dir)`` does the cheap pass — manifest +
  stat() checks per batch, derives ``(batch_id, doc_ids, issues)``.
- ``audit_shard_directory(shard_dir, input_dir=None)`` extends the scan
  with element-level metrics (kind counts, methods, dupe/empty hashes)
  for the ``verify-shards`` CLI and cross-run diffs.
- ``reconcile_checkpoint_with_shards(mgr, shard_dir)`` is the resume-time
  integrity gate: any batch whose shards are corrupted has its doc_ids
  dropped from the checkpoint and its files archived with a ``.corrupt``
  suffix so reads (which glob ``*.elements.parquet`` etc.) skip them.

Two boundary cases are explicitly *not* auto-recovered:

- Manifest unreadable / missing: we can't recover the batch's doc_id
  list, so reconcile logs and leaves the checkpoint untouched. The user
  must re-run without ``--resume``.
- Live runs: a scan during in-flight extraction will false-positive the
  current batch; callers should ensure the run is quiescent.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from womblex.cli._shared import NestedCorpusError, discover_files
from womblex.store.output import (
    _SHARD_ROLES,
    _SHARD_SUFFIX,
    CHUNKS_SUFFIX,
    _shard_paths,
    chunks_path_for,
    read_manifest,
)

logger = logging.getLogger(__name__)

ARCHIVE_SUFFIX = ".corrupt"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchIntegrity:
    """Per-batch integrity result.

    ``doc_ids`` is empty if the manifest itself is unreadable — in that
    case reconcile cannot drop anything and logs the gap.
    """

    batch_id: str
    base_path: Path
    files_present: bool
    files_nonempty: bool
    files_readable: bool
    manifest_consistent: bool
    doc_ids: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def is_healthy(self) -> bool:
        return (
            self.files_present
            and self.files_nonempty
            and self.files_readable
            and self.manifest_consistent
        )


@dataclass(frozen=True)
class ShardScanReport:
    shard_dir: Path
    batches: tuple[BatchIntegrity, ...]

    @property
    def corrupted_batches(self) -> tuple[BatchIntegrity, ...]:
        return tuple(b for b in self.batches if not b.is_healthy)

    @property
    def corrupted_doc_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for b in self.corrupted_batches:
            ids.extend(b.doc_ids)
        return tuple(ids)


@dataclass(frozen=True)
class ShardAuditReport:
    scan: ShardScanReport
    manifest_row_count: int
    status_error_rows: int
    zero_elem_docs: int
    empty_hashes: int
    dupe_hashes: int
    total_elements: int
    methods: Mapping[str, int] = field(default_factory=dict)
    kind_counts: Mapping[str, int] = field(default_factory=dict)
    source_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_dir": str(self.scan.shard_dir),
            "source_count": self.source_count,
            "manifest_row_count": self.manifest_row_count,
            "shard_count": len(self.scan.batches),
            "status_error_rows": self.status_error_rows,
            "zero_elem_docs": self.zero_elem_docs,
            "empty_hashes": self.empty_hashes,
            "dupe_hashes": self.dupe_hashes,
            "total_elements": self.total_elements,
            "methods": dict(self.methods),
            "kind_counts": dict(self.kind_counts),
            "corrupted_batches": [b.batch_id for b in self.scan.corrupted_batches],
        }


# ---------------------------------------------------------------------------
# Scan (cheap — manifest + stat only)
# ---------------------------------------------------------------------------


def _batch_bases(shard_dir: Path) -> list[Path]:
    """Return one ``Path`` per batch, suitable for ``_shard_paths(p)``.

    The legacy single-file shard path is ``batch-NNNN.parquet``; we
    synthesise it from any sibling we find (manifests are guaranteed,
    elements/sidecars may be missing in corrupted batches).
    """
    seen: set[str] = set()
    bases: list[Path] = []
    for role in _SHARD_ROLES:
        for p in shard_dir.glob(f"*{_SHARD_SUFFIX[role]}"):
            stem = p.name[: -len(_SHARD_SUFFIX[role])]
            if stem in seen:
                continue
            seen.add(stem)
            bases.append(shard_dir / f"{stem}.parquet")
    bases.sort(key=lambda p: p.name)
    return bases


def _check_batch(base: Path) -> BatchIntegrity:
    paths = _shard_paths(base)
    issues: list[str] = []
    files_present = True
    files_nonempty = True
    files_readable = True
    manifest_consistent = True
    doc_ids: tuple[str, ...] = ()

    role_ok: dict[str, bool] = {}
    for role, p in paths.items():
        if not p.exists():
            files_present = False
            issues.append(f"missing {role}")
            role_ok[role] = False
            continue
        if p.stat().st_size == 0:
            files_nonempty = False
            issues.append(f"zero-byte {role}")
            role_ok[role] = False
            continue
        try:
            _ = pq.ParquetFile(str(p)).metadata  # smoke-test readability
            role_ok[role] = True
        except Exception as e:
            files_readable = False
            issues.append(f"unreadable {role}: {e}")
            role_ok[role] = False

    # doc_ids depend only on the manifest — read them whenever the manifest
    # itself is healthy, regardless of sibling state. This is what lets
    # reconcile drop checkpoint entries even when only `elements.parquet`
    # is corrupted.
    if role_ok.get("manifest"):
        try:
            m = read_manifest(base)
            doc_ids = tuple(m.column("doc_id").to_pylist())

            # Manifest row counts must match the sidecar row counts they
            # claim — but only check sidecars that are themselves readable,
            # so a missing/corrupt sidecar doesn't double-report.
            if role_ok.get("elements"):
                elem_total = int(sum(m.column("elements_count").to_pylist()))
                actual_elem = pq.ParquetFile(str(paths["elements"])).metadata.num_rows
                if actual_elem != elem_total:
                    manifest_consistent = False
                    issues.append(
                        f"elements row count {actual_elem} != manifest sum {elem_total}"
                    )
            if role_ok.get("table_cells"):
                tc_total = int(sum(m.column("table_cells_count").to_pylist()))
                actual_tc = pq.ParquetFile(str(paths["table_cells"])).metadata.num_rows
                if actual_tc != tc_total:
                    manifest_consistent = False
                    issues.append(
                        f"table_cells row count {actual_tc} != manifest sum {tc_total}"
                    )
            if role_ok.get("form_fields"):
                ff_total = int(sum(m.column("form_fields_count").to_pylist()))
                actual_ff = pq.ParquetFile(str(paths["form_fields"])).metadata.num_rows
                if actual_ff != ff_total:
                    manifest_consistent = False
                    issues.append(
                        f"form_fields row count {actual_ff} != manifest sum {ff_total}"
                    )
        except Exception as e:
            manifest_consistent = False
            issues.append(f"manifest unreadable: {e}")

    return BatchIntegrity(
        batch_id=base.stem,
        base_path=base,
        files_present=files_present,
        files_nonempty=files_nonempty,
        files_readable=files_readable,
        manifest_consistent=manifest_consistent,
        doc_ids=doc_ids,
        issues=tuple(issues),
    )


def scan_shard_directory(shard_dir: Path) -> ShardScanReport:
    """Manifest + stat() integrity check across every batch in ``shard_dir``.

    Returns one ``BatchIntegrity`` per batch found; ``corrupted_batches``
    surfaces those failing any check. No element-level metrics.
    """
    batches = tuple(_check_batch(b) for b in _batch_bases(shard_dir))
    return ShardScanReport(shard_dir=shard_dir, batches=batches)


# ---------------------------------------------------------------------------
# Audit (extends scan with element-level metrics)
# ---------------------------------------------------------------------------


def _count_source_files(input_dir: Path | None) -> int | None:
    """How many documents a run would ingest from *input_dir*.

    The count exists to be compared against the manifest's row count, so it has
    to be the count a run would actually produce. That means the one
    enumeration rule every entry point ingests by (``discover_files``), not a
    second list of extensions that agrees with it until it does not.

    A corpus a run would refuse — documents in subdirectories — has no such
    count, so this reports ``None``, the value the field already carries for
    "not available", rather than a number that would read as drift.
    """
    if input_dir is None or not input_dir.is_dir():
        return None
    try:
        return len(discover_files(input_dir))
    except NestedCorpusError:
        return None


def audit_shard_directory(
    shard_dir: Path, input_dir: Path | None = None,
) -> ShardAuditReport:
    """Cross-batch structural audit + element-level metrics.

    Builds on ``scan_shard_directory`` by loading the elements table once
    for kind counts + total elements. Skips elements load if every batch
    is corrupted (nothing to load).
    """
    scan = scan_shard_directory(shard_dir)

    # Aggregate over healthy batches only — corrupted ones would skew the
    # counts (or fail to read).
    manifest_row_count = 0
    status_error_rows = 0
    zero_elem_docs = 0
    empty_hashes = 0
    methods: Counter[str] = Counter()
    hashes: list[str] = []
    total_elements = 0
    kind_counts: Counter[str] = Counter()

    for b in scan.batches:
        if not b.is_healthy:
            continue
        m = read_manifest(b.base_path)
        manifest_row_count += m.num_rows
        for s in m.column("status").to_pylist():
            if s != "completed":
                status_error_rows += 1
        for c in m.column("elements_count").to_pylist():
            if c == 0:
                zero_elem_docs += 1
        for h in m.column("source_hash").to_pylist():
            if not h:
                empty_hashes += 1
            else:
                hashes.append(h)
        methods.update(m.column("extraction_method").to_pylist())

        elements_path = _shard_paths(b.base_path)["elements"]
        try:
            e = pq.read_table(str(elements_path), columns=["kind"])
            total_elements += e.num_rows
            kind_counts.update(e.column("kind").to_pylist())
        except Exception as exc:  # readable in scan, may still fail on column read
            logger.warning("audit: failed to read elements for %s: %s", b.batch_id, exc)

    dupe_hashes = sum(1 for c in Counter(hashes).values() if c > 1)

    return ShardAuditReport(
        scan=scan,
        manifest_row_count=manifest_row_count,
        status_error_rows=status_error_rows,
        zero_elem_docs=zero_elem_docs,
        empty_hashes=empty_hashes,
        dupe_hashes=dupe_hashes,
        total_elements=total_elements,
        methods=dict(methods),
        kind_counts=dict(kind_counts),
        source_count=_count_source_files(input_dir),
    )


# ---------------------------------------------------------------------------
# Reconcile (resume-time integrity gate)
# ---------------------------------------------------------------------------


def reconcile_checkpoint_with_shards(
    mgr: Any, shard_dir: Path, *, archive_suffix: str = ARCHIVE_SUFFIX,
) -> list[str]:
    """Drop doc_ids whose backing shards are corrupted; archive bad shards.

    ``mgr`` is duck-typed as a ``CheckpointManager`` (avoid a hard import
    cycle). Mutates and saves the checkpoint in place. Returns the
    dropped doc_ids.

    Batches with unreadable manifests are logged but not auto-recovered
    — there's no way to know which doc_ids they covered.
    """
    if not shard_dir.is_dir():
        return []

    scan = scan_shard_directory(shard_dir)
    corrupted = scan.corrupted_batches
    if not corrupted:
        return []

    dropped: list[str] = []
    silent_batches: list[str] = []
    for batch in corrupted:
        if not batch.doc_ids:
            # Manifest itself is bad — we can't enumerate doc_ids.
            silent_batches.append(batch.batch_id)
            logger.error(
                "shard audit: batch %s corrupted with unreadable manifest "
                "(%s); cannot reconcile checkpoint automatically",
                batch.batch_id, "; ".join(batch.issues),
            )
            continue
        logger.warning(
            "shard audit: batch %s corrupted (%s); dropping %d doc(s) from checkpoint",
            batch.batch_id, "; ".join(batch.issues), len(batch.doc_ids),
        )
        for doc_id in batch.doc_ids:
            if doc_id in mgr.state.processed_ids:
                dropped.append(doc_id)
        _archive_batch(batch.base_path, archive_suffix)

    if dropped:
        mgr.drop(dropped)

    if silent_batches:
        # Manifest-corrupt batches leave checkpoint entries dangling. Make
        # this loud so the operator notices and re-runs from scratch.
        logger.error(
            "shard audit: %d batch(es) had unreadable manifests; "
            "checkpoint may reference docs with no recoverable extraction. "
            "Re-run without --resume to rebuild from source.",
            len(silent_batches),
        )

    return dropped


def _archive_batch(base_path: Path, suffix: str) -> None:
    """Rename every sibling shard for ``base_path`` with ``suffix`` so
    the reader globs skip them. Idempotent — re-archiving is a no-op.
    """
    for path in _shard_paths(base_path).values():
        if not path.exists():
            continue
        archived = path.with_name(path.name + suffix)
        if archived.exists():
            # Prior reconcile already archived this batch; leave both files
            # in place for forensics.
            continue
        try:
            path.rename(archived)
        except OSError as e:
            logger.error("shard audit: failed to archive %s -> %s: %s", path, archived, e)


# ---------------------------------------------------------------------------
# Chunks-side audit (independent of element-stream audit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunksBatchIntegrity:
    """Per-batch integrity for a ``*.chunks.parquet`` sidecar.

    ``source_hashes`` is populated whenever the file is readable; an
    unreadable / missing file means we can't enumerate it and the
    caller must consult the elements-side audit to know which docs are
    affected.
    """

    batch_id: str
    chunks_path: Path
    file_present: bool
    file_nonempty: bool
    file_readable: bool
    source_hashes: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def is_healthy(self) -> bool:
        return self.file_present and self.file_nonempty and self.file_readable


def _chunks_bases(shard_dir: Path) -> list[Path]:
    """Discover synthetic ``batch-NNNN.parquet`` bases for chunks sidecars only."""
    bases: list[Path] = []
    for p in sorted(shard_dir.glob(f"*{CHUNKS_SUFFIX}")):
        if p.name.endswith(f"{CHUNKS_SUFFIX}{ARCHIVE_SUFFIX}"):
            continue
        stem = p.name[: -len(CHUNKS_SUFFIX)]
        bases.append(shard_dir / f"{stem}.parquet")
    return bases


def scan_chunks_directory(shard_dir: Path) -> list[ChunksBatchIntegrity]:
    """File-level integrity for every ``*.chunks.parquet`` in ``shard_dir``.

    Independent of the element-stream audit — chunks may be present
    without their producing elements still being clean, and vice
    versa. A corrupt chunks file only justifies dropping chunk-stage
    checkpoint entries.
    """
    out: list[ChunksBatchIntegrity] = []
    for base in _chunks_bases(shard_dir):
        target = chunks_path_for(base)
        issues: list[str] = []
        file_present = target.exists()
        file_nonempty = file_present and target.stat().st_size > 0
        file_readable = False
        source_hashes: tuple[str, ...] = ()
        if not file_present:
            issues.append("missing chunks")
        elif not file_nonempty:
            issues.append("zero-byte chunks")
        else:
            try:
                _ = pq.ParquetFile(str(target)).metadata  # smoke-test readability
                file_readable = True
                src_col = pq.read_table(str(target), columns=["source_hash"]).column("source_hash")
                source_hashes = tuple(src_col.to_pylist())
            except Exception as e:
                issues.append(f"unreadable chunks: {e}")
        out.append(ChunksBatchIntegrity(
            batch_id=base.stem,
            chunks_path=target,
            file_present=file_present,
            file_nonempty=file_nonempty,
            file_readable=file_readable,
            source_hashes=source_hashes,
            issues=tuple(issues),
        ))
    return out


@dataclass
class SidecarBatchIntegrity:
    """File-level integrity of one per-batch sidecar (any suffix).

    Generic counterpart to :class:`ChunksBatchIntegrity` — used to
    self-heal any ``CheckpointManager``-backed downstream stage
    (chunk / enrich / embed / link) on resume.
    """

    batch_id: str
    sidecar_path: Path
    file_present: bool
    file_nonempty: bool
    file_readable: bool
    issues: tuple[str, ...]

    @property
    def is_healthy(self) -> bool:
        return self.file_present and self.file_nonempty and self.file_readable


def scan_sidecar_directory(shard_dir: Path, suffix: str) -> list[SidecarBatchIntegrity]:
    """File-level integrity for every ``*<suffix>`` sidecar in ``shard_dir``.

    ``suffix`` is a per-stage sidecar suffix (e.g. ``.enrichment_entities.parquet``,
    ``.embeddings.parquet``, ``.entity_links.parquet``, ``.chunks.parquet``).
    Checks presence / non-emptiness / parquet-readability only — enough to
    decide whether a resume can trust the sidecar or must re-do its batch.
    """
    out: list[SidecarBatchIntegrity] = []
    for p in sorted(shard_dir.glob(f"*{suffix}")):
        if p.name.endswith(f"{suffix}{ARCHIVE_SUFFIX}"):
            continue
        issues: list[str] = []
        present = p.exists()
        nonempty = present and p.stat().st_size > 0
        readable = False
        if not nonempty:
            issues.append(f"zero-byte {suffix}")
        else:
            try:
                _ = pq.ParquetFile(str(p)).metadata  # smoke-test readability
                readable = True
            except Exception as e:
                issues.append(f"unreadable {suffix}: {e}")
        out.append(SidecarBatchIntegrity(
            batch_id=p.name[: -len(suffix)],
            sidecar_path=p,
            file_present=present,
            file_nonempty=nonempty,
            file_readable=readable,
            issues=tuple(issues),
        ))
    return out


def reconcile_stage_checkpoint_with_shards(
    mgr: Any, shard_dir: Path, *, suffix: str, archive_suffix: str = ARCHIVE_SUFFIX,
) -> list[str]:
    """Generic resume-time self-heal for a downstream-stage sidecar.

    Drops checkpoint entries for any batch whose ``*<suffix>`` sidecar is
    corrupt (missing/zero-byte/unreadable) and archives the bad file so the
    next pass re-does that batch. Doc ids come from the batch's manifest.
    The element stream is never touched — a corrupt stage sidecar only
    invalidates that stage. This is the shared engine behind the
    chunk/enrich/embed/link resume scans.
    """
    if not shard_dir.is_dir():
        return []

    bad = [b for b in scan_sidecar_directory(shard_dir, suffix) if not b.is_healthy]
    if not bad:
        return []

    dropped: list[str] = []
    for batch in bad:
        try:
            manifest = read_manifest(batch.sidecar_path.parent / f"{batch.batch_id}.parquet")
            doc_ids = manifest.column("doc_id").to_pylist()
        except Exception:
            doc_ids = []

        if doc_ids:
            logger.warning(
                "%s audit: batch %s corrupted (%s); dropping %d stage doc(s)",
                suffix, batch.batch_id, "; ".join(batch.issues), len(doc_ids),
            )
            dropped.extend(d for d in doc_ids if d in mgr.state.processed_ids)
        else:
            logger.error(
                "%s audit: batch %s corrupted (%s) and manifest also unreadable; "
                "cannot reconcile checkpoint automatically",
                suffix, batch.batch_id, "; ".join(batch.issues),
            )

        if batch.file_present:
            archived = batch.sidecar_path.with_name(batch.sidecar_path.name + archive_suffix)
            if not archived.exists():
                try:
                    batch.sidecar_path.rename(archived)
                except OSError as e:
                    logger.error(
                        "%s audit: failed to archive %s -> %s: %s",
                        suffix, batch.sidecar_path, archived, e,
                    )

    if dropped:
        mgr.drop(dropped)
    return dropped


def reconcile_chunk_checkpoint_with_shards(
    mgr: Any, shard_dir: Path, *, archive_suffix: str = ARCHIVE_SUFFIX,
) -> list[str]:
    """Drop chunk-stage checkpoint entries for batches with corrupt chunks files.

    Thin wrapper over :func:`reconcile_stage_checkpoint_with_shards` for the
    ``*.chunks.parquet`` sidecar. The element-stream is left untouched;
    elements-side corruption is handled by
    :func:`reconcile_checkpoint_with_shards`.
    """
    return reconcile_stage_checkpoint_with_shards(
        mgr, shard_dir, suffix=CHUNKS_SUFFIX, archive_suffix=archive_suffix,
    )


# ---------------------------------------------------------------------------
# Formatters (used by the verify-shards CLI)
# ---------------------------------------------------------------------------


_SCALAR_KEYS = (
    "source_count",
    "manifest_row_count",
    "shard_count",
    "status_error_rows",
    "zero_elem_docs",
    "empty_hashes",
    "dupe_hashes",
    "total_elements",
)


def format_audit_text(report: ShardAuditReport) -> str:
    d = report.as_dict()
    lines = [f"=== {d['shard_dir']} ==="]
    for k in _SCALAR_KEYS:
        v = d.get(k)
        if v is None:
            continue
        lines.append(f"  {k:<20} {v}")
    if d["corrupted_batches"]:
        lines.append(f"  corrupted_batches    {len(d['corrupted_batches'])}")
        for bid in d["corrupted_batches"]:
            batch = next(b for b in report.scan.batches if b.batch_id == bid)
            lines.append(f"    - {bid}: {'; '.join(batch.issues)}")
    if d["methods"]:
        lines.append("  methods:")
        for m, c in sorted(d["methods"].items()):
            lines.append(f"    {m:<30} {c}")
    if d["kind_counts"]:
        lines.append("  kinds:")
        for k, c in sorted(d["kind_counts"].items()):
            lines.append(f"    {k:<30} {c}")
    return "\n".join(lines) + "\n"


def format_audit_diff(reports: Mapping[str, ShardAuditReport]) -> str:
    """Side-by-side comparison of two or more audits.

    Mirrors the ``cmp_table`` shape from the original investigation
    script: scalar metrics, then kind counts and methods stacked.
    """
    labels = list(reports.keys())
    out: list[str] = [f"=== cross-run comparison ({len(labels)} runs) ===", ""]

    header = f"{'metric':<22}" + "".join(f"{L:>22}" for L in labels)
    out.append(header)
    for k in _SCALAR_KEYS:
        row = f"{k:<22}"
        for L in labels:
            v = reports[L].as_dict().get(k)
            row += f"{('-' if v is None else v)!s:>22}"
        out.append(row)
    out.append("")

    all_kinds: set[str] = set()
    for r in reports.values():
        all_kinds.update(r.kind_counts)
    if all_kinds:
        out.append(f"{'kind':<22}" + "".join(f"{L:>22}" for L in labels))
        for k in sorted(all_kinds):
            row = f"{k:<22}"
            for L in labels:
                row += f"{reports[L].kind_counts.get(k, 0)!s:>22}"
            out.append(row)
        out.append("")

    all_methods: set[str] = set()
    for r in reports.values():
        all_methods.update(r.methods)
    if all_methods:
        out.append(f"{'method':<22}" + "".join(f"{L:>22}" for L in labels))
        for m in sorted(all_methods):
            row = f"{m:<22}"
            for L in labels:
                row += f"{reports[L].methods.get(m, 0)!s:>22}"
            out.append(row)
    return "\n".join(out) + "\n"


def format_audit_json(report: ShardAuditReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)
