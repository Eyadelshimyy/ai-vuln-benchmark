"""Parquet output writer for extraction results.

Per-batch output is split across four sibling parquet files:

- ``batch-NNNN.elements.parquet`` — one row per element (the canonical
  structural stream)
- ``batch-NNNN.table_cells.parquet`` — children of ``kind='table'``
  elements, one row per cell
- ``batch-NNNN.form_fields.parquet`` — children of ``kind='form'``
  elements, one row per field
- ``batch-NNNN._manifest.parquet`` — one row per source file

Sidecars are joinable via ``(source_hash, parent_elem_order)``. Text is
verbatim from extraction; no post-processing is applied at the schema
boundary.

The caller passes ``output_path`` as ``batch-NNNN.parquet``; the writer
derives the four sibling paths from its stem.

Per-batch chunk output (stage downstream of extraction) lives in a fifth
sibling parquet:

- ``batch-NNNN.chunks.parquet`` — one row per chunk

Chunks join back to elements via ``source_hash`` plus offset-range overlap
with the reassembled element text — not via ``elem_order``, since a chunk
straddles multiple elements. ``page_start`` / ``page_end`` are nullable
for sources without page semantics (DOCX, spreadsheets).

The one exception is ``elem_order``, which is populated **only** for table
chunks (a table chunk comes from exactly one element, so the anchor is
well-defined). It is the document-order anchor for the narrative ↔ table
order the two disjoint chunk projections otherwise lose. Null for narrative
chunks and for spreadsheet sheets.

The anchor and a narrative ``start_char`` are in different coordinate
spaces, so ordering the two together needs the elements as well:
``chunker.element_spans`` maps each element to its narrative offsets, and
``chunker.chunks_in_document_order(rows, spans)`` does the interleave.
Read the elements under the same ``text_source`` overlay the chunks were
written under — a different overlay is a different coordinate space.

``token_count`` is the chunk's length in the units the chunker budgeted
with — its own token counter, applied to the final chunk text. It earns a
column because it cannot be recovered from the row: character and word
counts follow from ``text``, a token count needs the tokeniser. It is
**exact at or below the budget and a floor above it** — semchunk's counter
short-circuits over-budget text to ``chunk_size + 1``, and only a
redaction-repaired chunk is ever over budget (see ``_count_tokens`` in
``process/chunker.py``). Null for shards written before the column existed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.ingest.extract import ExtractionResult
from womblex.store.content_digest import content_digest
from womblex.store.contract import contract_footer
from womblex.store.run_stamp import RunStamp, sidecar_footer
from womblex.store.source_provenance import IngestProvenance

logger = logging.getLogger(__name__)

PARSER_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_BBOX_TYPE = pa.struct([
    ("x", pa.float32()),
    ("y", pa.float32()),
    ("width", pa.float32()),
    ("height", pa.float32()),
])

ELEMENT_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("collection_id", pa.string()),
    ("elem_order", pa.int32()),
    ("kind", pa.string()),
    ("extractor", pa.string()),
    ("confidence", pa.float32()),
    ("page", pa.int32()),
    ("bbox", _BBOX_TYPE),
    ("text", pa.string()),
    ("alt_text", pa.string()),
    ("header_rows", pa.list_(pa.int32())),
    ("sheet", pa.string()),
    ("row", pa.int32()),
    ("col", pa.int32()),
    ("value", pa.string()),
    ("value_type", pa.string()),
    ("formula", pa.string()),
    ("number_format", pa.string()),
    ("merge_range", pa.string()),
    ("meta", pa.map_(pa.string(), pa.string())),
])

TABLE_CELLS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("parent_elem_order", pa.int32()),
    ("row", pa.int32()),
    ("col", pa.int32()),
    ("value", pa.string()),
    ("rowspan", pa.int32()),
    ("colspan", pa.int32()),
    ("value_type", pa.string()),
    ("bbox", _BBOX_TYPE),
])

FORM_FIELDS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("parent_elem_order", pa.int32()),
    ("field_index", pa.int32()),
    ("name", pa.string()),
    ("value", pa.string()),
    ("field_type", pa.string()),
])

MANIFEST_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("collection_id", pa.string()),
    ("doc_id", pa.string()),
    ("ingest_root", pa.string()),
    ("source_relpath", pa.string()),
    ("filename", pa.string()),
    ("ext", pa.string()),
    ("extraction_method", pa.string()),
    ("elements_count", pa.int64()),
    ("table_cells_count", pa.int64()),
    ("form_fields_count", pa.int64()),
    ("status", pa.string()),
    ("error", pa.string()),
    ("extracted_at_iso", pa.string()),
    ("parser_version", pa.string()),
    ("content_digest", pa.string()),
])

# Manifest columns added after parser 2.0. Older shards back-fill on read
# rather than failing — the compat convention `_CHUNKS_BACKFILL` states, but
# with "" where chunks use nulls: "" is exactly what a writer that declared no
# root emits, so "written before the columns" and "no root declared" read the
# same, which is the truth in both cases. `doc_id` keeps its own derivation
# (from `filename`) in `_read_shard` because a better value than "" exists.
_MANIFEST_BACKFILL: tuple[str, ...] = ("ingest_root", "source_relpath")

# `content_digest` back-fills as null, not "": a shard written before it has no
# digest to compare, which is unknown rather than an empty document's digest.
_MANIFEST_NULL_BACKFILL: tuple[str, ...] = ("content_digest",)

# `table_cells.bbox` (contract 1.3) back-fills as null: a cell written before it
# has no recorded geometry.
_TABLE_CELLS_NULL_BACKFILL: tuple[str, ...] = ("bbox",)

# Per role: (columns back-filled with "", columns back-filled with null).
_BACKFILL: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "manifest": (_MANIFEST_BACKFILL, _MANIFEST_NULL_BACKFILL),
    "table_cells": ((), _TABLE_CELLS_NULL_BACKFILL),
}

CHUNKS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("chunk_index", pa.int32()),
    ("text", pa.string()),
    ("start_char", pa.int32()),
    ("end_char", pa.int32()),
    ("content_type", pa.string()),
    ("has_redaction", pa.bool_()),
    ("page_start", pa.int32()),
    ("page_end", pa.int32()),
    ("elem_order", pa.int32()),
    ("token_count", pa.int32()),
])

# Columns added to CHUNKS_SCHEMA after parser 2.0. Shards written before them
# are back-filled with nulls on read instead of failing — the compat shim
# `_read_chunks_shard` asks for.
_CHUNKS_BACKFILL: tuple[str, ...] = ("elem_order", "token_count")


ELEMENTS_SUFFIX = ".elements.parquet"

_SHARD_ROLES = ("elements", "table_cells", "form_fields", "manifest")
_SHARD_SUFFIX = {
    "elements": ELEMENTS_SUFFIX,
    "table_cells": ".table_cells.parquet",
    "form_fields": ".form_fields.parquet",
    "manifest": "._manifest.parquet",
}
_SHARD_SCHEMA = {
    "elements": ELEMENT_SCHEMA,
    "table_cells": TABLE_CELLS_SCHEMA,
    "form_fields": FORM_FIELDS_SCHEMA,
    "manifest": MANIFEST_SCHEMA,
}

CHUNKS_SUFFIX = ".chunks.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shard_paths(output_path: Path) -> dict[str, Path]:
    """Map shard role → sibling parquet path derived from ``output_path``.

    ``output_path`` is the legacy single-file path (e.g.
    ``batch-0001.parquet``). Sibling sidecar files share its stem.
    """
    base = output_path.parent / output_path.stem
    # base is the caller's own output path; the only interpolated segment is a
    # module-constant suffix, so there is no caller-supplied path component here.
    # nosemgrep: runwai-python-path-traversal-sink -- suffix is a module constant
    return {role: Path(f"{base}{_SHARD_SUFFIX[role]}") for role in _SHARD_ROLES}


def _source_hash(source_path: str) -> str:
    """SHA-256 of the source bytes. Returns '' if the file is unreadable."""
    p = Path(source_path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bbox_dict(b: Any) -> dict[str, float] | None:
    if b is None:
        return None
    return {"x": float(b.x), "y": float(b.y), "width": float(b.width), "height": float(b.height)}


def _meta_pairs(meta: dict[str, str] | None) -> list[tuple[str, str]]:
    if not meta:
        return []
    return [(str(k), str(v)) for k, v in meta.items()]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_results(
    results: list[tuple[str, str, ExtractionResult]],
    output_path: Path,
    *,
    collection_id: str = "",
    provenance: IngestProvenance | None = None,
    stamp: RunStamp | None = None,
) -> Path:
    """Write a batch's results to sibling element + sidecar + manifest parquets.

    ``output_path`` is the legacy shard path (``batch-NNNN.parquet``);
    the four sibling files are written next to it sharing its stem.
    Returns ``output_path`` for caller back-compat.

    ``provenance`` declares where the documents were read from: it supplies
    the ``ingest_root`` / ``source_relpath`` manifest columns, the
    ``collection_id`` (overriding the bare keyword), and the namespaced footer
    key-value metadata stamped onto all four files. Without it the three
    columns are empty and no footer is written — the shape a caller that has
    not declared a root gets, never a root guessed from the process's cwd.

    ``stamp`` names the run that produced the shard — run id, version,
    configuration digest and stage — in the same footer, so a shard copied out
    of its run directory still says which run it belongs to. Supplied by the
    caller for the same reason ``provenance`` is: only the caller knows the run
    id. Omitted, no run keys are written.
    """
    paths = _shard_paths(output_path)
    if provenance is not None:
        collection_id = provenance.collection_id
    elements_rows: list[dict] = []
    table_cells_rows: list[dict] = []
    form_fields_rows: list[dict] = []
    manifest_rows: list[dict] = []
    extracted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    relpaths: list[str] = []
    hashes: list[str] = []

    for doc_id, source_path, res in results:
        src_hash = _source_hash(source_path)
        hashes.append(src_hash)
        relpath = provenance.relpath_for(source_path) if provenance is not None else ""
        relpaths.append(relpath)
        tc_count = 0
        ff_count = 0

        for e in res.elements:
            elements_rows.append({
                "source_hash": src_hash,
                "collection_id": collection_id,
                "elem_order": e.order,
                "kind": e.kind,
                "extractor": e.extractor,
                "confidence": float(e.confidence),
                "page": e.page,
                "bbox": _bbox_dict(e.bbox),
                "text": e.text,
                "alt_text": e.alt_text,
                "header_rows": e.header_rows,
                "sheet": e.sheet,
                "row": e.row,
                "col": e.col,
                "value": e.value,
                "value_type": e.value_type,
                "formula": e.formula,
                "number_format": e.number_format,
                "merge_range": e.merge_range,
                "meta": _meta_pairs(e.meta),
            })
            if e.kind == "table" and e.cells:
                for c in e.cells:
                    table_cells_rows.append({
                        "source_hash": src_hash,
                        "parent_elem_order": e.order,
                        "row": c.row,
                        "col": c.col,
                        "value": c.value,
                        "rowspan": c.rowspan,
                        "colspan": c.colspan,
                        "value_type": c.value_type,
                        "bbox": _bbox_dict(c.bbox),
                    })
                    tc_count += 1
            elif e.kind == "form" and e.fields:
                for i, f in enumerate(e.fields):
                    form_fields_rows.append({
                        "source_hash": src_hash,
                        "parent_elem_order": e.order,
                        "field_index": i,
                        "name": f.name,
                        "value": f.value,
                        "field_type": f.field_type,
                    })
                    ff_count += 1

        manifest_rows.append({
            "source_hash": src_hash,
            "collection_id": collection_id,
            "doc_id": doc_id,
            "ingest_root": provenance.ingest_root if provenance is not None else "",
            "source_relpath": relpath,
            "filename": Path(source_path).name,
            "ext": Path(source_path).suffix.lower(),
            "extraction_method": res.method,
            "elements_count": len(res.elements),
            "table_cells_count": tc_count,
            "form_fields_count": ff_count,
            "status": "error" if res.error else "completed",
            "error": res.error or "",
            "extracted_at_iso": extracted_at,
            "parser_version": PARSER_VERSION,
            "content_digest": content_digest(res.elements),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    footer = _merge_footers(
        provenance.footer_metadata(relpaths) if provenance is not None else None,
        stamp.footer_metadata() if stamp is not None else None,
    )
    outcome = next((res.layout for _, _, res in results if res.layout is not None), None)
    elements_footer = (
        _merge_footers(footer, outcome.fingerprint.footer_metadata()) if outcome else footer
    )
    _write_rows(elements_rows, paths["elements"], ELEMENT_SCHEMA, metadata=elements_footer)
    _write_rows(table_cells_rows, paths["table_cells"], TABLE_CELLS_SCHEMA, metadata=footer)
    _write_rows(form_fields_rows, paths["form_fields"], FORM_FIELDS_SCHEMA, metadata=footer)
    _write_rows(manifest_rows, paths["manifest"], MANIFEST_SCHEMA, metadata=footer)
    if outcome is not None:
        from womblex.store.layout_output import layout_rows, write_layout_regions

        write_layout_regions(
            layout_rows(
                (src, res.layout.pages) for src, (_, _, res) in zip(hashes, results, strict=True)
                if res.layout is not None
            ),
            output_path, outcome.fingerprint,
            redaction_consumed=outcome.redaction_consumed, metadata=footer,
        )

    logger.info(
        "Wrote shard %s: docs=%d elements=%d table_cells=%d form_fields=%d",
        output_path.stem,
        len(manifest_rows), len(elements_rows),
        len(table_cells_rows), len(form_fields_rows),
    )
    return output_path


def _merge_footers(*parts: dict[bytes, bytes] | None) -> dict[bytes, bytes] | None:
    """Combine the namespaced footer blocks a writer has; ``None`` when it has none."""
    merged: dict[bytes, bytes] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged or None


def _write_rows(
    rows: list[dict],
    path: Path,
    schema: pa.Schema,
    *,
    metadata: dict[bytes, bytes] | None = None,
    role: str | None = None,
) -> None:
    schema = schema.with_metadata({**(metadata or {}), **contract_footer(path, role=role)})
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    pq.write_table(table, str(path), compression="zstd", compression_level=3)


# ---------------------------------------------------------------------------
# Chunk-stage writer / reader (sibling parquet to the extraction shard)
# ---------------------------------------------------------------------------


def chunks_path_for(base_path: Path) -> Path:
    """Return ``<base>.chunks.parquet`` sibling for a shard base path.

    ``base_path`` is the legacy single-file shard path (e.g.
    ``batch-0001.parquet``); the returned chunk sidecar shares its stem.
    """
    return base_path.parent / f"{base_path.stem}{CHUNKS_SUFFIX}"


def write_chunks(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's chunk rows to ``batch-NNNN.chunks.parquet``.

    ``rows`` must match :data:`CHUNKS_SCHEMA`. Empty input produces an
    empty-but-schema-correct file so downstream readers can glob safely.
    """
    target = chunks_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, CHUNKS_SCHEMA, metadata=sidecar_footer(output_path, "chunk"))
    logger.info("Wrote chunks shard %s: rows=%d", target.name, len(rows))
    return target


def read_chunks(path: Path) -> pa.Table:
    """Read chunks from a single shard file or a shard-directory glob."""
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{CHUNKS_SUFFIX}"))
        if not shards:
            return pa.table(
                {f.name: pa.array([], type=f.type) for f in CHUNKS_SCHEMA},
                schema=CHUNKS_SCHEMA,
            )
        return pa.concat_tables([_read_chunks_shard(s) for s in shards])
    chunks_p = p if p.name.endswith(CHUNKS_SUFFIX) else chunks_path_for(p)
    return _read_chunks_shard(chunks_p)


def _read_chunks_shard(path: Path) -> pa.Table:
    raw = pq.read_table(str(path))
    missing = [f.name for f in CHUNKS_SCHEMA if f.name not in raw.schema.names]
    hard = [n for n in missing if n not in _CHUNKS_BACKFILL]
    if hard:
        raise ValueError(
            f"chunks shard {path} missing columns {hard}; schema bump without compat shim?"
        )
    for name in missing:
        fld = CHUNKS_SCHEMA.field(name)
        raw = raw.append_column(fld, pa.nulls(raw.num_rows, type=fld.type))
    return raw.select([f.name for f in CHUNKS_SCHEMA]).cast(CHUNKS_SCHEMA)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_elements(path: Path) -> pa.Table:
    """Read elements from a single shard file or a shard-directory glob."""
    return _read_role(path, role="elements")


def read_table_cells(path: Path) -> pa.Table:
    return _read_role(path, role="table_cells")


def read_form_fields(path: Path) -> pa.Table:
    return _read_role(path, role="form_fields")


def read_manifest(path: Path) -> pa.Table:
    return _read_role(path, role="manifest")


def _read_role(path: Path, *, role: str) -> pa.Table:
    schema = _SHARD_SCHEMA[role]
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{_SHARD_SUFFIX[role]}"))
        if not shards:
            return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
        return pa.concat_tables([_read_shard(s, role) for s in shards])
    # Single-file path: interpret as shard base (drop .parquet, append role suffix)
    role_path = p.parent / f"{p.stem}{_SHARD_SUFFIX[role]}"
    return _read_shard(role_path, role)


def _read_shard(shard_path: Path, role: str) -> pa.Table:
    """Read one shard parquet, tolerating older files missing newer columns.

    The manifest schema gained ``doc_id`` after initial release; for runs
    written before that, derive it from ``Path(filename).stem`` on read so
    existing checkpoints can still be reconciled. It later gained
    ``ingest_root`` / ``source_relpath``, which have no derivable value — a
    shard written before them says nothing about the root it was read from —
    so those back-fill as empty strings (``_MANIFEST_BACKFILL``); ``content_digest``
    and ``table_cells.bbox`` back-fill as null (``_BACKFILL``).
    """
    schema = _SHARD_SCHEMA[role]
    raw = pq.read_table(str(shard_path))
    missing = [f.name for f in schema if f.name not in raw.schema.names]
    if not missing:
        return raw.select([f.name for f in schema]).cast(schema)
    if role == "manifest" and "doc_id" in missing:
        filenames = raw.column("filename").to_pylist()
        derived = pa.array([Path(f).stem for f in filenames], type=pa.string())
        raw = raw.append_column("doc_id", derived)
        missing.remove("doc_id")
    empty_fill, null_fill = _BACKFILL.get(role, ((), ()))
    hard = [name for name in missing if name not in empty_fill + null_fill]
    if hard:
        raise ValueError(
            f"shard {shard_path} missing columns {hard}; schema bump without compat shim?"
        )
    for name in missing:
        typ = schema.field(name).type
        col = pa.nulls(raw.num_rows, type=typ) if name in null_fill else pa.array([""] * raw.num_rows, type=typ)
        raw = raw.append_column(name, col)
    return raw.select([f.name for f in schema]).cast(schema)


# Back-compat alias for callers that still expect read_results to return
# the element table.
read_results = read_elements


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


class ShardVerificationError(RuntimeError):
    """Raised when a per-batch parquet shard fails on-disk verification."""


def verify_shard_persistence(
    output_path: Path,
    expected_docs: int,
    prev_total_size: int,
) -> int:
    """Sanity-check shard files after a write.

    Checks: every shard file exists, none are empty, manifest row count
    matches ``expected_docs``, every (source_hash, parent_elem_order) in
    table_cells / form_fields references an element with the matching
    kind, and the cumulative on-disk size has not shrunk.

    Returns the new cumulative on-disk size of the shard directory.
    Raises ``ShardVerificationError`` on any anomaly.
    """
    paths = _shard_paths(output_path)
    for path in paths.values():
        if not path.exists():
            raise ShardVerificationError(f"shard missing after write: {path}")
        if path.stat().st_size == 0:
            raise ShardVerificationError(f"shard is zero bytes: {path}")
        try:
            _ = pq.ParquetFile(str(path)).metadata  # smoke-test readability
        except Exception as e:
            raise ShardVerificationError(f"shard unreadable: {path}: {e}") from e

    manifest_rows = pq.ParquetFile(str(paths["manifest"])).metadata.num_rows
    if manifest_rows != expected_docs:
        raise ShardVerificationError(
            f"manifest row count mismatch: {paths['manifest']} has {manifest_rows} rows, "
            f"expected {expected_docs}"
        )

    _verify_sidecar_integrity(paths)

    shard_dir = output_path.parent
    cumulative_size = sum(p.stat().st_size for p in shard_dir.glob("*.parquet"))
    if cumulative_size < prev_total_size:
        raise ShardVerificationError(
            f"shard directory shrank: {shard_dir} was {prev_total_size}b, "
            f"now {cumulative_size}b — likely overwrite"
        )
    return cumulative_size


def _verify_sidecar_integrity(paths: dict[str, Path]) -> None:
    """Every (source_hash, parent_elem_order) in sidecars must map to a
    matching-kind element. Raises ShardVerificationError on orphan refs.
    """
    elems = pq.read_table(str(paths["elements"]), columns=["source_hash", "elem_order", "kind"])
    # Build {(source_hash, elem_order): kind}
    src = elems.column("source_hash").to_pylist()
    ord_ = elems.column("elem_order").to_pylist()
    kind = elems.column("kind").to_pylist()
    elem_index = {(s, o): k for s, o, k in zip(src, ord_, kind)}

    _check_sidecar(paths["table_cells"], elem_index, expected_kind="table", role="table_cells")
    _check_sidecar(paths["form_fields"], elem_index, expected_kind="form", role="form_fields")


def _check_sidecar(
    path: Path, elem_index: dict[tuple[str, int], str], *, expected_kind: str, role: str,
) -> None:
    t = pq.read_table(str(path), columns=["source_hash", "parent_elem_order"])
    src = t.column("source_hash").to_pylist()
    parent = t.column("parent_elem_order").to_pylist()
    for s, o in zip(src, parent):
        k = elem_index.get((s, o))
        if k is None:
            raise ShardVerificationError(
                f"{role} references missing element: source_hash={s} parent_elem_order={o}"
            )
        if k != expected_kind:
            raise ShardVerificationError(
                f"{role} references element with wrong kind: "
                f"source_hash={s} parent_elem_order={o} kind={k!r} (expected {expected_kind!r})"
            )


def verify_chunks_persistence(
    output_path: Path, expected_source_hashes: set[str], prev_total_size: int,
) -> int:
    """Sanity-check ``<base>.chunks.parquet`` after a write.

    Checks: the file exists, is non-empty, is readable as parquet, and
    every ``source_hash`` it contains is in ``expected_source_hashes``.
    Returns the cumulative on-disk size of all ``.chunks.parquet`` files
    in the same directory; raises :class:`ShardVerificationError` on any
    anomaly.

    ``expected_source_hashes`` may be empty when the batch had no
    completed documents — in that case the chunks file is also expected
    empty.
    """
    target = chunks_path_for(output_path)
    if not target.exists():
        raise ShardVerificationError(f"chunks shard missing after write: {target}")
    if target.stat().st_size == 0:
        raise ShardVerificationError(f"chunks shard is zero bytes: {target}")
    try:
        _ = pq.ParquetFile(str(target)).metadata  # smoke-test readability
    except Exception as e:
        raise ShardVerificationError(f"chunks shard unreadable: {target}: {e}") from e

    src_col = pq.read_table(str(target), columns=["source_hash"]).column("source_hash").to_pylist()
    if expected_source_hashes:
        unknown = {s for s in src_col if s not in expected_source_hashes}
        if unknown:
            raise ShardVerificationError(
                f"chunks shard references {len(unknown)} unknown source_hash(es): {target}"
            )
    elif src_col:
        raise ShardVerificationError(
            f"chunks shard {target} has {len(src_col)} rows but no source_hashes were expected"
        )

    shard_dir = output_path.parent
    cumulative = sum(p.stat().st_size for p in shard_dir.glob(f"*{CHUNKS_SUFFIX}"))
    if cumulative < prev_total_size:
        raise ShardVerificationError(
            f"chunks shard directory shrank: {shard_dir} was {prev_total_size}b, "
            f"now {cumulative}b — likely overwrite"
        )
    return cumulative
