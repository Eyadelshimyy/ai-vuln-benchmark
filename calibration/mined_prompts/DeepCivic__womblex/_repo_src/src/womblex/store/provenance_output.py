"""Per-batch provenance sidecar (``*.provenance.parquet``) + corpus manifest.

A pre-extracted corpus (OALC, and any future record set) carries source
metadata that is *not* derivable from the extracted text — a stable record
id, a citation, a jurisdiction, a source URL. The NLP stages
(``enrich``/``chunk``/``embed``) only need ``*.elements.parquet`` +
``*._manifest.parquet``; this sidecar carries the record metadata alongside
them, keyed by ``source_hash`` so it joins the other sidecars, and is
consolidated into a single ``manifest.parquet`` at the run root that maps
``source_hash`` back to the record's provenance columns.

Self-contained store module (mirrors ``store/enrichment_doc.py`` /
``store/normalise_output.py``). The provenance columns are declared by the
caller (``ingest.records`` reads them from a ``stories/<corpus>`` field
mapping), so the schema is built dynamically — every provenance column is a
verbatim string, no type coercion, absent values become ``""``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.contract import contract_footer

logger = logging.getLogger(__name__)

PROVENANCE_SUFFIX = ".provenance.parquet"

# Leading columns every provenance sidecar carries; the corpus-declared
# provenance fields follow, all string-typed.
_FIXED_COLUMNS = ("source_hash", "doc_id")


def provenance_schema(provenance_fields: list[str]) -> pa.Schema:
    """Build the provenance schema: source_hash, doc_id, then declared fields."""
    fields = [(c, pa.string()) for c in _FIXED_COLUMNS]
    fields += [(c, pa.string()) for c in provenance_fields if c not in _FIXED_COLUMNS]
    return pa.schema(fields)


def provenance_path_for(base_path: Path) -> Path:
    """Return ``<base>.provenance.parquet`` sibling for a shard base path."""
    return base_path.parent / f"{base_path.stem}{PROVENANCE_SUFFIX}"


def write_provenance_shard(
    rows: list[dict[str, str]], provenance_fields: list[str], base_path: Path,
) -> Path:
    """Write a batch's provenance rows to ``<base>.provenance.parquet``.

    ``rows`` carry ``source_hash`` + ``doc_id`` + each declared provenance
    field (missing keys default to ``""``). Empty input still writes a
    schema-correct empty file so downstream globs are safe.
    """
    schema = provenance_schema(provenance_fields)
    target = provenance_path_for(base_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = schema.with_metadata(contract_footer(target))
    if rows:
        norm = [{f.name: str(r.get(f.name, "")) for f in schema} for r in rows]
        table = pa.Table.from_pylist(norm, schema=schema)
    else:
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    pq.write_table(table, str(target), compression="zstd", compression_level=3)
    logger.info("Wrote provenance shard %s: rows=%d", target.name, len(rows))
    return target


def read_provenance(path: Path) -> pa.Table:
    """Read provenance rows from a single sibling file or a shard-dir glob.

    A directory concatenates every ``*.provenance.parquet`` under it; the
    union of columns across shards is preserved (later ingests may add
    provenance fields), missing columns filled with ``""``.
    """
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{PROVENANCE_SUFFIX}"))
        if not shards:
            schema = provenance_schema([])
            return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
        tables = [pq.read_table(str(s)) for s in shards]
        return _concat_widening(tables)
    target = p if p.name.endswith(PROVENANCE_SUFFIX) else provenance_path_for(p)
    return pq.read_table(str(target))


def write_corpus_manifest(shard_dir: Path, output_path: Path | None = None) -> Path:
    """Consolidate per-batch provenance sidecars into one ``manifest.parquet``.

    Mirrors ``store/run_manifest.write_run_manifest`` but for the corpus
    provenance layer: reads every ``*.provenance.parquet`` under
    ``shard_dir`` and writes the source_hash → provenance table to
    ``output_path`` (default ``<shard_dir>/../manifest.parquet`` — the run
    root, per the asset layout). Returns the written path.
    """
    table = read_provenance(shard_dir)
    out = output_path or shard_dir.parent / "manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Carries the corpus-declared provenance columns, so it is labelled as provenance.
    table = table.replace_schema_metadata(contract_footer(out, role="provenance"))
    pq.write_table(table, str(out), compression="zstd", compression_level=3)
    logger.info("Wrote corpus manifest %s: rows=%d", out, table.num_rows)
    return out


def _concat_widening(tables: list[pa.Table]) -> pa.Table:
    """Concatenate provenance tables, unioning columns (missing → '')."""
    all_cols: list[str] = []
    for t in tables:
        for name in t.schema.names:
            if name not in all_cols:
                all_cols.append(name)
    aligned: list[pa.Table] = []
    for t in tables:
        cols = {name: t.column(name) for name in t.schema.names}
        arrays = []
        for name in all_cols:
            if name in cols:
                arrays.append(cols[name].cast(pa.string()))
            else:
                arrays.append(pa.array([""] * t.num_rows, type=pa.string()))
        aligned.append(pa.table(dict(zip(all_cols, arrays))))
    return pa.concat_tables(aligned)


__all__ = [
    "PROVENANCE_SUFFIX",
    "provenance_path_for",
    "provenance_schema",
    "read_provenance",
    "write_corpus_manifest",
    "write_provenance_shard",
]
