"""G-NAF PSV → Parquet ingest.

Reads pipe-delimited G-NAF files and writes one Parquet file per input PSV,
preserving exact relational structure with zero semantic mutation. The G-NAF
distribution ships each PSV with an uppercase header row naming its columns;
that row is detected and skipped so it never leaks in as a data row, while a
headerless file is read as-is. Column names always come from the static
schema, not the header.

Designed for the national G-NAF distribution (all states, all table types).
Authority Code and Standard tables are written as separate Parquet files.

Usage::

    from womblex.ingest.gnaf import ingest_gnaf_directory
    results = ingest_gnaf_directory(Path("G-NAF/G-NAF FEBRUARY 2026"), Path("output/gnaf"))
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from womblex.ingest.gnaf_schema import ALL_TABLES, SCHEMA_VERSION
from womblex.utils.checksum import md5_file

logger = logging.getLogger(__name__)

# G-NAF PSV filenames follow two patterns:
#   Standard:       {STATE}_{TABLE_NAME}_psv.psv
#   Authority Code: Authority_Code_{TABLE_NAME}_psv.psv
_AUTHORITY_RE = re.compile(r"^Authority_Code_(.+)_psv$", re.IGNORECASE)
_STANDARD_RE = re.compile(r"^([A-Z]{2,3})_(.+)_psv$", re.IGNORECASE)


def _parse_filename(stem: str) -> tuple[str | None, str | None]:
    """Extract (state, table_name) from a PSV filename stem.

    Returns (None, table_name) for Authority Code files.
    Returns (state, table_name) for Standard files.
    Returns (None, None) if the filename doesn't match either pattern.
    """
    m = _AUTHORITY_RE.match(stem)
    if m:
        return None, m.group(1).upper()

    m = _STANDARD_RE.match(stem)
    if m:
        return m.group(1).upper(), m.group(2).upper()

    return None, None


def _has_header_row(psv_path: Path, columns: list[str]) -> bool:
    """True if the file's first line is a header row naming the schema columns.

    G-NAF Standard and Authority Code PSVs carry an uppercase header row; a
    headerless file does not. Compared case-insensitively with any leading BOM
    stripped, so the header is skipped only when it is genuinely present and a
    real first data row is never mistaken for one.
    """
    try:
        with psv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            first = fh.readline()
    except (OSError, UnicodeDecodeError):
        return False
    if not first:
        return False
    fields = [f.strip() for f in first.rstrip("\r\n").split("|")]
    return [f.lower() for f in fields] == [c.lower() for c in columns]


def ingest_psv(
    psv_path: Path,
    output_dir: Path,
    *,
    compute_md5: bool = True,
) -> Path | None:
    """Convert a single G-NAF PSV file to Parquet.

    Args:
        psv_path: Path to the ``.psv`` file.
        output_dir: Directory to write the output Parquet file.
        compute_md5: Attach source MD5 as Parquet metadata (default True).

    Returns:
        Path to the written Parquet file, or None if the file was skipped.
    """
    stem = psv_path.stem
    state, table_name = _parse_filename(stem)

    if table_name is None:
        logger.warning("gnaf: unrecognised filename pattern, skipping: %s", psv_path.name)
        return None

    columns = ALL_TABLES.get(table_name)
    if columns is None:
        logger.warning("gnaf: no schema for table %s, skipping: %s", table_name, psv_path.name)
        return None

    # Read with pyarrow.csv — streamed, constant memory, all columns as strings
    # (zero semantic mutation: no type coercion, no null inference). The G-NAF
    # header row, when present, is skipped so it never lands as a data row;
    # column names come from the schema regardless.
    read_opts = pcsv.ReadOptions(
        column_names=columns,
        skip_rows=1 if _has_header_row(psv_path, columns) else 0,
        block_size=1 << 20,  # 1 MB read blocks
    )
    parse_opts = pcsv.ParseOptions(delimiter="|")
    convert_opts = pcsv.ConvertOptions(
        column_types={col: pa.string() for col in columns},
        strings_can_be_null=False,
    )

    try:
        table = pcsv.read_csv(
            str(psv_path),
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts,
        )
    except Exception as e:
        logger.error("gnaf: failed to read %s: %s", psv_path.name, e)
        return None

    # Validate column count matches schema.
    if table.num_columns != len(columns):
        logger.error(
            "gnaf: column count mismatch for %s: expected %d, got %d",
            psv_path.name, len(columns), table.num_columns,
        )
        return None

    # Provenance metadata.
    metadata = {
        b"gnaf.schema_version": SCHEMA_VERSION.encode(),
        b"gnaf.table_name": table_name.encode(),
        b"gnaf.source_file": psv_path.name.encode(),
        b"gnaf.row_count": str(table.num_rows).encode(),
    }
    if state:
        metadata[b"gnaf.state"] = state.encode()
    if compute_md5:
        metadata[b"gnaf.source_md5"] = md5_file(psv_path).encode()

    # Merge with any existing Arrow metadata.
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, **metadata})

    # Write.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{stem}.parquet"
    out_path = output_dir / out_name

    pq.write_table(table, str(out_path))
    logger.info(
        "gnaf: %s → %s (%d rows)",
        psv_path.name, out_path.name, table.num_rows,
    )
    return out_path


def discover_psv_files(root: Path) -> list[Path]:
    """Recursively find all ``.psv`` files under a directory."""
    return sorted(root.rglob("*.psv"))


def ingest_gnaf_directory(
    root: Path,
    output_dir: Path,
    *,
    compute_md5: bool = True,
) -> list[Path]:
    """Ingest all G-NAF PSV files under *root* into Parquet.

    Walks the directory tree, converts each ``.psv`` file, and writes
    output Parquet files into *output_dir* preserving no subdirectory
    structure (flat output).

    Args:
        root: Root directory of the G-NAF distribution (e.g.
            ``G-NAF/G-NAF FEBRUARY 2026``).
        output_dir: Directory for output Parquet files.
        compute_md5: Attach source MD5 as Parquet metadata.

    Returns:
        List of paths to successfully written Parquet files.
    """
    psv_files = discover_psv_files(root)
    if not psv_files:
        logger.warning("gnaf: no .psv files found under %s", root)
        return []

    logger.info("gnaf: found %d PSV files under %s", len(psv_files), root)

    written: list[Path] = []
    skipped = 0
    for psv_path in psv_files:
        result = ingest_psv(psv_path, output_dir, compute_md5=compute_md5)
        if result:
            written.append(result)
        else:
            skipped += 1

    logger.info(
        "gnaf: complete — %d written, %d skipped",
        len(written), skipped,
    )
    return written
