"""Parquet embed transport for the artifact data layer.

The artifact embeds a figure's data as **Parquet** — compact (≈8–17× smaller
than the same CSV), and the format the client-side worker engine decodes in the
browser. See ``docs/architecture/large-detail-artifacts.md``.

The receipt for a Parquet block hashes the SHIPPED BYTES. The manifest records
``payload_sha256`` — SHA-256 over the exact base64-decoded Parquet payload the
page carries — and ``verify --file`` recomputes that same hash from the file, so
the check is byte-exact, host-independent, and needs no Parquet reader at all.
Nothing is ever re-derived on the verifier's machine: three adversarial reviews
showed that ANY re-derivation drifts (writer dtype mapping, ``os.linesep``,
library versions) and turns an untouched artifact into a false FILE ALTERED —
so the design hashes what ships, which removes the failure mode instead of
narrowing it. The binding's ``embedded_sha256`` (the source frame's content
fingerprint) is still recorded beside it, tying the record to its section, and
existing CSV-embed receipts are untouched.

**Why pyarrow and not DuckDB (for the writer).** pyarrow is Parquet's reference
implementation and preserves pandas frames far more faithfully — DuckDB maps
frames through *its* SQL type system, and notably rendered timezone-aware
timestamps in the READER'S local zone. The receipt no longer depends on any of
this (bytes are hashed as shipped), but faithful encoding still matters for
what the browser engine DISPLAYS, and ``_renders_identically`` in ``embed.py``
is the gate that keeps display-divergent types on the CSV transport.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def _require_pyarrow():
    """Import pyarrow, or fail loudly naming the extras key (invariant 4)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading or writing a Parquet-embedded artifact requires PyArrow. "
            "Install with: pip install 'tracebi[reports]'"
        ) from exc


def to_parquet_bytes(df: "pd.DataFrame") -> bytes:
    """Serialize *df* to Parquet (ZSTD) bytes for embedding.

    The caller (``plan_embed``) hashes the returned bytes into the manifest's
    ``payload_sha256`` and embeds the SAME bytes in the page — encode once,
    ship it, hash it. Parquet is not byte-reproducible across writers, which
    is exactly why the receipt is over these shipped bytes and never over a
    re-encoding.
    """
    _require_pyarrow()
    buf = io.BytesIO()
    # index=False: the fingerprint covers columns/dtypes/values, never the index.
    df.to_parquet(buf, engine="pyarrow", compression="zstd", index=False)
    return buf.getvalue()


#: Decode bounds for untrusted Parquet. ``verify --file`` runs against a file
#: someone was *sent*, so its embedded data is attacker-controllable: Parquet is
#: compressed and columnar, so a small block can describe an enormous frame, and
#: an unbounded decode would let a tiny .html exhaust the checker's memory. These
#: caps sit far above any artifact a browser could render (the artifact is itself
#: bounded by what the page can hold) and are checked against the file's own
#: metadata, before any data is read.
MAX_DECODE_ROWS = 50_000_000
MAX_DECODE_CELLS = 200_000_000        # rows × columns, before any data is read
MAX_DECODE_UNCOMPRESSED_BYTES = 2 * 1024**3  # 2 GiB, measured while reading


class ParquetTooLarge(ValueError):
    """An embedded Parquet block declares more data than we will decode."""


def from_parquet_bytes(data: bytes) -> "pd.DataFrame":
    """Decode embedded Parquet *data* back to a DataFrame.

    Not on the verify path — ``verify --file`` hashes a Parquet payload's bytes
    and never decodes them. This is for build-time tooling and tests, and it
    keeps the defensive posture required of anything that MIGHT be pointed at
    untrusted bytes.

    Refuses a block whose own metadata declares more than :data:`MAX_DECODE_ROWS`
    rows or :data:`MAX_DECODE_UNCOMPRESSED_BYTES` of data, so a hostile file
    cannot turn the checker into a decompression bomb.
    """
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO(data)
    pf = pq.ParquetFile(buf)
    md = pf.metadata

    # The footer is a CLAIM by the file, so it can only be used to refuse early,
    # never to conclude that reading is safe.
    if md.num_rows > MAX_DECODE_ROWS:
        raise ParquetTooLarge(
            f"embedded data declares {md.num_rows:,} rows, above the "
            f"{MAX_DECODE_ROWS:,} decode limit"
        )
    if md.num_rows * md.num_columns > MAX_DECODE_CELLS:
        raise ParquetTooLarge(
            f"embedded data declares {md.num_rows:,} rows × {md.num_columns} "
            f"columns, above the {MAX_DECODE_CELLS:,} cell decode limit"
        )

    # Then MEASURE while reading. ``total_byte_size`` counts encoded bytes —
    # dictionary- and RLE-encoded data materialises far larger — so a footer
    # figure cannot bound memory. Reading in batches and stopping at the real
    # accumulated size bounds what is actually allocated, whatever the file says.
    batches, seen = [], 0
    for batch in pf.iter_batches(batch_size=65_536):
        seen += batch.nbytes
        if seen > MAX_DECODE_UNCOMPRESSED_BYTES:
            raise ParquetTooLarge(
                f"embedded data exceeded the {MAX_DECODE_UNCOMPRESSED_BYTES:,} "
                f"byte decode limit while reading"
            )
        batches.append(batch)

    table = pa.Table.from_batches(batches, schema=pf.schema_arrow)
    frame = table.to_pandas()
    _reject_hostile_metadata(frame, pf.schema_arrow)
    return frame


def _reject_hostile_metadata(frame, schema) -> None:
    """Refuse a file whose pandas metadata rewrote the frame it decoded to.

    A Parquet file carries an optional ``pandas`` metadata block, and pyarrow
    honours it — it can rename columns or promote one to an index. The browser
    engine ignores that block entirely, so a crafted file could decode one way
    for the checker and render another way for the reader: verify would vouch
    for a frame nobody sees. The columns must therefore be exactly the schema's
    fields, in order, with nothing hidden in an index.
    """
    fields = list(schema.names)
    if list(frame.columns) != fields:
        raise ValueError(
            "embedded Parquet metadata rewrote the decoded columns "
            f"({list(frame.columns)!r} vs the schema's {fields!r}) — refusing a "
            "file that decodes differently than it renders"
        )
    index = frame.index
    if index.name is not None or index.nlevels != 1:
        raise ValueError(
            "embedded Parquet metadata promoted data into the frame index — "
            "refusing a file that decodes differently than it renders"
        )
