"""Parquet IO for the money annotation op (``womblex money``).

Two siblings per batch, self-contained like :mod:`womblex.store.quality_output`:

- ``*.money_spans.parquet`` — one row per extracted amount, joinable on
  ``source_hash``. Three loci share the file and are discriminated by
  ``locus``; each is located by the shared evidence reference
  (:mod:`womblex.store.evidence`):

  =============  =========================================================
  ``narrative``  ``char_start`` / ``char_end`` index the reassembled
                 narrative under ``text_layer`` — the space enrichment
                 mentions use, so the two join and map to chunks the way
                 ``graph_refresh`` does.
  ``table_cell`` ``elem_order`` is the table element; ``cell_row`` /
                 ``cell_col`` address the cell on ``*.table_cells.parquet``,
                 and the offsets index that cell's value.
  ``sheet_cell`` ``elem_order`` is the ``sheet_cell`` element, with
                 ``sheet`` / ``cell_row`` / ``cell_col``.
  =============  =========================================================

- ``*.money_columns.parquet`` — the column-classification audit: one row per
  column considered, money or not, with the evidence that decided it. The
  column path carries ~98.7% of the corpus's amounts off a single per-column
  verdict, and with no labelled ground truth yet (``docs/money-extraction.md``) this is
  the surface that makes that verdict reviewable.

``value`` is ``decimal128(38, 4)`` — exact, not float. Aggregating 48,997
register amounts accumulates float error, and reconciliation compares values
for equality. Sub-hundredth-of-a-cent precision is rounded away by contract.

Annotation only: neither element nor chunk text is ever mutated.
"""

from __future__ import annotations

import logging
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from womblex.store.evidence import ELEMENT, EVIDENCE_FIELDS, SPAN, backfill_evidence
from womblex.store.output import _write_rows
from womblex.store.run_stamp import sidecar_footer

logger = logging.getLogger(__name__)

MONEY_SPANS_SUFFIX = ".money_spans.parquet"
MONEY_COLUMNS_SUFFIX = ".money_columns.parquet"

VALUE_TYPE = pa.decimal128(38, 4)
_QUANT = Decimal("0.0001")
# decimal128(38, 4) holds 34 integer digits; anything wider cannot be stored.
_MAX_VALUE = Decimal(10) ** 34

MONEY_SPANS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("locus", pa.string()),            # narrative | table_cell | sheet_cell
    *EVIDENCE_FIELDS,                  # where the amount is, and how precisely
    ("text", pa.string()),             # original — never lost
    ("value", VALUE_TYPE),
    ("currency", pa.string()),         # nullable — money-marked, currency unresolved
    ("currency_source", pa.string()),  # symbol|iso|word|number_format|column_header|document_default
    ("evidence", pa.string()),         # p1..p11 | number_format | header+numeric | header_currency
    ("modifier", pa.string()),         # approximately | up to | … — never folded into value
    ("multiplier", pa.string()),       # thousand | million | billion | trillion | cents
    ("negative", pa.bool_()),
    ("confidence", pa.float32()),
    ("range_group", pa.int32()),       # nullable — links range endpoints
    ("range_role", pa.string()),       # lower | upper
    ("column_id", pa.string()),        # nullable — the classified column a cell inherited
    ("context", pa.string()),
])

MONEY_COLUMNS_SCHEMA = pa.schema([
    ("source_hash", pa.string()),
    ("column_id", pa.string()),
    ("locus", pa.string()),            # table_cell | sheet_cell
    ("parent_elem_order", pa.int32()),
    ("sheet", pa.string()),
    ("col", pa.int32()),
    ("header_text", pa.string()),
    ("number_format", pa.string()),
    ("verdict", pa.string()),          # money | vetoed | insufficient
    ("evidence", pa.string()),
    ("veto_term", pa.string()),
    ("currency", pa.string()),
    ("scale", pa.string()),
    ("numeric_fraction", pa.float32()),
    ("null_fraction", pa.float32()),
    ("confidence", pa.float32()),
    ("cells_total", pa.int32()),
    ("cells_extracted", pa.int32()),
])


def money_spans_path_for(base_path: Path) -> Path:
    return base_path.parent / f"{base_path.stem}{MONEY_SPANS_SUFFIX}"


def money_columns_path_for(base_path: Path) -> Path:
    return base_path.parent / f"{base_path.stem}{MONEY_COLUMNS_SUFFIX}"


def quantise(value: Decimal) -> Decimal | None:
    """Round to the stored scale; ``None`` when the value cannot be stored.

    An amount wider than ``decimal128(38, 4)`` is dropped rather than silently
    truncated — a wrong number is worse than a missing one here.
    """
    try:
        if abs(value) >= _MAX_VALUE:
            return None
        return value.quantize(_QUANT, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, OverflowError):
        return None


def write_money_spans(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's money spans (rows match :data:`MONEY_SPANS_SCHEMA`)."""
    target = money_spans_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, MONEY_SPANS_SCHEMA, metadata=sidecar_footer(output_path, "money"))
    logger.info("Wrote money_spans shard %s: rows=%d", target.name, len(rows))
    return target


def write_money_columns(rows: list[dict], output_path: Path) -> Path:
    """Write a batch's column verdicts (rows match :data:`MONEY_COLUMNS_SCHEMA`)."""
    target = money_columns_path_for(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, target, MONEY_COLUMNS_SCHEMA,
                metadata=sidecar_footer(output_path, "money"))
    logger.info("Wrote money_columns shard %s: rows=%d", target.name, len(rows))
    return target


def read_money_spans(path: Path) -> pa.Table:
    """Read money spans from a single shard file or a shard-directory glob."""
    return _read(path, MONEY_SPANS_SUFFIX, MONEY_SPANS_SCHEMA, money_spans_path_for)


def read_money_columns(path: Path) -> pa.Table:
    """Read column verdicts from a single shard file or a shard-directory glob."""
    return _read(path, MONEY_COLUMNS_SUFFIX, MONEY_COLUMNS_SCHEMA, money_columns_path_for)


def _read(path: Path, suffix: str, schema: pa.Schema, path_for) -> pa.Table:
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob(f"*{suffix}"))
        if not shards:
            return _empty(schema)
        return pa.concat_tables([_read_shard(s, schema) for s in shards])
    target = p if p.name.endswith(suffix) else path_for(p)
    return _read_shard(target, schema)


def _empty(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)


def money_spans_from_legacy(raw: pa.Table) -> pa.Table:
    """Contract 1.x money spans carried their own anchors; keep what maps unchanged.

    A narrative span keeps its offsets, page and layer (``span``). A cell span's
    offsets within the cell were never recorded, so it is located to its cell
    (``element``): the table element, row and column, or the sheet.
    """
    names = raw.schema.names
    if "anchor_level" in names or "start_char" not in names:
        return raw
    narrative = pc.equal(raw["locus"], "narrative")
    nulls = pa.nulls(raw.num_rows, pa.int32())
    elem = pc.coalesce(raw["parent_elem_order"], raw["elem_order"])
    raw = raw.set_column(names.index("elem_order"), "elem_order", elem)
    raw = raw.append_column("cell_row", pc.if_else(narrative, nulls, raw["row"]))
    raw = raw.append_column("cell_col", pc.if_else(narrative, nulls, raw["col"]))
    raw = raw.append_column("char_start", pc.if_else(narrative, raw["start_char"], nulls))
    raw = raw.append_column("char_end", pc.if_else(narrative, raw["end_char"], nulls))
    raw = raw.append_column("text_layer", pc.if_else(
        narrative, raw["text_source"], pa.scalar(None, pa.string())))
    return raw.append_column("anchor_level", pc.if_else(
        narrative, pa.scalar(SPAN), pa.scalar(ELEMENT)))


def _read_shard(path: Path, schema: pa.Schema) -> pa.Table:
    raw = pq.read_table(str(path))
    if schema is MONEY_SPANS_SCHEMA:
        # `from_legacy` sets every row's level; the rest of the columns read null.
        raw = backfill_evidence(money_spans_from_legacy(raw), level=ELEMENT)
    missing = [f.name for f in schema if f.name not in raw.schema.names]
    if missing:
        raise ValueError(
            f"money shard {path} missing columns {missing}; schema bump without compat shim?"
        )
    return raw.select([f.name for f in schema]).cast(schema)


__all__ = [
    "MONEY_COLUMNS_SCHEMA",
    "MONEY_COLUMNS_SUFFIX",
    "MONEY_SPANS_SCHEMA",
    "MONEY_SPANS_SUFFIX",
    "VALUE_TYPE",
    "money_columns_path_for",
    "money_spans_from_legacy",
    "money_spans_path_for",
    "quantise",
    "read_money_columns",
    "read_money_spans",
    "write_money_columns",
    "write_money_spans",
]
