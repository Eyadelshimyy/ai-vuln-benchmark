"""The evidence reference: one shape for "where in the source this came from".

Every annotation sidecar that marks a span of source text (money, entity links,
PII) carries these columns in place of an anchor of its own, so a consumer
resolves any of them the same way. A row never carries a null receipt: it says
how precisely it is located, in ``anchor_level``, and carries the most precise
location that is reliable.

==============  ============================================================
``anchor_level``  what the row is located to, and what a check can say
==============  ============================================================
``span``        ``char_start`` / ``char_end`` index the text named by
                ``text_layer``; the source slice equals the span's text
``element``     ``elem_order`` (and ``cell_row`` / ``cell_col`` or ``sheet``
                for a cell); the span's text lies within that element or cell
``chunk``       the chunk the row names (``chunk_index``), with its pages and,
                for a table chunk, the table via ``elem_order``; the span's
                text lies within that chunk
``document``    the document only; the span's text lies within its narrative
                or one of its table markdowns
==============  ============================================================

``text_layer`` says what the offsets of a ``span`` row index: ``elements`` /
``normalised`` / ``spellfix`` (the narrative under that element-text layer, the
space chunks and enrichment mentions use), ``table_markdown`` (the markdown of
the table ``elem_order``, or of the sheet ``sheet``) or ``cell`` (one cell's
value). A position is never searched for: offsets are carried from where they
were computed, or the row is given a coarser level.

``evidence_check`` is the outcome of checking the receipt at its own level
before the file was written: ``holds``, ``fails`` or ``unchecked`` (its source or
text was not to hand, or the file predates the check). A receipt that fails is
written and labelled, never refused: the run completes and says what it could
not verify.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from womblex.store.output import _BBOX_TYPE

TABLE_LAYER = "table_markdown"
CELL_LAYER = "cell"

SPAN, ELEMENT, CHUNK, DOCUMENT = "span", "element", "chunk", "document"
HOLDS, FAILS, UNCHECKED = "holds", "fails", "unchecked"

EVIDENCE_FIELDS: tuple[pa.Field, ...] = (
    pa.field("elem_order", pa.int32()),
    pa.field("page", pa.int32()),
    pa.field("bbox", _BBOX_TYPE),
    pa.field("sheet", pa.string()),
    pa.field("cell_row", pa.int32()),
    pa.field("cell_col", pa.int32()),
    pa.field("char_start", pa.int32()),
    pa.field("char_end", pa.int32()),
    pa.field("text_layer", pa.string()),
    pa.field("anchor_level", pa.string()),
    pa.field("evidence_check", pa.string()),
)
EVIDENCE_COLUMNS: tuple[str, ...] = tuple(f.name for f in EVIDENCE_FIELDS)


def evidence(level: str, **fields: Any) -> dict[str, Any]:
    """An evidence reference at ``level``; columns not given are null."""
    unknown = set(fields) - set(EVIDENCE_COLUMNS)
    if level not in (SPAN, ELEMENT, CHUNK, DOCUMENT) or unknown:
        raise ValueError(f"bad evidence reference: level={level!r} unknown={sorted(unknown)}")
    return {**dict.fromkeys(EVIDENCE_COLUMNS), **fields, "anchor_level": level}


def backfill_evidence(
    raw: pa.Table, columns: dict[str, str] | None = None, *, level: str,
) -> pa.Table:
    """Add the evidence columns a pre-evidence file lacks.

    ``columns`` maps an evidence column to the legacy column that carried the
    same value unchanged (a cell's row and column, a sheet name); everything
    else is null, ``anchor_level`` is ``level`` and ``evidence_check`` is
    ``unchecked`` for every row. Offsets of the old shape are not re-expressed
    here: that needs the elements.
    """
    if level not in (SPAN, ELEMENT, CHUNK, DOCUMENT):
        raise ValueError(f"bad evidence level: {level!r}")
    columns = columns or {}
    for field in EVIDENCE_FIELDS:
        if field.name in raw.schema.names:
            continue
        legacy = columns.get(field.name)
        if legacy is not None and legacy in raw.schema.names:
            raw = raw.append_column(field.name, raw.column(legacy).cast(field.type))
        elif field.name in ("anchor_level", "evidence_check"):
            value = level if field.name == "anchor_level" else UNCHECKED
            raw = raw.append_column(field.name, pa.array([value] * raw.num_rows, field.type))
        else:
            raw = raw.append_column(field.name, pa.nulls(raw.num_rows, field.type))
    return raw
