"""Build a minimal extraction shard (manifest + elements + table_cells) by hand."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.process.chunker import table_texts
from womblex.store.output import ELEMENT_SCHEMA, MANIFEST_SCHEMA, TABLE_CELLS_SCHEMA

DOC = "doc1"


def element(order: int, kind: str, *, source_hash: str = DOC, **kwargs) -> dict:
    row = {f.name: None for f in ELEMENT_SCHEMA}
    row.update({
        "source_hash": source_hash, "collection_id": "c", "elem_order": order,
        "kind": kind, "extractor": "test", "confidence": 1.0, "page": 1,
    })
    row.update(kwargs)
    return row


def cell(parent: int, row_i: int, col: int, value: str, *, source_hash: str = DOC) -> dict:
    return {
        "source_hash": source_hash, "parent_elem_order": parent, "row": row_i, "col": col,
        "value": value, "rowspan": 1, "colspan": 1, "value_type": "text",
    }


def build_shard(
    d: Path, elements: list[dict], cells: list[dict] | None = None, *, batch: str = "0001",
) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    base = d / f"batch-{batch}.parquet"
    hashes = sorted({e["source_hash"] for e in elements})
    manifest = []
    for h in hashes:
        man = {f.name: None for f in MANIFEST_SCHEMA}
        man.update({"source_hash": h, "doc_id": h, "filename": f"{h}.pdf", "status": "ok"})
        manifest.append(man)
    pq.write_table(pa.Table.from_pylist(manifest, schema=MANIFEST_SCHEMA),
                   str(d / f"batch-{batch}._manifest.parquet"))
    pq.write_table(pa.Table.from_pylist(elements, schema=ELEMENT_SCHEMA),
                   str(d / f"batch-{batch}.elements.parquet"))
    pq.write_table(pa.Table.from_pylist(cells or [], schema=TABLE_CELLS_SCHEMA),
                   str(d / f"batch-{batch}.table_cells.parquet"))
    return base


def contact_shard(d: Path) -> Path:
    elements = [
        element(0, "paragraph", text="Contact Jane Doe about the $5,000 grant."),
        element(1, "table", header_rows=[0]),
        element(2, "paragraph", text="Signed by the delegate."),
    ]
    cells = [
        cell(1, 0, 0, "Name"), cell(1, 0, 1, "Amount $"),
        cell(1, 1, 0, "John Citizen"), cell(1, 1, 1, "1,200"),
        cell(1, 2, 0, "Mary Major"), cell(1, 2, 1, "900"),
        cell(1, 3, 0, "Bob Lee"), cell(1, 3, 1, "2,400"),
        cell(1, 4, 0, "Ann Roe"), cell(1, 4, 1, "3,100"),
    ]
    return build_shard(d, elements, cells)


def table_markdown(base: Path) -> str:
    from womblex.process.chunk_stage import _load_elements

    return table_texts(_load_elements(base)[DOC])[0].markdown


NARRATIVE = "Contact Jane Doe about the $5,000 grant.\n\nSigned by the delegate."


def write_contact_chunks(base: Path, markdown: str, *, table_start: int = 0) -> None:
    """Chunks for the contact shard: its narrative, and its table (from ``table_start``)."""
    from womblex.store.output import write_chunks

    common = {"source_hash": DOC, "has_redaction": False, "page_start": 1, "page_end": 1}
    write_chunks([
        {**common, "chunk_index": 0, "text": NARRATIVE, "start_char": 0,
         "end_char": len(NARRATIVE), "content_type": "narrative", "elem_order": None},
        {**common, "chunk_index": 1, "text": markdown[table_start:], "start_char": table_start,
         "end_char": len(markdown), "content_type": "table", "elem_order": 1},
    ], base)
