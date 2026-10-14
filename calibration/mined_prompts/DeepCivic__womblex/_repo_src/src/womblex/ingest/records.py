"""Pre-extracted text records → element shards.

Womblex stages consume ``*.elements.parquet`` + ``*._manifest.parquet``
shard directories, which the file extractors normally produce. A
pre-extracted corpus (the Open Australian Legal Corpus; any JSONL/record
set of already-clean text) needs no extraction — this ingest turns records
straight into the same shard layout so the NLP pipeline (``enrich`` →
``chunk`` → ``embed`` → graph refresh) runs over it unchanged.

Unlike ``ingest/gnaf.py`` and ``ingest/abn_bulk.py`` — standalone register
ingests that *bypass* the NLP pipeline — this one *feeds* it: it writes the
element stream, not a flat register table.

Per record:

- ``source_hash = sha256(record_id + text)`` — content-addressed, so a
  re-ingest of unchanged records is a cache hit by construction (the asset
  refresh procedure relies on this) and the id survives text-identical
  reruns.
- text is split into blocks on blank lines; each block becomes one
  ``paragraph`` element. The canonical document text for the asset is
  therefore the reassembled narrative (blocks ``\\n\\n``-joined by
  :func:`womblex.process.chunker.reassemble_narrative`) — byte-identical to
  the source for the ``\\n\\n``-delimited majority, and enrichment/chunk
  offsets are internally consistent by construction (both stages reassemble
  the same elements).
- record metadata (id, citation, jurisdiction, …) lands in a
  ``*.provenance.parquet`` sidecar; the id is also the manifest ``doc_id``
  so checkpoints key on it.

The ingest is corpus-agnostic — a :class:`RecordFieldMapping` (declared by a
thin ``stories/<corpus>`` config) names which record fields are the id, the
text, and the provenance columns.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from womblex.ingest.elements import Element
from womblex.store.content_digest import content_digest
from womblex.store.output import (
    ELEMENT_SCHEMA,
    FORM_FIELDS_SCHEMA,
    MANIFEST_SCHEMA,
    PARSER_VERSION,
    TABLE_CELLS_SCHEMA,
    _shard_paths,
    _write_rows,
)
from womblex.store.provenance_output import write_provenance_shard
from womblex.store.source_provenance import qualify_root

logger = logging.getLogger(__name__)

# Text blocks are separated by one or more blank lines (runs of 2+ newlines,
# tolerating trailing spaces on the blank line). Each block is one paragraph
# element; the reassembler rejoins blocks with a single ``\n\n``.
_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n+")

_EXTRACTION_METHOD = "records"


@dataclass
class RecordFieldMapping:
    """Declares which record fields carry the id, text and provenance.

    Corpus-specific (declared by ``stories/<corpus>``); the ingest itself
    knows nothing about OALC or any other record set.
    """

    id_field: str
    text_field: str
    provenance_fields: list[str] = field(default_factory=list)
    collection_id: str = ""
    # Where the record set was published, recorded on every manifest row.
    # `source_relpath` stays empty: a record is content-addressed over its id
    # plus text, not file bytes, so no file under the root names it — the
    # provenance sidecar is the back-link.
    ingest_root: str = ""


@dataclass
class RecordsIngestResult:
    batches_written: int
    docs_ingested: int
    elements_written: int
    empty_docs: int


def records_source_hash(record_id: str, text: str) -> str:
    """Content-address a record: ``sha256(record_id + text)`` hex digest."""
    return hashlib.sha256(f"{record_id}{text}".encode()).hexdigest()


def split_text_blocks(text: str) -> list[str]:
    """Split verbatim text into paragraph blocks on blank lines.

    Empty/whitespace-only blocks are dropped (a leading/trailing blank line
    or a run of >2 newlines never yields an element). No text inside a block
    is altered — the verbatim policy holds at the ingest boundary.
    """
    if not text:
        return []
    return [b for b in _BLOCK_SPLIT.split(text.strip("\n")) if b.strip()]


def _record_to_elements(text: str) -> list[Element]:
    """One ``paragraph`` element per text block, ordered from 0."""
    return [
        Element(order=i, kind="paragraph", extractor=_EXTRACTION_METHOD, text=block)
        for i, block in enumerate(split_text_blocks(text))
    ]


def _element_row(source_hash: str, collection_id: str, elem: Element) -> dict:
    """Project an Element onto ELEMENT_SCHEMA (text kinds only — no cells/geometry)."""
    return {
        "source_hash": source_hash,
        "collection_id": collection_id,
        "elem_order": elem.order,
        "kind": elem.kind,
        "extractor": elem.extractor,
        "confidence": 0.0,
        "page": None,
        "bbox": None,
        "text": elem.text,
        "alt_text": None,
        "header_rows": None,
        "sheet": None,
        "row": None,
        "col": None,
        "value": None,
        "value_type": None,
        "formula": None,
        "number_format": None,
        "merge_range": None,
        "meta": [],
    }


def _batched(records: Iterable[dict], batch_size: int) -> Iterator[list[dict]]:
    buf: list[dict] = []
    for rec in records:
        buf.append(rec)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def _write_batch(
    batch: list[dict],
    mapping: RecordFieldMapping,
    output_path: Path,
) -> tuple[int, int, int]:
    """Write one batch's four element shards + provenance sidecar.

    Returns ``(docs, elements, empty_docs)``. ``empty_docs`` are records
    whose text produced no blocks — they still get a manifest row (status
    ``empty``) and a provenance row so the record is accounted for, but no
    elements, so downstream stages skip them.
    """
    import time

    extracted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ingest_root = qualify_root(mapping.ingest_root) if mapping.ingest_root else ""
    element_rows: list[dict] = []
    manifest_rows: list[dict] = []
    provenance_rows: list[dict] = []
    empty_docs = 0

    for rec in batch:
        doc_id = str(rec.get(mapping.id_field, "") or "")
        text = str(rec.get(mapping.text_field, "") or "")
        source_hash = records_source_hash(doc_id, text)
        elements = _record_to_elements(text)
        if not elements:
            empty_docs += 1
        for elem in elements:
            element_rows.append(_element_row(source_hash, mapping.collection_id, elem))
        manifest_rows.append({
            "source_hash": source_hash,
            "collection_id": mapping.collection_id,
            "doc_id": doc_id,
            "ingest_root": ingest_root,
            "source_relpath": "",
            "filename": doc_id,
            "ext": "",
            "extraction_method": _EXTRACTION_METHOD,
            "elements_count": len(elements),
            "table_cells_count": 0,
            "form_fields_count": 0,
            "status": "completed" if elements else "empty",
            "error": "",
            "extracted_at_iso": extracted_at,
            "parser_version": PARSER_VERSION,
            "content_digest": content_digest(elements),
        })
        prov = {"source_hash": source_hash, "doc_id": doc_id}
        for pf in mapping.provenance_fields:
            prov[pf] = str(rec.get(pf, "") or "")
        provenance_rows.append(prov)

    paths = _shard_paths(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(element_rows, paths["elements"], ELEMENT_SCHEMA)
    _write_rows([], paths["table_cells"], TABLE_CELLS_SCHEMA)
    _write_rows([], paths["form_fields"], FORM_FIELDS_SCHEMA)
    _write_rows(manifest_rows, paths["manifest"], MANIFEST_SCHEMA)
    write_provenance_shard(provenance_rows, mapping.provenance_fields, output_path)
    return len(batch), len(element_rows), empty_docs


def ingest_records(
    records: Iterable[dict],
    output_dir: Path,
    mapping: RecordFieldMapping,
    *,
    batch_size: int = 500,
    start_batch: int = 1,
) -> RecordsIngestResult:
    """Ingest pre-extracted text records into element-shard batches.

    ``records`` is any iterable of dicts (streamed — never fully
    materialised, so a multi-gigabyte JSONL ingests in constant memory).
    Writes ``batch-NNNN.{elements,table_cells,form_fields,_manifest,
    provenance}.parquet`` under *output_dir*, ~``batch_size`` docs each,
    numbered from ``start_batch`` (so a resumed/extended ingest continues
    past existing shards without overwriting them).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_written = 0
    docs_ingested = 0
    elements_written = 0
    empty_docs = 0

    for offset, batch in enumerate(_batched(records, batch_size)):
        batch_num = start_batch + offset
        output_path = output_dir / f"batch-{batch_num:04d}.parquet"
        docs, elems, empties = _write_batch(batch, mapping, output_path)
        batches_written += 1
        docs_ingested += docs
        elements_written += elems
        empty_docs += empties
        logger.info(
            "records: batch-%04d ingested %d docs (%d elements, %d empty)",
            batch_num, docs, elems, empties,
        )

    logger.info(
        "records: complete — %d batches, %d docs, %d elements, %d empty",
        batches_written, docs_ingested, elements_written, empty_docs,
    )
    return RecordsIngestResult(
        batches_written=batches_written,
        docs_ingested=docs_ingested,
        elements_written=elements_written,
        empty_docs=empty_docs,
    )


__all__ = [
    "RecordFieldMapping",
    "RecordsIngestResult",
    "ingest_records",
    "records_source_hash",
    "split_text_blocks",
]
