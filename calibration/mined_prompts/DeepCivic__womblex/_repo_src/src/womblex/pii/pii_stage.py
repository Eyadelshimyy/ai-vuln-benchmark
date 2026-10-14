"""Per-stage PII detection + masking over an existing shard directory.

Consumes ``*.chunks.parquet`` + ``*.enrichment_entities.parquet`` and writes two
siblings per batch:

- ``*.pii_spans.parquet`` — one row per detected PII span (audit/reversible),
  with the graph ``entity_id`` and the ``<PERSON_n>`` replacement it maps to,
  located by the shared evidence reference (:mod:`womblex.store.evidence`) at
  the most precise level that is reliable and checked before the batch is
  written.
- ``*.clean_text.parquet`` — the masked, publishable text layer (one row per
  chunk; masked where spans were found, verbatim passthrough otherwise), a
  drop-in for ``*.chunks.parquet``. Gated by ``PIIConfig.write_clean_text``.
  ``mask_status`` records whether a candidate source covered each chunk, so
  verbatim text is never read as masked.

**Terminal stage — runs AFTER enrich + embed.** The Kanon-2 graph (built on raw
text) is the primary entity source; masking never rewrites the raw chunks that
feed Isaacus. Graph person/address mentions (full-narrative offsets) map into
``narrative`` chunks via ``chunk.start_char``; mentions from a table enriched on
its own (offsets into that table's markdown) map into that table's ``table``
chunks the same way, but only when the chunk's text is the markdown's slice at
its offsets (a repaired or stale chunk takes no table spans and reads
``not_masked``); the regex/context backstop
(``use_regex_backstop``, default off) covers all chunks when enabled. Replacement
tags are typed + numbered per document off the graph ``entity_id``
(``<PERSON_1>``, …) — Presidio-style, distinct per entity for downstream utility.

Mirrors :mod:`womblex.analyse.enrich_stage`: per-stage ``CheckpointManager``,
skip-existing on resume, batch-level isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from womblex.config import PIIConfig
from womblex.pii.cleaner import PIICleaner
from womblex.process.chunk_stage import _batch_bases
from womblex.process.chunker import TableText
from womblex.process.evidence import (
    EvidenceIndex,
    EvidenceIndexes,
    chunk_text_from,
    stamp_evidence,
)
from womblex.process.text_overlay import require_overlays
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_output import (
    enrichment_entities_path_for,
    enrichment_meta_path_for,
    read_enrichment_entities,
)
from womblex.store.evidence import CHUNK, TABLE_LAYER, evidence
from womblex.store.output import read_chunks, read_manifest
from womblex.store.pii_output import (
    MASK_MASKED,
    MASK_NO_ENTITY,
    MASK_NOT_MASKED,
    pii_spans_path_for,
    write_clean_text,
    write_pii_spans,
)

logger = logging.getLogger(__name__)


@dataclass
class PIIStageResult:
    batches_written: int
    spans_written: int
    chunks_masked: int


def pii_shards(
    shard_dir: Path,
    config: PIIConfig,
    *,
    text_source: str = "elements",
    checkpoint_mgr: CheckpointManager | None = None,
) -> PIIStageResult:
    """Detect + mask PII for every batch's chunks; write spans + clean_text siblings.

    ``text_source`` is the element-text layer the chunks and the enrichment were
    produced under; a declared layer missing from any batch refuses the stage.
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("pii_shards: no batches found in %s", shard_dir)
        return PIIStageResult(0, 0, 0)

    require_overlays(bases, text_source)
    cleaner = PIICleaner(
        entities=config.entities,
        model=config.model,
        model_options=config.model_options or None,
        context_similarity_threshold=config.context_similarity_threshold,
    )
    person_types = set(config.person_types)
    entities = set(config.entities)
    use_regex = config.use_regex_backstop
    write_clean = config.write_clean_text

    batches_written = 0
    spans_written = 0
    chunks_masked = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("pii_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        known_by_doc, table_known = _known_spans_by_doc(base, person_types, entities)
        enriched_docs, table_docs = _enriched_docs(base)
        indexes = EvidenceIndexes(base, text_source)
        off_source = 0

        # Group chunks per doc, in chunk_index order, so <ENTITY_n> numbering is
        # stable across a document (a doc's chunks all live in this one batch).
        by_doc: dict[str, list[dict]] = {}
        chunk_texts: dict[tuple[str, int], str] = {}
        for c in read_chunks(base).to_pylist():
            by_doc.setdefault(c["source_hash"], []).append(c)
            chunk_texts[(c["source_hash"], c["chunk_index"])] = c["text"] or ""

        span_rows: list[dict] = []
        clean_rows: list[dict] = []

        for source_hash, chunks in by_doc.items():
            chunks.sort(key=lambda c: c["chunk_index"])
            known = known_by_doc.get(source_hash)
            doc_enriched = source_hash in enriched_docs
            index = indexes.get(source_hash)
            numbering: dict[tuple[str, str], int] = {}
            counters: dict[str, int] = {}

            for c in chunks:
                text = c["text"] or ""
                is_narrative = c["content_type"] == "narrative"
                start = c["start_char"] or 0
                table = None if is_narrative or index is None else index.resolve_table(
                    elem_order=c["elem_order"], chunk_text=text, start=start)
                # A chunk lines up with its source when its text is the source's slice
                # at its offsets; only then is a span's offset a position in the source.
                source = index.narrative if is_narrative and index else (
                    table.markdown if table else None)
                aligned = source is not None and source[start:start + len(text)] == text
                off_source += index is not None and not aligned
                chunk_known, covered = known, doc_enriched
                if not is_narrative:
                    chunk_known = (
                        table_known.get((source_hash, table.elem_order, table.sheet))
                        if table is not None and aligned else None
                    )
                    covered = aligned and source_hash in table_docs
                spans = []
                if text.strip():
                    spans = cleaner.detect_spans(
                        text,
                        known_spans=chunk_known,
                        text_offset=start,
                        use_regex=use_regex,
                    )
                placed: list[tuple[int, int, str]] = []
                for s in spans:
                    key = (
                        (s.entity_type, s.entity_id) if s.entity_id
                        else (s.entity_type, f"__rx__{text[s.start:s.end].strip().lower()}")
                    )
                    if key not in numbering:
                        counters[s.entity_type] = counters.get(s.entity_type, 0) + 1
                        numbering[key] = counters[s.entity_type]
                    repl = f"<{s.entity_type}_{numbering[key]}>"
                    span_rows.append({
                        "source_hash": source_hash,
                        "chunk_index": c["chunk_index"],
                        "content_type": c["content_type"],
                        **_receipt(index, table, c, aligned, start + s.start, start + s.end,
                                   text[s.start:s.end]),
                        "text": text[s.start:s.end],
                        "entity_type": s.entity_type,
                        "entity_id": s.entity_id,
                        "detector": s.detector,
                        "score": float(s.score),
                        "replacement": repl,
                    })
                    placed.append((s.start, s.end, repl))

                if write_clean:
                    clean_rows.append({
                        "source_hash": source_hash,
                        "chunk_index": c["chunk_index"],
                        "content_type": c["content_type"],
                        "text": _apply_mask(text, placed),
                        "n_masked": len(placed),
                        "mask_status": _mask_status(bool(placed), use_regex or covered),
                    })
                    if placed:
                        chunks_masked += 1

        stamp_evidence(
            span_rows, indexes.get, text_key="text", label="pii_spans", base=base,
            chunk_text=chunk_text_from(chunk_texts),
        )
        write_pii_spans(span_rows, base)
        if write_clean:
            write_clean_text(clean_rows, base)
        batches_written += 1
        spans_written += len(span_rows)

        if checkpoint_mgr is not None:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids,
                    succeeded=len(span_rows),
                    failed=0,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info(
            "pii_shards: %s wrote %d spans, %d masked chunks",
            base.stem, len(span_rows), sum(1 for r in clean_rows if r["n_masked"]),
        )
        if off_source:
            logger.warning(
                "pii_shards: %s has %d chunk(s) that do not line up with their source text; "
                "their spans are located to the document or chunk, and a table chunk takes "
                "no table spans", base.stem, off_source,
            )
        uncovered = sum(1 for r in clean_rows if r["mask_status"] == MASK_NOT_MASKED)
        if uncovered:
            logger.warning(
                "pii_shards: %s has %d chunk(s) no PII candidate source covered; "
                "their clean_text is verbatim (mask_status=%s)",
                base.stem, uncovered, MASK_NOT_MASKED,
            )

    return PIIStageResult(
        batches_written=batches_written,
        spans_written=spans_written,
        chunks_masked=chunks_masked,
    )


def _mask_status(masked: bool, covered: bool) -> str:
    """``masked`` if a span was replaced, else ``no_entity`` when a candidate
    source covered the chunk, else ``not_masked``."""
    if masked:
        return MASK_MASKED
    return MASK_NO_ENTITY if covered else MASK_NOT_MASKED


def _enriched_docs(base_path: Path) -> tuple[set[str], set[str]]:
    """``(documents whose narrative was enriched, documents whose tables were enriched)``.

    The first includes documents with no entities, and leaves out a document
    whose narrative was skipped but whose tables were sent: its table mentions
    and its placeholder meta row (empty ``doc_type_enriched``) cover no
    narrative. A document's tables count as covered when its meta row records
    at least one table sent to the enricher.
    """
    docs: set[str] = set()
    table_docs: set[str] = set()
    entities = enrichment_entities_path_for(base_path)
    if entities.exists():
        docs.update(
            r["source_hash"] for r in read_enrichment_entities(entities)
            .select(["source_hash", "text_layer"]).to_pylist()
            if r["text_layer"] != TABLE_LAYER
        )
    meta = enrichment_meta_path_for(base_path)
    if meta.exists():
        names = pq.read_schema(str(meta)).names
        cols = ["source_hash"] + [c for c in ("doc_type_enriched", "table_count") if c in names]
        for row in pq.read_table(str(meta), columns=cols).to_pylist():
            if row.get("doc_type_enriched") != "":
                docs.add(row["source_hash"])
            if (row.get("table_count") or 0) > 0:
                table_docs.add(row["source_hash"])
    return docs, table_docs


def _receipt(
    index: EvidenceIndex | None, table: TableText | None, c: dict, aligned: bool, start: int, end: int,
    span_text: str,
) -> dict:
    """The span's evidence at the most precise level that is reliable.

    ``span`` where the chunk lines up with its source; ``document`` where it
    does not but the span's text is in the document; ``chunk`` where it is not
    (the chunk has moved on from the elements), or where the elements, or the
    chunk's table, are not to hand. A position is never searched for.
    """
    chunk_ref = evidence(CHUNK, elem_order=c["elem_order"], page=c["page_start"])
    if index is None or (table is None and c["content_type"] != "narrative"):
        return chunk_ref
    if not aligned:
        doc_ref = index.document_ref()
        return doc_ref if index.holds(doc_ref, span_text) else chunk_ref
    ref = (index.table_ref(table, start, end) if table is not None
           else index.narrative_ref(start, end))
    return ref or index.document_ref()


def _apply_mask(text: str, placed: list[tuple[int, int, str]]) -> str:
    """Replace each ``(start, end, replacement)`` in ``text`` (spans are
    non-overlapping; apply right-to-left to keep earlier offsets valid)."""
    if not placed:
        return text
    for start, end, repl in sorted(placed, key=lambda x: x[0], reverse=True):
        text = text[:start] + repl + text[end:]
    return text


def _known_spans_by_doc(
    base_path: Path,
    person_types: set[str],
    entities: set[str],
) -> tuple[
    dict[str, list[tuple[int, int, str, str]]],
    dict[tuple[str, int | None, str | None], list[tuple[int, int, str, str]]],
]:
    """Build ``(narrative, table)`` known spans from enrichment.

    Maps the Kanon-2 taxonomy onto PII tags: ``entity_type ∈ person_types`` →
    ``PERSON``; ``entity_type == 'address'`` → ``ADDRESS`` (each gated by
    ``entities``). Narrative spans are ``source_hash -> [(start, end, PII_TYPE,
    entity_id)]`` in full-narrative coordinates (the space narrative chunks index
    into); table spans are keyed ``(source_hash, elem_order, sheet)`` and index
    that table's markdown (the space its chunks index into). ``entity_id`` carries
    the graph grouping for numbering.
    """
    path = enrichment_entities_path_for(base_path)
    if not path.exists():
        return {}, {}
    table = read_enrichment_entities(path)
    narrative: dict[str, list[tuple[int, int, str, str]]] = {}
    in_tables: dict[tuple[str, int | None, str | None], list[tuple[int, int, str, str]]] = {}
    want_person = "PERSON" in entities
    want_address = "ADDRESS" in entities
    for r in table.to_pylist():
        et = r["entity_type"]
        if want_person and et in person_types:
            tag = "PERSON"
        elif want_address and et == "address":
            tag = "ADDRESS"
        else:
            continue
        if r["mention_start"] < 0 or r["mention_end"] <= r["mention_start"]:
            continue
        span = (r["mention_start"], r["mention_end"], tag, r["entity_id"] or "")
        if r["text_layer"] == TABLE_LAYER:
            in_tables.setdefault((r["source_hash"], r["elem_order"], r["sheet"]), []).append(span)
        else:
            narrative.setdefault(r["source_hash"], []).append(span)
    return narrative, in_tables


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not pii_spans_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["PIIStageResult", "pii_shards"]
