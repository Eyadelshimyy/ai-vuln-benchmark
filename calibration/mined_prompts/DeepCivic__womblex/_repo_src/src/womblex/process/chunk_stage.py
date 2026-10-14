"""Per-stage chunking over an existing shard directory.

Consumes ``*.elements.parquet`` + ``*.table_cells.parquet`` +
``*._manifest.parquet`` already written by the extraction stage and
writes a ``*.chunks.parquet`` sibling per batch.

Reassembles narrative as the ``\\n\\n``-joined text of TEXT_KINDS
elements in ``elem_order`` per source_hash (matching the canonical
"element stream is the source of truth" policy). Tables are derived
from the table_cells sidecar plus the synthetic spreadsheet-sheet
view from ``ingest.views``; the same legacy projection
``ExtractionResult.tables`` uses, so the in-memory E2E path and this
per-stage path see the same tables.

One ``Chunker`` is created at the top of :func:`chunk_shards` and
reused across every batch — semchunk's memoise cache then accumulates
across the whole shard directory.

**Offline by default; only AI chunking needs the API.** Plain token
chunking sizes chunks with the kanon-2 tokeniser, which is vendored under
``_models/`` and runs entirely client-side — so a keyless, air-gapped run
still chunks (gated only on the tokeniser resolving locally, via
:func:`~womblex.utils.availability.tokenizer_available`). AI chunking
(``chunking.chunking_model`` set) calls the Isaacus enricher per document
at chunk time and is the only path that needs a configured deployment
(:func:`~womblex.utils.availability.isaacus_available`).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from womblex.config import ChunkingConfig
from womblex.ingest.elements import BBox, Cell, Element
from womblex.process.chunker import (
    ChunkInput,
    TextChunk,
    build_chunk_input,
    chunk_batch,
    create_chunker,
    resolve_tokenizer,
)
from womblex.process.text_overlay import apply_overlay, load_overlay, require_overlays
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_doc import read_enrichment_docs
from womblex.store.output import (
    _SHARD_ROLES,
    _SHARD_SUFFIX,
    CHUNKS_SUFFIX,
    chunks_path_for,
    read_elements,
    read_manifest,
    read_table_cells,
    write_chunks,
)
from womblex.utils.availability import isaacus_available, tokenizer_available
from womblex.utils.isaacus_client import make_ai_chunking_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ChunkStageResult:
    batches_written: int
    docs_chunked: int
    total_chunks: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_shards(
    shard_dir: Path,
    chunking_config: ChunkingConfig,
    *,
    text_source: str = "elements",
    checkpoint_mgr: CheckpointManager | None = None,
) -> ChunkStageResult:
    """Chunk every batch in ``shard_dir`` and write ``*.chunks.parquet`` siblings.

    Skips batches whose chunks file already exists when ``checkpoint_mgr``
    is provided and reports every contained doc as already processed.
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("chunk_shards: no batches found in %s", shard_dir)
        return ChunkStageResult(0, 0, 0)
    require_overlays(bases, text_source)

    # Resolved first so an unknown tokeniser name is an error either way.
    tokenizer = resolve_tokenizer(
        chunking_config.tokenizer, chunking_config.tokenizer_options
    )

    # AI chunking (chunking_model set) calls the Kanon-2 *API* at chunk time, so
    # it needs a configured deployment. Plain token chunking sizes chunks with
    # the vendored tokeniser and runs fully offline — a keyless local run must
    # still chunk, gating only on the tokeniser resolving locally.
    if chunking_config.chunking_model:
        if not isaacus_available():
            logger.warning(
                "chunk_shards: AI chunking is enabled (chunking.chunking_model=%r) "
                "but Isaacus is not available (needs the isaacus SDK + "
                "ISAACUS_API_KEY, or ISAACUS_SAGEMAKER_ENDPOINTS) — skipping "
                "chunking for %s; no *.chunks.parquet written. Unset "
                "chunking.chunking_model to chunk offline with the local "
                "tokeniser.", chunking_config.chunking_model, shard_dir,
            )
            return ChunkStageResult(0, 0, 0)
    elif not tokenizer_available(tokenizer):
        logger.warning(
            "chunk_shards: chunk-size tokeniser %r is not resolvable locally "
            "(needs `transformers` plus a bundled copy under _models/ or "
            "WOMBLEX_MODELS_DIR) — skipping chunking for %s; no *.chunks.parquet "
            "written. The default kanon-2-tokenizer is vendored; a custom "
            "tokenizer must be bundled to keep chunking offline.",
            chunking_config.tokenizer, shard_dir,
        )
        return ChunkStageResult(0, 0, 0)

    chunker = create_chunker(
        tokenizer=tokenizer,
        chunk_size=chunking_config.chunk_size,
        chunking_model=chunking_config.chunking_model,
        isaacus_client=make_ai_chunking_client(chunking_config.chunking_model),
        tokenizer_kwargs=chunking_config.tokenizer_kwargs,
        memoize=chunking_config.memoize,
        cache_maxsize=chunking_config.cache_maxsize,
        max_token_chars=chunking_config.max_token_chars,
    )

    batches_written = 0
    docs_chunked = 0
    total_chunks = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("chunk_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        inputs, doc_ids_by_hash = _build_inputs_for_batch(
            base, chunking_config.chunk_tables, text_source,
        )

        if inputs:
            overrides = (
                _load_narrative_overrides(base, text_source)
                if chunking_config.chunking_model else None
            )
            chunks_by_hash = chunk_batch(
                inputs,
                chunker,
                overlap=chunking_config.overlap,
                processes=chunking_config.processes,
                progress=chunking_config.progress,
                narrative_overrides=overrides,
            )
            rows = _chunks_to_rows(chunks_by_hash)
            chunked_now = sum(1 for c in chunks_by_hash.values() if c)
            chunks_now = sum(len(c) for c in chunks_by_hash.values())
        else:
            # Manifest exists but no chunkable docs (all rows error or zero
            # elements). Still write an empty sidecar so downstream globs
            # account for this batch, and still advance the checkpoint so
            # resume doesn't re-attempt forever.
            rows = []
            chunked_now = 0
            chunks_now = 0

        write_chunks(rows, base)
        batches_written += 1
        docs_chunked += chunked_now
        total_chunks += chunks_now

        if checkpoint_mgr is not None and doc_ids_by_hash:
            # Checkpoint every doc in the manifest, not just the ones that
            # produced chunks — error/empty docs are "done" at this stage too.
            all_doc_ids = list(doc_ids_by_hash.values())
            checkpoint_mgr.update(
                doc_ids=all_doc_ids,
                succeeded=chunked_now,
                failed=len(all_doc_ids) - chunked_now,
                batch_num=int(base.stem.replace("batch-", "") or 0),
            )

        logger.info(
            "chunk_shards: %s wrote %d chunks across %d docs",
            base.stem, chunks_now, chunked_now,
        )

    return ChunkStageResult(
        batches_written=batches_written,
        docs_chunked=docs_chunked,
        total_chunks=total_chunks,
    )


# ---------------------------------------------------------------------------
# Batch discovery (mirror of shard_audit._batch_bases without the cycle)
# ---------------------------------------------------------------------------


def _batch_bases(shard_dir: Path) -> list[Path]:
    seen: set[str] = set()
    bases: list[Path] = []
    for role in _SHARD_ROLES:
        for p in shard_dir.glob(f"*{_SHARD_SUFFIX[role]}"):
            stem = p.name[: -len(_SHARD_SUFFIX[role])]
            if stem.endswith(".corrupt"):
                continue
            if stem in seen:
                continue
            seen.add(stem)
            bases.append(shard_dir / f"{stem}.parquet")
    bases.sort(key=lambda p: p.name)
    return bases


# ---------------------------------------------------------------------------
# Per-batch input construction
# ---------------------------------------------------------------------------


def _build_inputs_for_batch(
    base_path: Path, chunk_tables_enabled: bool, text_source: str = "elements",
) -> tuple[list[ChunkInput], dict[str, str]]:
    """Read a batch's element-stream parquets and produce ChunkInputs.

    Returns ``(inputs, {source_hash: doc_id})``. The ``src_to_doc`` map
    covers every row in the manifest, including ``status='error'`` docs
    that contribute no chunks — callers need it to checkpoint those as
    "done at this stage" alongside the chunkable ones.

    When ``text_source`` is not ``'elements'`` the matching element-text
    overlay (normalise / spellfix) is applied before reassembly, so chunks
    inherit the cleaned/repaired text.
    """
    try:
        manifest = read_manifest(base_path)
    except Exception:
        # No manifest → no doc_id mapping; treat as nothing-to-chunk.
        return [], {}
    src_to_doc = dict(zip(
        manifest.column("source_hash").to_pylist(),
        manifest.column("doc_id").to_pylist(),
    ))

    elements_by_hash = _load_elements(base_path)
    overrides = load_overlay(base_path, text_source, required=True)
    inputs = []
    for source_hash, elements in elements_by_hash.items():
        apply_overlay(source_hash, elements, overrides)
        inputs.append(build_chunk_input(source_hash, elements, include_tables=chunk_tables_enabled))
    return inputs, src_to_doc


# ---------------------------------------------------------------------------
# AI-chunking reuse: rehydrate the enrich stage's persisted ILGS Documents
# ---------------------------------------------------------------------------


def _load_narrative_overrides(
    base_path: Path, text_source: str,
) -> dict[str, object] | None:
    """Load and rehydrate ``*.enrichment_doc.parquet`` Documents for a batch.

    Returns ``{source_hash: Document}`` for chunk_batch's ``narrative_overrides``
    (the byte-identity guard there decides whether each is actually reused), or
    ``None`` when no sidecar exists / the SDK is unavailable — both cases fall
    back to semchunk self-enrich. The stamped ``text_source`` is used only as a
    cheap pre-filter; a stamp mismatch is logged but still passed through, since
    the authoritative guard is byte-identity of ``Document.text``.
    """
    stored = read_enrichment_docs(base_path)
    if not stored:
        return None
    try:
        from isaacus.types.ilgs.v1.document import Document
    except ImportError:
        logger.warning(
            "chunk reuse: isaacus SDK not importable; cannot rehydrate "
            "persisted Documents for %s — self-enriching instead.", base_path.stem,
        )
        return None

    overrides: dict[str, object] = {}
    for source_hash, (stamp, doc_json) in stored.items():
        if not doc_json:
            continue
        if stamp and stamp != text_source:
            logger.info(
                "chunk reuse: %s persisted under text_source=%r but chunking "
                "under %r; byte-identity guard will decide.",
                source_hash, stamp, text_source,
            )
        try:
            overrides[source_hash] = Document.model_validate_json(doc_json)
        except Exception as e:  # corrupt blob — skip, fall back per doc
            logger.warning(
                "chunk reuse: could not rehydrate Document for %s (%s); "
                "self-enriching instead.", source_hash, e,
            )
    return overrides or None


# ---------------------------------------------------------------------------
# Element materialisation
# ---------------------------------------------------------------------------


def _bbox_from_row(raw: dict | None) -> BBox | None:
    if raw is None:
        return None
    return BBox(x=raw["x"], y=raw["y"], width=raw["width"], height=raw["height"])


def _load_elements(base_path: Path) -> dict[str, list[Element]]:
    """Read elements + table_cells from a shard and return per-source elements.

    Cells are stitched back onto their ``kind='table'`` parent via
    ``(source_hash, parent_elem_order)``. Form fields are not needed for
    chunking (no FieldEntry restoration here) — they live in their own
    sidecar consumed by other stages.
    """
    elem_table = read_elements(base_path)
    if elem_table.num_rows == 0:
        return {}

    cells_table = read_table_cells(base_path)
    cells_by_parent: dict[tuple[str, int], list[Cell]] = defaultdict(list)
    for row in cells_table.to_pylist():
        key = (row["source_hash"], row["parent_elem_order"])
        cells_by_parent[key].append(Cell(
            row=row["row"],
            col=row["col"],
            value=row["value"] or "",
            rowspan=row["rowspan"] or 1,
            colspan=row["colspan"] or 1,
            value_type=row["value_type"] or "text",
            bbox=_bbox_from_row(row.get("bbox")),
        ))

    out: dict[str, list[Element]] = defaultdict(list)
    for row in elem_table.to_pylist():
        bbox = _bbox_from_row(row.get("bbox"))
        elem = Element(
            order=row["elem_order"],
            kind=row["kind"],
            extractor=row["extractor"] or "",
            confidence=row["confidence"] or 0.0,
            page=row["page"],
            bbox=bbox,
            text=row["text"],
            cells=cells_by_parent.get((row["source_hash"], row["elem_order"])),
            header_rows=row["header_rows"],
            fields=None,
            alt_text=row["alt_text"],
            sheet=row["sheet"],
            row=row["row"],
            col=row["col"],
            value=row["value"],
            value_type=row["value_type"],
            formula=row["formula"],
            number_format=row["number_format"],
            merge_range=row["merge_range"],
            meta=dict(row["meta"]) if row["meta"] else {},
        )
        out[row["source_hash"]].append(elem)

    for elements in out.values():
        elements.sort(key=lambda e: e.order)
    return dict(out)


# ---------------------------------------------------------------------------
# Output rows
# ---------------------------------------------------------------------------


def _chunks_to_rows(
    chunks_by_hash: dict[str, list[TextChunk]],
) -> list[dict]:
    rows: list[dict] = []
    for source_hash, chunks in chunks_by_hash.items():
        for c in chunks:
            rows.append({
                "source_hash": source_hash,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "content_type": c.content_type,
                "has_redaction": c.has_redaction,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "elem_order": c.elem_order,
                "token_count": c.token_count,
            })
    return rows


# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    """Return True if every doc in this batch's manifest is in the checkpoint."""
    if chunks_path_for(base_path).exists():
        # The sidecar exists; treat checkpoint as authoritative. If the
        # checkpoint disagrees we still re-chunk (caller passes mgr
        # explicitly only when they want skip semantics).
        try:
            m = read_manifest(base_path)
        except Exception:
            return False
        doc_ids = m.column("doc_id").to_pylist()
        return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)
    return False


__all__ = ["CHUNKS_SUFFIX", "ChunkStageResult", "chunk_shards"]
