"""Per-stage Kanon-2 enrichment over an existing shard directory.

Consumes ``*.elements.parquet`` + ``*._manifest.parquet`` already written
by the extraction stage and writes ``*.enrichment_entities.parquet`` +
``*.enrichment_meta.parquet`` + ``*.graph_edges.parquet`` siblings per
batch, joinable to the other sidecars on ``source_hash``. The graph-edges
sidecar is the shippable enrichment graph (relations, hierarchy, citations,
and — when a ``*.chunks.parquet`` sibling exists — mention→chunk edges).

Reassembles the document narrative as the ``\\n\\n``-joined text of
TEXT_KINDS elements in ``elem_order`` per source_hash via the shared
:func:`womblex.process.chunker.reassemble_narrative` — the same text the
chunker sees, by construction — and feeds it to the enrichment API.

Requests are **token-budget packed** (:mod:`womblex.utils.token_packer`):
docs are grouped into ``min(max_texts_per_request, token_budget)`` per
enrichment call — the 8-doc API cap and the per-request token ceiling that
rate limits bind on, whichever binds first. A doc over the budget is sent
solo; a doc over ``split_ceiling`` is split on structural boundaries into
sub-documents enriched separately and offset-merged
(:func:`womblex.analyse.enrich_merge.merge_segment_results`). A failed request
isolates to its group (those docs are left unprocessed so a resume retries
them), mirroring the corpus-wide "individual document failures shouldn't stop
the batch" policy. The downstream :mod:`womblex.link.stage` consumes the
entities sidecar.

Tables are enriched on their own: each table's markdown (the string the chunker
reads, via :func:`womblex.process.chunker.table_texts`) is a separate request
text, so narrative offsets do not move and a table mention's offsets index that
table's markdown (``text_layer='table_markdown'``). ``enrichment.include_tables``
switches the pass off.

Split documents are not persisted for AI-chunking reuse (no single ILGS
Document spans the full narrative); the chunk stage self-enriches those few
long-tail docs. All other docs persist normally when ``persist_document``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from womblex.analyse.enrich import enrich_documents_raw
from womblex.analyse.enrich_merge import merge_segment_results
from womblex.analyse.graph import build_document_graph
from womblex.analyse.models import EnrichmentResult
from womblex.config import EnrichmentConfig
from womblex.ingest.elements import Element
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.process.chunker import TableText, TextChunk, reassemble_narrative, table_texts
from womblex.process.text_overlay import apply_overlay, load_overlay, require_overlays
from womblex.store.checkpoint import CheckpointManager
from womblex.store.enrichment_doc import (
    enrichment_doc_path_for,
    write_enrichment_doc_shard,
)
from womblex.store.enrichment_output import (
    TABLES_ENRICHED_KEY,
    TableEnrichment,
    enrichment_entities_path_for,
    enrichment_meta_path_for,
    graph_edges_path_for,
    write_enrichment_entities_shard,
    write_enrichment_meta_shard,
    write_graph_edges_shard,
)
from womblex.store.output import chunks_path_for, read_chunks, read_elements, read_manifest
from womblex.utils.token_packer import (
    TextSegment,
    TokenCounter,
    pack_by_tokens,
    split_on_boundaries,
)

logger = logging.getLogger(__name__)


@dataclass
class EnrichStageResult:
    batches_written: int
    docs_enriched: int
    total_entities: int
    narrative_tokens: int = 0
    table_tokens: int = 0


def enrich_shards(
    shard_dir: Path,
    enrichment_config: EnrichmentConfig,
    *,
    client: object,
    text_source: str = "elements",
    persist_document: bool = False,
    checkpoint_mgr: CheckpointManager | None = None,
    token_counter: TokenCounter | None = None,
) -> EnrichStageResult:
    """Enrich every batch in ``shard_dir`` and write enrichment siblings.

    ``client`` is an ``isaacus.Isaacus`` instance (injected so tests can
    pass a fake). Skips batches whose entities sidecar already exists when
    ``checkpoint_mgr`` is provided and every contained doc is checkpointed.

    When ``persist_document`` is true, also writes a
    ``*.enrichment_doc.parquet`` sibling holding the raw ILGS Document per
    doc (stamped with ``text_source``) so the chunk stage can reuse it for
    semchunk-4 AI chunking without re-enriching (``docs/decisions.md``).
    """
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("enrich_shards: no batches found in %s", shard_dir)
        return EnrichStageResult(0, 0, 0)
    require_overlays(bases, text_source)

    counter = token_counter or TokenCounter(enrichment_config.tokenizer)

    batches_written = 0
    docs_enriched = 0
    total_entities = 0
    narrative_tokens = 0
    table_tokens = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(
            base, checkpoint_mgr, persist_document=persist_document,
            include_tables=enrichment_config.include_tables,
        ):
            logger.info("enrich_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        narratives, doc_ids_by_hash = _load_narratives(base, text_source)
        min_chars = max(1, enrichment_config.skip_short_documents)
        packable = [
            (h, t) for h, t in narratives.items() if len(t.strip()) >= min_chars
        ]
        tables = _load_tables(base) if enrichment_config.include_tables else {}
        table_items = [
            (_table_key(h, i), t.markdown)
            for h, ts in tables.items() for i, t in enumerate(ts)
        ]
        narrative_tokens += sum(counter.count_batch([t for _, t in packable]))
        table_tokens += sum(counter.count_batch([t for _, t in table_items]))

        all_results, doc_rows, errored_keys = _enrich_packed(
            packable + table_items, client, enrichment_config, counter,
            text_source=text_source, persist_document=persist_document,
            doc_ids_by_hash=doc_ids_by_hash,
        )
        results = [(k, r) for k, r in all_results if _SEP not in k]
        doc_rows = [row for row in doc_rows if _SEP not in row[0]]
        table_results = _table_enrichments(all_results, tables)
        errored = {k.split(_SEP)[0] for k in errored_keys}
        for _src, enr in results:
            total_entities += _entity_total(enr)
        total_entities += sum(_entity_total(t.result) for t in table_results)

        write_enrichment_entities_shard(
            results, base, text_layer=text_source, tables=table_results)
        write_enrichment_meta_shard(results, base, table_counts={
            h: len(ts) for h, ts in tables.items() if h not in errored
        } if enrichment_config.include_tables else None)
        chunks_by_hash = _load_chunks(base)
        write_graph_edges_shard(
            [
                (src, build_document_graph(src, enr, chunks_by_hash.get(src)))
                for src, enr in results
            ],
            base,
        )
        if persist_document:
            write_enrichment_doc_shard(doc_rows, base)
        batches_written += 1
        docs_enriched += len(results)

        if checkpoint_mgr is not None and doc_ids_by_hash:
            # Checkpoint every doc EXCEPT those that errored — an errored doc
            # (transient API/network failure) stays unprocessed so a resume
            # retries it rather than skipping it forever. The batch's sidecar
            # is rewritten on that resume.
            resolved_doc_ids = [
                doc_id for h, doc_id in doc_ids_by_hash.items() if h not in errored
            ]
            if resolved_doc_ids:
                checkpoint_mgr.update(
                    doc_ids=resolved_doc_ids,
                    succeeded=len([h for h, _ in results if h not in errored]),
                    failed=len(errored),
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info(
            "enrich_shards: %s enriched %d docs, %d table(s)",
            base.stem, len(results), len(table_results),
        )

    return EnrichStageResult(
        batches_written=batches_written,
        docs_enriched=docs_enriched,
        total_entities=total_entities,
        narrative_tokens=narrative_tokens,
        table_tokens=table_tokens,
    )


@dataclass
class EnrichPlan:
    """What `enrich_shards` would send, counted offline (no client, no request)."""

    documents: int = 0
    tables: int = 0
    narrative_tokens: int = 0
    table_tokens: int = 0
    requests: int = 0
    split_documents: int = 0

    @property
    def tokens(self) -> int:
        return self.narrative_tokens + self.table_tokens


def plan_enrichment(
    shard_dir: Path,
    enrichment_config: EnrichmentConfig,
    *,
    text_source: str = "elements",
    token_counter: TokenCounter | None = None,
) -> EnrichPlan:
    """Count the requests and tokens `enrich_shards` would spend over *shard_dir*.

    Same texts and packing as the stage, so the counts are the run's own. Raises
    `MissingOverlayError` as the stage does. Never calls the API.
    """
    bases = _batch_bases(shard_dir)
    require_overlays(bases, text_source)
    counter = token_counter or TokenCounter(enrichment_config.tokenizer)
    plan = EnrichPlan()
    min_chars = max(1, enrichment_config.skip_short_documents)
    for base in bases:
        narratives, _ = _load_narratives(base, text_source)
        packable = [(h, t) for h, t in narratives.items() if len(t.strip()) >= min_chars]
        tables = _load_tables(base) if enrichment_config.include_tables else {}
        table_items = [
            (_table_key(h, i), t.markdown) for h, ts in tables.items() for i, t in enumerate(ts)
        ]
        plan.documents += len(packable)
        plan.tables += len(table_items)
        for group in pack_by_tokens(
            packable + table_items, counter.count_batch,
            max_items=enrichment_config.max_texts_per_request,
            token_budget=enrichment_config.token_budget,
        ):
            if group.solo and group.total_tokens > enrichment_config.split_ceiling:
                plan.split_documents += 1
                plan.requests += len(split_on_boundaries(
                    group.items[0].text, counter.count_batch,
                    max_tokens=enrichment_config.split_ceiling,
                ))
            else:
                plan.requests += 1
            for it in group.items:
                if _SEP in it.key:
                    plan.table_tokens += it.tokens
                else:
                    plan.narrative_tokens += it.tokens
    return plan


# ---------------------------------------------------------------------------
# Token-budgeted request packing + long-doc split
# ---------------------------------------------------------------------------


def _enrich_packed(
    packable: list[tuple[str, str]],
    client: object,
    config: EnrichmentConfig,
    counter: TokenCounter,
    *,
    text_source: str,
    persist_document: bool,
    doc_ids_by_hash: dict[str, str],
) -> tuple[list[tuple[str, EnrichmentResult]], list[tuple[str, str, str]], set[str]]:
    """Pack ``(source_hash, narrative)`` docs into requests and enrich each.

    Returns ``(results, doc_rows, errored)``. A request's failure marks every
    doc in that group errored — left unprocessed so a resume retries it. A doc
    over ``split_ceiling`` is split + offset-merged and never persisted for
    AI-chunking reuse (no single Document spans the full narrative).
    """
    results: list[tuple[str, EnrichmentResult]] = []
    doc_rows: list[tuple[str, str, str]] = []
    errored: set[str] = set()

    groups = pack_by_tokens(
        packable, counter.count_batch,
        max_items=config.max_texts_per_request,
        token_budget=config.token_budget,
    )
    for group in groups:
        keys = [it.key for it in group.items]
        try:
            if group.solo and group.total_tokens > config.split_ceiling:
                src = group.items[0].key
                results.append((src, _enrich_split_doc(group.items[0].text, client, config, counter)))
            else:
                pairs = enrich_documents_raw(
                    [it.text for it in group.items], client,
                    model=config.model,
                    overflow_strategy=config.overflow_strategy,
                    max_retries=config.max_retries,
                    retry_base_delay=config.retry_base_delay,
                )
                for it, (enr, raw_doc) in zip(group.items, pairs):
                    results.append((it.key, enr))
                    if persist_document:
                        # raw_doc is the SDK ILGS Document; model_dump_json round-trips it.
                        doc_rows.append((it.key, text_source, raw_doc.model_dump_json()))  # type: ignore[attr-defined]
        except Exception as e:  # transient (network/429-exhausted) — leave for retry
            logger.error(
                "enrich_shards: enrichment failed for %s: %s",
                [doc_ids_by_hash.get(k.split(_SEP)[0], k) for k in keys], e,
            )
            errored.update(keys)
            continue
    return results, doc_rows, errored


def _enrich_split_doc(
    text: str, client: object, config: EnrichmentConfig, counter: TokenCounter,
) -> EnrichmentResult:
    """Split an over-ceiling doc on boundaries, enrich each segment, offset-merge."""
    segments = split_on_boundaries(text, counter.count_batch, max_tokens=config.split_ceiling)
    seg_results: list[tuple[TextSegment, EnrichmentResult]] = []
    for seg in segments:
        pairs = enrich_documents_raw(
            [seg.text], client,
            model=config.model,
            overflow_strategy=config.overflow_strategy,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay,
        )
        seg_results.append((seg, pairs[0][0]))
    logger.info(
        "enrich_shards: split long doc (%d tokens) into %d segments",
        sum(s.tokens for s in segments), len(segments),
    )
    return merge_segment_results(text, seg_results)


# ---------------------------------------------------------------------------
# Narrative materialisation (text-only — no cells/fields needed for enrichment)
# ---------------------------------------------------------------------------


def _load_narratives(
    base_path: Path, text_source: str = "elements",
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``({source_hash: narrative}, {source_hash: doc_id})`` for a batch.

    When ``text_source`` is not ``'elements'`` the matching element-text overlay
    (normalise / spellfix) is applied before reassembly, so Kanon-2 enriches the
    same repaired text the chunker chunks — keeping mention/chunk offsets aligned.
    """
    try:
        manifest = read_manifest(base_path)
    except Exception:
        return {}, {}
    src_to_doc = dict(zip(
        manifest.column("source_hash").to_pylist(),
        manifest.column("doc_id").to_pylist(),
    ))

    elem_table = read_elements(base_path)
    if elem_table.num_rows == 0:
        return {}, src_to_doc

    by_hash: dict[str, list[Element]] = defaultdict(list)
    for row in elem_table.to_pylist():
        by_hash[row["source_hash"]].append(Element(
            order=row["elem_order"],
            kind=row["kind"],
            extractor=row["extractor"] or "",
            page=row["page"],
            text=row["text"],
        ))

    overrides = load_overlay(base_path, text_source, required=True)
    narratives: dict[str, str] = {}
    for src, elems in by_hash.items():
        elems.sort(key=lambda e: e.order)
        apply_overlay(src, elems, overrides)
        text, _ = reassemble_narrative(elems)
        narratives[src] = text
    return narratives, src_to_doc


_SEP = "\x1f"


def _table_key(source_hash: str, ordinal: int) -> str:
    """Request key for one table: a table shares its document's ``source_hash``."""
    return f"{source_hash}{_SEP}{ordinal}"


def _load_tables(base_path: Path) -> dict[str, list[TableText]]:
    """Per-document table markdowns — the strings the chunker reads."""
    out = {src: table_texts(elems) for src, elems in _load_elements(base_path).items()}
    return {src: ts for src, ts in out.items() if ts}


def _table_enrichments(
    results: list[tuple[str, EnrichmentResult]], tables: dict[str, list[TableText]],
) -> list[TableEnrichment]:
    out: list[TableEnrichment] = []
    for key, enr in results:
        if _SEP not in key:
            continue
        src, ordinal = key.split(_SEP)
        t = tables[src][int(ordinal)]
        out.append(TableEnrichment(src, int(ordinal), t.elem_order, t.sheet, enr))
    return out


def _entity_total(enr: EnrichmentResult) -> int:
    return len(enr.persons) + len(enr.locations) + len(enr.terms) + len(enr.external_documents)


def _load_chunks(base_path: Path) -> dict[str, list[TextChunk]]:
    """Narrative chunks per source_hash from the batch's chunks sidecar, if present.

    Entity mention spans are offsets into the reassembled narrative, so only
    narrative chunks (which share that coordinate space) feed the graph's
    chunk-mention edges; table chunks carry offsets in their own markdown space.
    """
    if not chunks_path_for(base_path).exists():
        return {}
    try:
        chunk_rows = read_chunks(base_path).to_pylist()
    except Exception as e:  # unreadable sidecar — graph still ships, minus chunk edges
        logger.warning("enrich_shards: unreadable chunks sidecar for %s: %s", base_path.stem, e)
        return {}
    by_hash: dict[str, list[TextChunk]] = defaultdict(list)
    for row in chunk_rows:
        if row["content_type"] != "narrative":
            continue
        by_hash[row["source_hash"]].append(TextChunk(
            text=row["text"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            chunk_index=row["chunk_index"],
            content_type=row["content_type"],
        ))
    return by_hash


def _all_docs_checkpointed(
    base_path: Path, mgr: CheckpointManager, *, persist_document: bool = False,
    include_tables: bool = False,
) -> bool:
    """True if the entities sidecar exists and every manifest doc is checkpointed.

    The graph-edges sidecar must also exist — a batch enriched before graph
    shipping landed is re-enriched on resume so it gains its
    ``*.graph_edges.parquet`` (mirroring ``persist_document``: the graph is
    only buildable from the live EnrichmentResult).

    When ``include_tables`` is on, the meta sidecar's footer must record that
    tables were sent — a batch enriched without them is re-enriched on resume.

    When ``persist_document`` is requested, the doc sidecar must also exist —
    otherwise a batch enriched before the flag was enabled would be skipped on
    resume and never gain its ``*.enrichment_doc.parquet``.
    """
    if not enrichment_entities_path_for(base_path).exists():
        return False
    if not graph_edges_path_for(base_path).exists():
        return False
    if persist_document and not enrichment_doc_path_for(base_path).exists():
        return False
    if include_tables:
        meta = enrichment_meta_path_for(base_path)
        if not meta.exists() or TABLES_ENRICHED_KEY not in (pq.read_schema(str(meta)).metadata or {}):
            return False
    try:
        m = read_manifest(base_path)
    except Exception:
        return False
    doc_ids = m.column("doc_id").to_pylist()
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["EnrichStageResult", "enrich_shards"]
