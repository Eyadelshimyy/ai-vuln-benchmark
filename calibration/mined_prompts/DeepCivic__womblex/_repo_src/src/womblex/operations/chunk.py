"""run_chunking — split extracted text into token-bounded chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from womblex.config import WomblexConfig
from womblex.operations.models import DocumentResult
from womblex.process.chunker import (
    build_chunk_input,
    chunk_batch,
    create_chunker,
    resolve_tokenizer,
)
from womblex.redact.stage import annotate_chunks
from womblex.utils.availability import isaacus_available, tokenizer_available
from womblex.utils.isaacus_client import make_ai_chunking_client

if TYPE_CHECKING:
    from womblex.ingest.extract import ExtractionResult

logger = logging.getLogger(__name__)


def run_chunking(
    results: list[DocumentResult], config: WomblexConfig,
) -> list[DocumentResult]:
    """Split extracted text into token-bounded chunks.

    Reassembles narrative + tables from each result's element stream
    (the canonical source of truth) and feeds every result into a
    single :func:`chunk_batch` call so semchunk's ``processes`` and
    progress arguments parallelise across the whole batch. Applies
    ``flag``-mode redaction annotation to chunks when a redaction
    report exists on the extraction.
    """
    if not config.chunking.enabled:
        return results

    # Resolved first so an unknown tokeniser name is an error either way.
    tokenizer = resolve_tokenizer(
        config.chunking.tokenizer, config.chunking.tokenizer_options
    )

    # AI chunking calls the enricher API; plain token chunking uses the vendored
    # tokeniser and runs offline. Gate each on what it actually needs so a
    # keyless local run still chunks (mirrors process/chunk_stage.py).
    if config.chunking.chunking_model:
        if not isaacus_available():
            logger.warning(
                "run_chunking: AI chunking is enabled (chunking.chunking_model=%r) "
                "but Isaacus is not available (needs the isaacus SDK + "
                "ISAACUS_API_KEY, or ISAACUS_SAGEMAKER_ENDPOINTS) — skipping "
                "chunking. Unset chunking.chunking_model to chunk offline with "
                "the local tokeniser.", config.chunking.chunking_model,
            )
            return results
    elif not tokenizer_available(tokenizer):
        logger.warning(
            "run_chunking: chunk-size tokeniser %r is not resolvable locally "
            "(needs `transformers` plus a bundled copy under _models/ or "
            "WOMBLEX_MODELS_DIR) — skipping chunking. The default "
            "kanon-2-tokenizer is vendored; a custom tokenizer must be bundled "
            "to keep chunking offline.", config.chunking.tokenizer,
        )
        return results

    chunker = create_chunker(
        tokenizer=tokenizer,
        chunk_size=config.chunking.chunk_size,
        chunking_model=config.chunking.chunking_model,
        isaacus_client=make_ai_chunking_client(config.chunking.chunking_model),
        tokenizer_kwargs=config.chunking.tokenizer_kwargs,
        memoize=config.chunking.memoize,
        cache_maxsize=config.chunking.cache_maxsize,
        max_token_chars=config.chunking.max_token_chars,
    )

    chunk_cfg = config.chunking
    eligible: list[tuple[DocumentResult, ExtractionResult]] = [
        (dr, dr.extraction) for dr in results
        if dr.status == "completed" and dr.extraction is not None
    ]
    if not eligible:
        return results

    inputs = [
        build_chunk_input(
            source_hash=dr.doc_id,
            elements=extraction.elements,
            include_tables=chunk_cfg.chunk_tables,
        )
        for dr, extraction in eligible
    ]

    chunks_by_doc = chunk_batch(
        inputs,
        chunker,
        overlap=chunk_cfg.overlap,
        processes=chunk_cfg.processes,
        progress=chunk_cfg.progress,
    )

    for dr, extraction in eligible:
        dr.chunks = chunks_by_doc.get(dr.doc_id, [])
        redaction_report = extraction.redaction_report
        if (
            config.redaction.enabled
            and config.redaction.mode == "flag"
            and redaction_report
        ):
            dr.chunks = annotate_chunks(dr.chunks, redaction_report)
        logger.debug(
            "chunked: doc=%s chunks=%d narrative=%d table=%d",
            dr.doc_id,
            len(dr.chunks),
            sum(1 for c in dr.chunks if c.content_type == "narrative"),
            sum(1 for c in dr.chunks if c.content_type == "table"),
        )

    return results
